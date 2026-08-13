// Excluded from WebGL player builds: depends on the native MediaPipe plugin (Mediapipe.Runtime
// has no wasm library, so IL2CPP linking fails). Kept for the Editor regardless of build target.
#if !UNITY_WEBGL || UNITY_EDITOR
using Mediapipe.Unity;
using Mediapipe.Unity.FaceMesh;
using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// Native (Windows/macOS/Linux) gaze provider: the homuler MediaPipe FaceMesh for landmarks +
    /// the EyeMU model on Unity's Inference Engine (via HomulerEyeMURunner), plus HomulerEyeHelper for
    /// blink/drowsy/distance. This is the desktop implementation of IGazeProvider; it cannot run on WebGL
    /// (native plugin), which is exactly why the seam exists.
    /// </summary>
    public class NativeGazeProvider : IGazeProvider, IGazeRecordingSource
    {
        private readonly FaceMeshSolution _faceMesh;
        private readonly WebCamSource _webcam;
        private IGazeBackbone _backbone;   // not readonly: SetBackbone swaps it at runtime
        private readonly HomulerEyeHelper _eyeHelper;

        private Vector2 _rawGaze;
        private bool _isBlinking;
        private bool _isDrowsy;
        private float _distanceMm = -1000f;
        //EyeFeature() is 8 distance calcs + trig; compute it once per Tick and reuse for blink, drowsy
        //and the CSV EyeFeature accessor instead of recomputing it 3x per frame.
        private float _eyeFeature = float.NaN;

        //Whether the backbones pipeline their GPU readbacks (see HomulerGaze._asyncGpuReadback).
        private readonly bool _asyncReadback;
        //Whether the direction backbones run horizontal-flip test-time augmentation (sync mode only).
        private readonly bool _flipAugmentation;
        //Whether the direction backbones roll-normalize their face crop (2D data normalization).
        private readonly bool _rollNormalize;
        //Delivered camera rate, used to warn when the webcam — not inference — is the bottleneck.
        //Gaze can only update as often as the camera produces frames (see the didUpdateThisFrame gate
        //in Tick), so a camera running slowly caps the whole pipeline no matter how fast the machine is.
        private float _camRateWindowStart;
        private int _camFramesInWindow;
        private bool _lowFrameRateWarned;
        //Below this fraction of the requested rate the camera is considered the limiting factor.
        private const float LowFrameRateFraction = 0.5f;
        //Do not warn before this many seconds of samples: startup, autofocus and auto-exposure settling
        //all produce a legitimately slow first second.
        private const float RateWarnAfterSeconds = 3f;

        //eyeBlink blendshape score above which the frame counts as a blink (both eyes maxed). The
        //blendshape gate replaces the EAR heuristic when blendshapes are available: the attention model
        //separates lid closure from downward gaze, which EAR conflates (downward gaze looked like a blink).
        private const float BlinkBlendshapeThreshold = 0.5f;

        public NativeGazeProvider(GameObject mediaPipeGO, GazeBackbone backbone = GazeBackbone.EyeMU,
            bool asyncGpuReadback = false, bool flipAugmentation = false, bool rollNormalize = true)
        {
            //Validate the MediaPipe GameObject up front. Without this the missing component surfaced as a
            //bare NullReferenceException inside the constructor, which left HomulerGaze._provider null and
            //turned into one NRE per frame out of LateUpdate — noise that says nothing about the real cause.
            //(The MediaPipe 0.16.3 Task-API migration stripped the dead Solution-era components from the
            //scenes; a scene that never got FaceMeshSolution/WebCamSource re-added lands exactly here.)
            if (mediaPipeGO == null)
                throw new MissingComponentException(
                    "UnitEye: HomulerGaze._mediaPipeGO is not assigned. Point it at a GameObject carrying " +
                    "FaceMeshSolution + WebCamSource (see the UnitEyeUsingHomulerMediapipe prefab).");

            _webcam = mediaPipeGO.GetComponent<WebCamSource>();
            _faceMesh = mediaPipeGO.GetComponent<FaceMeshSolution>();
            if (_webcam == null || _faceMesh == null)
            {
                var missing = _faceMesh == null
                    ? (_webcam == null ? "FaceMeshSolution and WebCamSource" : "FaceMeshSolution")
                    : "WebCamSource";
                throw new MissingComponentException(
                    $"UnitEye: GameObject '{mediaPipeGO.name}' (HomulerGaze._mediaPipeGO) is missing {missing}. " +
                    "Add the component(s) to it, or replace it with the UnitEyeUsingHomulerMediapipe prefab.");
            }

            _eyeHelper = new HomulerEyeHelper(_faceMesh, _webcam.name);
            _asyncReadback = asyncGpuReadback;
            _flipAugmentation = flipAugmentation;
            _rollNormalize = rollNormalize;

            //Pick the gaze model behind the shared face-mesh/blink/distance stack.
            _backbone = CreateBackbone(backbone);
        }

        private IGazeBackbone CreateBackbone(GazeBackbone backbone)
        {
            switch (backbone)
            {
                case GazeBackbone.GazeMobileOne:
                    return new GazeEstimationRunner(_faceMesh, "ONNX/GazeEstimation/mobileone_s0_gaze", _asyncReadback, _flipAugmentation, _rollNormalize);
                case GazeBackbone.GazeMobileNetV2:
                    return new GazeEstimationRunner(_faceMesh, "ONNX/GazeEstimation/mobilenetv2_gaze", _asyncReadback, _flipAugmentation, _rollNormalize);
                case GazeBackbone.GazeResNet34:
                    return new GazeEstimationRunner(_faceMesh, "ONNX/GazeEstimation/resnet34_gaze", _asyncReadback, _flipAugmentation, _rollNormalize);
                case GazeBackbone.EyeMUPlusResNet34:
                    //Ensemble: both models every frame, concatenated calibration features (~2x inference cost).
                    return new CompositeGazeBackbone(
                        new HomulerEyeMURunner(_faceMesh, _asyncReadback),
                        new GazeEstimationRunner(_faceMesh, "ONNX/GazeEstimation/resnet34_gaze", _asyncReadback, _flipAugmentation, _rollNormalize));
                default:
                    return new HomulerEyeMURunner(_faceMesh, _asyncReadback);
            }
        }

        //Swap the gaze model at runtime: dispose the old backbone and build the new one. The shared
        //face-mesh/blink/distance stack is unchanged. Calibration is per-backbone (different feature
        //vector), so RefineGazeLocation falls back to raw gaze until the new backbone is recalibrated.
        public void SetBackbone(GazeBackbone backbone)
        {
            _backbone?.Dispose();
            _backbone = CreateBackbone(backbone);
        }

        /// <summary>
        /// Warn once if the camera is delivering far fewer frames than requested.
        ///
        /// Gaze updates are gated on didUpdateThisFrame, so the delivered camera rate is a hard ceiling
        /// on the gaze rate — 5fps from the camera means 5 gaze updates per second regardless of GPU,
        /// model or frame rate. That failure mode is invisible from inside the app: everything reports
        /// healthy, the renderer is fast, inference is fast, and gaze simply feels laggy.
        ///
        /// It is also usually not a code problem. Webcams commonly drop to 5-7.5fps in low light because
        /// auto-exposure lengthens exposure time, and a requested width/fps combination the device does
        /// not support can silently fall back to a slow mode. Saying so directly turns an afternoon of
        /// profiling into "turn a light on".
        /// </summary>
        private void TrackCameraFrameRate()
        {
            _camFramesInWindow++;

            if (_camRateWindowStart <= 0f)
            {
                _camRateWindowStart = Time.unscaledTime;
                return;
            }

            float elapsed = Time.unscaledTime - _camRateWindowStart;
            if (elapsed < RateWarnAfterSeconds) return;

            float actual = _camFramesInWindow / elapsed;
            _camFramesInWindow = 0;
            _camRateWindowStart = Time.unscaledTime;

            int requested = _webcam != null ? _webcam.requestedFps : 0;
            if (requested <= 0 || _lowFrameRateWarned) return;

            if (actual < requested * LowFrameRateFraction)
            {
                _lowFrameRateWarned = true;
                Debug.LogWarning(
                    $"UnitEye: the webcam is delivering only {actual:F1} fps (requested {requested}). " +
                    "Gaze cannot update faster than the camera, so this — not inference speed — is the " +
                    "limiting factor. Most often this is low light (auto-exposure lengthens exposure " +
                    "time and drops the frame rate), a resolution/fps combination the device does not " +
                    "support, or another application holding the camera. Try brighter lighting or a " +
                    "lower requested resolution.");
            }
        }

        public bool Tick()
        {
            // FaceMeshSolution only updates landmarks when the webcam provides a new image. Running the
            // gaze model again between camera frames would turn one observation into several identical
            // calibration/evaluation samples and make the filter appear more responsive than the camera.
            if (_webcam == null || !_webcam.didUpdateThisFrame)
                return false;

            TrackCameraFrameRate();

            if (!_backbone.PerformInference(_webcam))
                return false;

            _rawGaze = _backbone.RawGaze;
            //Compute EyeFeature once, then derive blink/drowsy from it (was recomputed inside each call).
            _eyeFeature = _eyeHelper.EyeFeature();
            _isDrowsy = _eyeHelper.IsDrowsyFromFeature(_eyeFeature);
            //Blink gate: prefer the eyeBlink blendshapes (dedicated lid-closure signal; no downward-gaze
            //false positives, no per-user threshold calibration) and fall back to the EAR heuristic when
            //blendshapes are unavailable. Drowsiness stays on the EAR feature (its calibrated statistics
            //describe the smoothed EAR, not the blendshape).
            _isBlinking = _faceMesh != null && _faceMesh.HasBlendshapes
                ? Mathf.Max(_faceMesh.EyeBlinkLeft, _faceMesh.EyeBlinkRight) > BlinkBlendshapeThreshold
                : _eyeHelper.IsBlinkingFromFeature(_eyeFeature);
            _distanceMm = _eyeHelper.CalculateCamDistanceFocal();

            //Binocular consistency: the two eyes move conjugately, so their normalized iris offsets should
            //(nearly) agree. Disagreement is a free per-frame quality signal — it spikes on half-blinks,
            //partial occlusion and landmark failures that the blink gate misses. Exposed for capture
            //gates / logging / host-game confidence displays.
            HomulerFunctions.FillIrisFeatures(_faceMesh.FaceLandmarks, _irisScratch, 0);
            BinocularIrisDisagreement = new Vector2(_irisScratch[0] - _irisScratch[2],
                                                    _irisScratch[1] - _irisScratch[3]).magnitude;
            return true;
        }
        private readonly float[] _irisScratch = new float[4];

        public Vector2 RawGaze => _rawGaze;
        //Returns the backbone's reused feature buffer (no per-frame copy). Valid only until the next
        //Tick; the calibration capture, which retains samples, clones it (see HomulerGazeCalibration).
        public float[] GetFeatures() => _backbone.Features;
        public double CaptureTimestamp => _backbone.CaptureTimestamp;
        public float BinocularIrisDisagreement { get; private set; }
        public bool IsFacePresent => _faceMesh != null && _faceMesh.FaceLandmarks != null;
        public bool IsBlinking => _isBlinking;
        public bool IsDrowsy => _isDrowsy;
        public float DistanceMm => _distanceMm;
        public float EyeFeature => _eyeFeature;
        //Head pose comes from the shared FaceMesh, so it's the same regardless of gaze backbone.
        public Vector3 HeadPoseEuler => new Vector3(_faceMesh.HeadPitch, _faceMesh.HeadYaw, _faceMesh.HeadRoll);
        public RenderTexture LeftEyeTexture => _backbone.LeftEyeTexture;
        public RenderTexture RightEyeTexture => _backbone.RightEyeTexture;

        public bool AnnotateFaceMesh
        {
            get => _faceMesh != null && _faceMesh.Annotate;
            set { if (_faceMesh != null) _faceMesh.Annotate = value; }
        }

        //Forwards to the cached FaceMeshSolution (no per-toggle GetComponent; HomulerGaze goes through
        //the seam like it already does for AnnotateFaceMesh).
        public void SetRendering(bool rendering)
        {
            if (_faceMesh != null) _faceMesh.IsRendering = rendering;
        }

        public bool IsCalibratingDrowsy => _eyeHelper.Calibrating;
        public int DrowsyCalibrationCount => _eyeHelper.CalibrationCount;
        public void CalibrateDistance() => _eyeHelper.CalibrateFocalLength();
        public void CalibrateBlinking() => _eyeHelper.CalibrateBlinking();
        public void CalibrateDrowsy(bool calibrating) => _eyeHelper.CalibrateDrowsyStats(calibrating);

        public string CurrentCameraName => _webcam != null ? _webcam.sourceName : "";

        public void NextCamera()
        {
            if (_webcam == null) return;
            _webcam.SelectSource(_webcam.GetCameraIndex() + 1);
            _eyeHelper.CameraChanged(_webcam.sourceName);
        }

        public void PreviousCamera()
        {
            if (_webcam == null) return;
            _webcam.SelectSource(_webcam.GetCameraIndex() - 1);
            _eyeHelper.CameraChanged(_webcam.sourceName);
        }

        public void Dispose() => _backbone?.Dispose();

        #region IGazeRecordingSource

        //Async readback publishes frame N-1's gaze while the crop textures already hold frame N (see
        //IGazeBackbone), so imagery and features describe different moments. Report that honestly and let
        //the recorder refuse imagery rather than emit a dataset whose pixels and labels disagree.
        public bool ImageryInSyncWithFeatures => !_asyncReadback;

        public int LandmarkCount => _faceMesh != null && _faceMesh.FaceLandmarks != null
            ? _faceMesh.FaceLandmarks.Count : 0;

        public int TryCopyLandmarks(float[] dest) => _faceMesh != null ? _faceMesh.CopyLandmarks(dest) : 0;

        public bool TryCopyEyeBlendshapes(float[] dest) =>
            _faceMesh != null && _faceMesh.TryCopyEyeBlendshapes(dest);

        public int FrameWidth => _faceMesh != null ? _faceMesh.FrameWidth : 0;
        public int FrameHeight => _faceMesh != null ? _faceMesh.FrameHeight : 0;
        public Texture CameraTexture => _faceMesh != null ? _faceMesh.CameraTexture : null;
        public Rect FaceBoundsNormalized => _faceMesh != null ? _faceMesh.FaceBoundsNormalized : default;
        public bool LandmarksSmoothed => _faceMesh != null && _faceMesh.GazeLandmarksSmoothed;
        public bool FrameFlippedHorizontally => _faceMesh != null && _faceMesh.FrameFlippedHorizontally;
        public bool FrameFlippedVertically => _faceMesh != null && _faceMesh.FrameFlippedVertically;

        #endregion
    }
}
#endif
