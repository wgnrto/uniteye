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
    public class NativeGazeProvider : IGazeProvider
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

        public NativeGazeProvider(GameObject mediaPipeGO, GazeBackbone backbone = GazeBackbone.EyeMU)
        {
            _webcam = mediaPipeGO.GetComponent<WebCamSource>();
            _faceMesh = mediaPipeGO.GetComponent<FaceMeshSolution>();
            _eyeHelper = new HomulerEyeHelper(_faceMesh, _webcam.name);

            //Pick the gaze model behind the shared face-mesh/blink/distance stack.
            _backbone = CreateBackbone(backbone);
        }

        private IGazeBackbone CreateBackbone(GazeBackbone backbone)
        {
            switch (backbone)
            {
                case GazeBackbone.GazeMobileOne:
                    return new GazeEstimationRunner(_faceMesh, "ONNX/GazeEstimation/mobileone_s0_gaze");
                case GazeBackbone.GazeMobileNetV2:
                    return new GazeEstimationRunner(_faceMesh, "ONNX/GazeEstimation/mobilenetv2_gaze");
                case GazeBackbone.GazeResNet34:
                    return new GazeEstimationRunner(_faceMesh, "ONNX/GazeEstimation/resnet34_gaze");
                default:
                    return new HomulerEyeMURunner(_faceMesh);
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

        public bool Tick()
        {
            // FaceMeshSolution only updates landmarks when the webcam provides a new image. Running the
            // gaze model again between camera frames would turn one observation into several identical
            // calibration/evaluation samples and make the filter appear more responsive than the camera.
            if (_webcam == null || !_webcam.didUpdateThisFrame)
                return false;

            if (!_backbone.PerformInference(_webcam))
                return false;

            _rawGaze = _backbone.RawGaze;
            //Compute EyeFeature once, then derive blink/drowsy from it (was recomputed inside each call).
            _eyeFeature = _eyeHelper.EyeFeature();
            _isDrowsy = _eyeHelper.IsDrowsyFromFeature(_eyeFeature);
            _isBlinking = _eyeHelper.IsBlinkingFromFeature(_eyeFeature);
            _distanceMm = _eyeHelper.CalculateCamDistanceFocal();
            return true;
        }

        public Vector2 RawGaze => _rawGaze;
        //Returns the backbone's reused feature buffer (no per-frame copy). Valid only until the next
        //Tick; the calibration capture, which retains samples, clones it (see HomulerGazeCalibration).
        public float[] GetFeatures() => _backbone.Features;
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
    }
}
#endif
