// Rewritten for the MediaPipe Unity Plugin 0.16.3 Task API (FaceLandmarker).
// The 0.15+ releases removed the legacy Solution API (ImageSourceSolution / GraphRunner / typed
// packets) this used to build on, so this is now a self-contained facade over Tasks.Vision.FaceLandmarker
// that keeps the SAME public surface the UnitEye pipeline consumes (FaceLandmarks, iris, head pose,
// EyeCorners, Annotate/IsRendering) — see docs/HOMULER-UPGRADE.md.
#if !UNITY_WEBGL || UNITY_EDITOR
using System.Collections.Generic;
using System.IO;
using Mediapipe.Tasks.Core;
using Mediapipe.Tasks.Vision.Core;
using Mediapipe.Tasks.Vision.FaceLandmarker;
using Mediapipe.Unity.Experimental;
using UnityEngine;

namespace Mediapipe.Unity.FaceMesh
{
    public class FaceMeshSolution : MonoBehaviour
    {
        // The .task bundle for the Task API (self-contained: detector + landmarker + iris/attention).
        // Copied into StreamingAssets by MediaPipeAssetInstaller.
        public const string ModelFileName = "face_landmarker_v2.bytes";

        // face_landmarker_v2 outputs 478 landmarks: 0..467 face mesh, 468..472 left-eye iris, 473..477 right.
        private const int FaceLandmarkCount = 468;
        private const int LeftIrisStart = 468;
        private const int RightIrisStart = 473;
        private const int IrisCount = 5;
        private const int TotalLandmarks = 478;

        [SerializeField] private WebCamSource _webCamSource;
        [SerializeField] private bool _annotate = true;
        [SerializeField] private int maxNumFaces = 1;
        // Unity textures are bottom-left origin; MediaPipe expects top-left. Flip vertically so the
        // landmarks come back in MediaPipe (y-down, top-left) convention, matching the old pipeline.
        // If gaze is upside-down/mirrored on your webcam, flip these (a hand-test knob).
        [SerializeField] private bool _flipHorizontally = false;
        [SerializeField] private bool _flipVertically = true;

        // The Task API dropped the old graph's landmark-smoothing calculators, so the raw landmarks jitter
        // frame to frame — which shakes the eye crops, the EyeCorners model input and the iris gaze
        // features. Smooth ONLY the six gaze-relevant landmarks (4 eye corners + 2 iris centers) with a
        // light One-Euro filter: fixation jitter is damped, fast head/eye motion passes through. Head-pose
        // and eyelid landmarks are untouched.
        [Tooltip("One-Euro-smooth the 6 gaze landmarks (eye corners + iris centers) to reduce crop/feature jitter. Off = raw Task-API landmarks.")]
        [SerializeField] private bool _smoothGazeLandmarks = true;

        [Header("Debug preview (IMGUI)")]
        // The old Solution API showed the webcam on a Canvas "Screen" RawImage with the facemesh drawn on
        // top; the Task-API migration deleted that display stack (and the now-dead white RawImage was
        // disabled in MediapipeAnnotation.prefab). We redraw an equivalent preview here in IMGUI, where we
        // already hold the webcam texture and the 478 landmarks — no Canvas / annotation-controller /
        // coordinate-marshaling dependencies. Camera + dots share one flip/mirror transform so they stay
        // aligned; the knobs are serialized because correct orientation is a per-webcam hand-test.
        [Tooltip("Draw the live webcam + facemesh landmark overlay in IMGUI. Turn off to see the scene behind it.")]
        [SerializeField] private bool _drawPreview = true;
        [Tooltip("Flip the whole preview (camera + dots) vertically. Toggle if the view is upside-down.")]
        [SerializeField] private bool _previewFlipVertically = false;
        [Tooltip("Mirror the whole preview (camera + dots) horizontally, e.g. a selfie view.")]
        [SerializeField] private bool _previewMirror = false;
        [Tooltip("Flip ONLY the landmark dots vertically (not the camera). Use if dots sit upside-down on the image.")]
        [SerializeField] private bool _overlayFlipVertically = false;
        // Bare Color/Rect/ScaleMode here would bind to the protobuf Mediapipe.* types (the enclosing
        // namespace shadows both the UnityEngine using and any file-scope alias), so qualify them.
        [Tooltip("Facemesh landmark dot color.")]
        [SerializeField] private UnityEngine.Color _overlayColor = new UnityEngine.Color(0f, 1f, 0f, 0.85f);
        [Tooltip("Landmark dot size in pixels at 1080p (scales up on high-DPI).")]
        [SerializeField] private float _overlayDotSize = 3f;
        [Tooltip("IMGUI draw order; higher = drawn behind the eye-crop/crosshair/Gaze-UI overlays.")]
        [SerializeField] private int _previewGuiDepth = 5;

        private FaceLandmarker _faceLandmarker;
        private TextureFramePool _textureFramePool;
        private int _poolWidth, _poolHeight;
        private FaceLandmarkerResult _result;
        private readonly System.Diagnostics.Stopwatch _stopwatch = new System.Diagnostics.Stopwatch();
        private bool _warnedRotation;

        // Gaze-landmark smoothing state. This assembly cannot reference UnitEye.OneEuroFilter (the UnitEye
        // runtime assembly references THIS one), so a minimal scalar One-Euro lives here. Parameters are in
        // NORMALIZED landmark units: velocities are ~100-1000x smaller than the pixel-space values the
        // classic 1-euro defaults were tuned for, hence the much larger beta. Filters reset on face loss so
        // a reacquired face doesn't get dragged from its last position.
        private const float LandmarkMinCutoff = 1.5f;   // Hz: fixation-jitter damping floor
        private const float LandmarkBeta = 5f;          // opens the cutoff during fast (saccade/head) motion
        private const float LandmarkDCutoff = 1f;
        private static readonly int[] SmoothedLandmarkIndices = { 362, 263, 33, 133, LeftIrisStart, RightIrisStart };
        private readonly OneEuroScalar[] _landmarkFilters = new OneEuroScalar[SmoothedLandmarkIndices.Length * 2];
        private long _lastLandmarkTimestampMs = -1;

        private struct OneEuroScalar
        {
            private float _value, _derivative;
            private bool _initialized;

            public void Reset() => _initialized = false;

            public float Filter(float sample, float dt)
            {
                if (!_initialized || dt <= 0f)
                {
                    _initialized = true;
                    _value = sample;
                    _derivative = 0f;
                    return sample;
                }
                float rawDerivative = (sample - _value) / dt;
                _derivative += Alpha(LandmarkDCutoff, dt) * (rawDerivative - _derivative);
                float cutoff = LandmarkMinCutoff + LandmarkBeta * Mathf.Abs(_derivative);
                _value += Alpha(cutoff, dt) * (sample - _value);
                return _value;
            }

            private static float Alpha(float cutoff, float dt)
            {
                float tau = 1f / (2f * Mathf.PI * cutoff);
                return 1f / (1f + tau / dt);
            }
        }
        // Face landmark bbox, computed once per detected frame in CopyLandmarks (which already iterates
        // all landmarks) instead of per HeadArea access — HeadArea is read 2-4x per frame downstream.
        private float _bboxMinX, _bboxMinY, _bboxMaxX, _bboxMaxY;

        // Reusable Mediapipe.NormalizedLandmark objects so we expose the SAME type the consumers already
        // use (.X/.Y/.Z) without allocating 478 objects every frame — we mutate them in place.
        private readonly List<NormalizedLandmark> _mpLandmarks = new List<NormalizedLandmark>(TotalLandmarks);
        private readonly List<NormalizedLandmark> _leftIris = new List<NormalizedLandmark>(IrisCount);
        private readonly List<NormalizedLandmark> _rightIris = new List<NormalizedLandmark>(IrisCount);

        // ---- Public surface consumed by the UnitEye pipeline (unchanged from the old Solution facade) ----
        public IList<NormalizedLandmark> FaceLandmarks { get; private set; }
        public IList<NormalizedLandmark> LeftIrisLandmarks { get; private set; }
        public IList<NormalizedLandmark> RightIrisLandmarks { get; private set; }
        public bool IsFaceDetected { get; private set; }

        /// <summary>Toggle the debug face-mesh overlay. (Overlay drawing is not reimplemented on the Task
        /// API path yet; this stores the preference so the Gaze UI toggle / IsRendering keep working.)</summary>
        public bool Annotate { get => _annotate; set => _annotate = value; }
        /// <summary>Kept for API parity with the old Solution facade (calibration toggles it).</summary>
        public bool IsRendering { get; set; } = true;

        public float HeadYaw
        {
            get
            {
                if (FaceLandmarks == null) return 0f;
                var l50 = FaceLandmarks[50];
                var l280 = FaceLandmarks[280];
                return Mathf.Atan((l50.Z - l280.Z) / (l50.X - l280.X));
            }
        }

        public float HeadPitch
        {
            get
            {
                if (FaceLandmarks == null) return 0f;
                var l10 = FaceLandmarks[10];
                var l168 = FaceLandmarks[168];
                return Mathf.Atan((l10.Z - l168.Z) / (l168.Y - l10.Y));
            }
        }

        public float HeadRoll
        {
            get
            {
                if (FaceLandmarks == null) return 0f;
                var l6 = FaceLandmarks[6];
                var l151 = FaceLandmarks[151];
                float roll = Mathf.Atan2(l151.X - l6.X, l6.Y - l151.Y);
                //Divide by 2 to lessen roll impact (same as the old facade)
                return roll >= 0 ? (roll - Mathf.PI) / 2 : (roll + Mathf.PI) / 2;
            }
        }

        // Head "area" from the face landmark bounding box (the old FaceRects source is gone in the Task
        // API). NOTE: the old pipeline's FaceRects was null in sync mode, so HeadArea used to be 0 — this
        // is a nonzero value now, so recalibrate after the migration. The bbox is cached per detected
        // frame in CopyLandmarks; consumers read it 2-4x per frame, so no per-access 468-point loop.
        public float HeadArea
        {
            get
            {
                if (FaceLandmarks == null || FaceLandmarks.Count == 0) return 0f;
                return Mathf.Max(0f, _bboxMaxX - _bboxMinX) * Mathf.Max(0f, _bboxMaxY - _bboxMinY);
            }
        }

        /// <summary>Camera frame width in pixels (0 until the webcam delivers a real frame). Landmarks are
        /// normalized in THIS space — consumers converting them to pixels must use these dims, not the
        /// game window's Screen size.</summary>
        public int FrameWidth => _webCamSource != null ? _webCamSource.textureWidth : 0;
        /// <summary>Camera frame height in pixels (0 until the webcam delivers a real frame).</summary>
        public int FrameHeight => _webCamSource != null ? _webCamSource.textureHeight : 0;

        /// <summary>Face landmark (0..467) bounding box in normalized image coords (y-down), cached per
        /// detected frame. Zero rect while no face is tracked.</summary>
        public UnityEngine.Rect FaceBoundsNormalized
        {
            get
            {
                if (FaceLandmarks == null || FaceLandmarks.Count == 0) return default;
                return UnityEngine.Rect.MinMaxRect(_bboxMinX, _bboxMinY, _bboxMaxX, _bboxMaxY);
            }
        }

        public float[] HeadGeom => new float[4] { HeadYaw, HeadPitch, HeadRoll, HeadArea };

        //Reused so the per-frame EyeMU input (HomulerEyeMURunner reads this every inference) allocates no
        //float[8]. The consumer copies it straight into a tensor, so a shared buffer is safe (same as the
        //runner's _poseBuffer). Caller reads this only when FaceLandmarks is populated (after ComputeEyes).
        private readonly float[] _eyeCornersBuffer = new float[8];
        public float[] EyeCorners
        {
            get
            {
                var b = _eyeCornersBuffer;
                b[0] = FaceLandmarks[263].X; b[1] = FaceLandmarks[263].Y;
                b[2] = FaceLandmarks[362].X; b[3] = FaceLandmarks[362].Y;
                b[4] = FaceLandmarks[33].X;  b[5] = FaceLandmarks[33].Y;
                b[6] = FaceLandmarks[133].X; b[7] = FaceLandmarks[133].Y;
                return b;
            }
        }

        private void Start()
        {
            // OnGUI below is Repaint-only and uses no GUILayout; skipping the layout pass halves the
            // IMGUI overhead of the preview.
            useGUILayout = false;

            if (_webCamSource == null)
                _webCamSource = GetComponent<WebCamSource>();

            // Preallocate the reusable landmark objects and the iris slice views (they reference the same
            // objects, which are mutated in place each frame).
            for (int i = 0; i < TotalLandmarks; i++)
                _mpLandmarks.Add(new NormalizedLandmark());
            for (int i = 0; i < IrisCount; i++)
            {
                _leftIris.Add(_mpLandmarks[LeftIrisStart + i]);
                _rightIris.Add(_mpLandmarks[RightIrisStart + i]);
            }

            var modelPath = Path.Combine(Application.streamingAssetsPath, ModelFileName);
            if (!File.Exists(modelPath))
            {
                Debug.LogError($"FaceMeshSolution: MediaPipe model not found at {modelPath}. " +
                               "Run 'UnitEye ▸ Install MediaPipe StreamingAssets'.");
                enabled = false;
                return;
            }

            var baseOptions = new BaseOptions(BaseOptions.Delegate.CPU, modelAssetBuffer: File.ReadAllBytes(modelPath));
            var options = new FaceLandmarkerOptions(
                baseOptions,
                runningMode: RunningMode.VIDEO,
                numFaces: maxNumFaces,
                minFaceDetectionConfidence: 0.5f,
                minFacePresenceConfidence: 0.5f,
                minTrackingConfidence: 0.5f);
            _faceLandmarker = FaceLandmarker.CreateFromOptions(options, GpuManager.GpuResources);
            _result = FaceLandmarkerResult.Alloc(maxNumFaces);
            _stopwatch.Start();
        }

        private void Update()
        {
            if (_faceLandmarker == null || _webCamSource == null || !_webCamSource.isPrepared)
                return;

            // Only run the (expensive) readback + inference when the camera actually delivered a new
            // frame. Without this gate a 144 Hz display reprocesses the same 30 fps webcam frame ~5x,
            // each time paying a full GPU->CPU ReadPixels stall plus blocking CPU inference for
            // identical landmarks.
            if (!_webCamSource.didUpdateThisFrame)
                return;

            var texture = _webCamSource.GetCurrentTexture();
            if (texture == null)
                return;

            // (Re)create the frame pool when the source resolution changes (Next/Previous Camera can
            // switch to a device with a different resolution). A stale pool size makes ReadTextureOnCPU
            // pad/crop the live frame inside an old-sized texture, so landmarks come back scaled/offset
            // relative to the real frame while the crop math uses the real texture size.
            if (_textureFramePool == null || _poolWidth != texture.width || _poolHeight != texture.height)
            {
                _textureFramePool?.Dispose();
                _textureFramePool = new TextureFramePool(texture.width, texture.height, TextureFormat.RGBA32, 10);
                _poolWidth = texture.width;
                _poolHeight = texture.height;
            }

            if (!_textureFramePool.TryGetTextureFrame(out var textureFrame))
                return;

            // Fold the camera's reported orientation into the base convention flips (the old GraphRunner
            // emitted input_rotation/input_*_flipped side packets per device; the Task-API facade handles
            // the flip cases here). videoVerticallyMirrored XORs into the vertical flip; a 180-degree
            // rotation equals flipping both axes. 90/270 would swap the frame's w/h, which the whole
            // downstream crop/preview pipeline does not support — warn once instead of silently
            // producing rotated-garbage landmarks (README: camera-rotation rework is documented out of scope).
            bool flipH = _flipHorizontally;
            bool flipV = _flipVertically ^ _webCamSource.isVerticallyFlipped;
            int rotation = _webCamSource.rotation;
            if (rotation == 180)
            {
                flipH = !flipH;
                flipV = !flipV;
            }
            else if ((rotation == 90 || rotation == 270) && !_warnedRotation)
            {
                _warnedRotation = true;
                Debug.LogWarning($"FaceMeshSolution: camera '{_webCamSource.sourceName}' reports videoRotationAngle={rotation}. " +
                                 "The crop pipeline assumes an unrotated frame, so gaze will be wrong on this device.");
            }

            textureFrame.ReadTextureOnCPU(texture, flipH, flipV);
            var image = textureFrame.BuildCPUImage();
            textureFrame.Release();

            long timestampMs = _stopwatch.ElapsedMilliseconds;
            bool detected = _faceLandmarker.TryDetectForVideo(image, timestampMs, null, ref _result);
            image.Dispose();

            if (detected && _result.faceLandmarks != null && _result.faceLandmarks.Count > 0)
            {
                CopyLandmarks(_result.faceLandmarks[0].landmarks);
                if (_smoothGazeLandmarks)
                    SmoothGazeLandmarks(timestampMs);
                _lastLandmarkTimestampMs = timestampMs;
                IsFaceDetected = true;
                FaceLandmarks = _mpLandmarks;
                LeftIrisLandmarks = _leftIris;
                RightIrisLandmarks = _rightIris;
            }
            else
            {
                // Propagate face loss: null the landmark views so IsFacePresent turns false and the
                // backbones stop cropping/inferring from a stale rect. Without this, walking away from
                // the camera kept the whole pipeline running on frozen landmarks — fabricating gaze,
                // blink, distance and CSV rows for the entire absence. (The lists themselves are reused;
                // the next successful detection re-points these views at them.)
                IsFaceDetected = false;
                FaceLandmarks = null;
                LeftIrisLandmarks = null;
                RightIrisLandmarks = null;
                //Reset the gaze-landmark filters: after a face loss the next detection may be anywhere, and
                //filter state from the old position would drag the fresh landmarks toward it.
                for (int i = 0; i < _landmarkFilters.Length; i++)
                    _landmarkFilters[i].Reset();
                _lastLandmarkTimestampMs = -1;
            }
        }

        //Applies the One-Euro filters to the six gaze landmarks IN PLACE (the reused protobuf objects the
        //consumers read). Downstream this stabilizes the eye-crop rects, the EyeCorners model input and the
        //iris gaze features. Z stays raw (no gaze consumer reads Z of these landmarks).
        private void SmoothGazeLandmarks(long timestampMs)
        {
            float dt = _lastLandmarkTimestampMs >= 0 ? (timestampMs - _lastLandmarkTimestampMs) / 1000f : 0f;
            for (int i = 0; i < SmoothedLandmarkIndices.Length; i++)
            {
                var landmark = _mpLandmarks[SmoothedLandmarkIndices[i]];
                landmark.X = _landmarkFilters[i * 2].Filter(landmark.X, dt);
                landmark.Y = _landmarkFilters[i * 2 + 1].Filter(landmark.Y, dt);
            }
        }

        // Copy the Task-API landmarks (Tasks.Components.Containers.NormalizedLandmark, .x/.y/.z) into the
        // reused protobuf Mediapipe.NormalizedLandmark objects (.X/.Y/.Z) the consumers read. Also computes
        // the face (0..467) bounding box in the same pass — HeadArea/FaceBoundsNormalized read the cached
        // values instead of re-looping per access.
        private void CopyLandmarks(List<Tasks.Components.Containers.NormalizedLandmark> src)
        {
            float minX = 1f, minY = 1f, maxX = 0f, maxY = 0f;
            int n = Mathf.Min(src.Count, _mpLandmarks.Count);
            for (int i = 0; i < n; i++)
            {
                var s = src[i];
                var d = _mpLandmarks[i];
                d.X = s.x;
                d.Y = s.y;
                d.Z = s.z;
                if (i < FaceLandmarkCount)
                {
                    if (s.x < minX) minX = s.x;
                    if (s.x > maxX) maxX = s.x;
                    if (s.y < minY) minY = s.y;
                    if (s.y > maxY) maxY = s.y;
                }
            }
            _bboxMinX = minX;
            _bboxMinY = minY;
            _bboxMaxX = maxX;
            _bboxMaxY = maxY;
        }

        // Debug preview: full-screen webcam + facemesh landmark dots. The camera shows whenever IsRendering
        // (so it hides during calibration, like the old Screen did); the dots additionally require Annotate
        // (the Gaze UI "Show/Hide FaceMesh" toggle). Camera and dots use the same flip/mirror transform so
        // they stay registered on each other.
        private void OnGUI()
        {
            // OnGUI runs multiple times per frame (Layout + Repaint + one pass per input event); only the
            // Repaint pass actually draws, so skip the 478-iteration dot loop on all the others.
            if (Event.current.type != EventType.Repaint) return;
            if (!_drawPreview || !IsRendering) return;
            var tex = _webCamSource != null ? _webCamSource.GetCurrentTexture() : null;
            if (tex == null) return;

            // Draw behind HomulerGaze's eye-crop thumbnails / crosshair / Gaze UI (lower GUI.depth = on top).
            GUI.depth = _previewGuiDepth;

            float w = Screen.width, h = Screen.height;

            // Full-screen camera. Negative width/height flips the texture (same trick as the mirrored eye
            // thumbnail in HomulerGaze.OnGUI).
            var camRect = new UnityEngine.Rect(
                _previewMirror ? w : 0f,
                _previewFlipVertically ? h : 0f,
                _previewMirror ? -w : w,
                _previewFlipVertically ? -h : h);
            GUI.DrawTexture(camRect, tex, UnityEngine.ScaleMode.StretchToFill, false);

            // Facemesh landmark overlay (dots), gated by the Show/Hide FaceMesh toggle.
            var pts = FaceLandmarks;
            if (_annotate && pts != null && pts.Count > 0)
            {
                bool flipY = _previewFlipVertically ^ _overlayFlipVertically;
                float size = Mathf.Max(_overlayDotSize, _overlayDotSize * h / 1080f);
                float half = size * 0.5f;
                var prevColor = GUI.color;
                GUI.color = _overlayColor;
                for (int i = 0; i < pts.Count; i++)
                {
                    float nx = _previewMirror ? 1f - pts[i].X : pts[i].X;
                    float ny = flipY ? 1f - pts[i].Y : pts[i].Y;
                    GUI.DrawTexture(new UnityEngine.Rect(nx * w - half, ny * h - half, size, size), Texture2D.whiteTexture);
                }
                GUI.color = prevColor;
            }
        }

        private void OnDestroy()
        {
            _faceLandmarker?.Close();
            _faceLandmarker = null;
            _textureFramePool?.Dispose();
            _textureFramePool = null;
        }
    }
}
#endif
