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
        private FaceLandmarkerResult _result;
        private readonly System.Diagnostics.Stopwatch _stopwatch = new System.Diagnostics.Stopwatch();

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
        // is a nonzero value now, so recalibrate after the migration.
        public float HeadArea
        {
            get
            {
                if (FaceLandmarks == null || FaceLandmarks.Count == 0) return 0f;
                float minX = 1f, minY = 1f, maxX = 0f, maxY = 0f;
                for (int i = 0; i < FaceLandmarkCount && i < FaceLandmarks.Count; i++)
                {
                    var l = FaceLandmarks[i];
                    if (l.X < minX) minX = l.X;
                    if (l.X > maxX) maxX = l.X;
                    if (l.Y < minY) minY = l.Y;
                    if (l.Y > maxY) maxY = l.Y;
                }
                return Mathf.Max(0f, maxX - minX) * Mathf.Max(0f, maxY - minY);
            }
        }

        public float[] HeadGeom => new float[4] { HeadYaw, HeadPitch, HeadRoll, HeadArea };

        public float[] EyeCorners => new float[] {
            FaceLandmarks[263].X, FaceLandmarks[263].Y,
            FaceLandmarks[362].X, FaceLandmarks[362].Y,
            FaceLandmarks[33].X, FaceLandmarks[33].Y,
            FaceLandmarks[133].X, FaceLandmarks[133].Y,
        };

        private void Start()
        {
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

            var texture = _webCamSource.GetCurrentTexture();
            if (texture == null)
                return;

            if (_textureFramePool == null)
                _textureFramePool = new TextureFramePool(texture.width, texture.height, TextureFormat.RGBA32, 10);

            if (!_textureFramePool.TryGetTextureFrame(out var textureFrame))
                return;

            textureFrame.ReadTextureOnCPU(texture, _flipHorizontally, _flipVertically);
            var image = textureFrame.BuildCPUImage();
            textureFrame.Release();

            long timestampMs = _stopwatch.ElapsedMilliseconds;
            bool detected = _faceLandmarker.TryDetectForVideo(image, timestampMs, null, ref _result);
            image.Dispose();

            if (detected && _result.faceLandmarks != null && _result.faceLandmarks.Count > 0)
            {
                CopyLandmarks(_result.faceLandmarks[0].landmarks);
                IsFaceDetected = true;
                FaceLandmarks = _mpLandmarks;
                LeftIrisLandmarks = _leftIris;
                RightIrisLandmarks = _rightIris;
            }
            else
            {
                IsFaceDetected = false;
            }
        }

        // Copy the Task-API landmarks (Tasks.Components.Containers.NormalizedLandmark, .x/.y/.z) into the
        // reused protobuf Mediapipe.NormalizedLandmark objects (.X/.Y/.Z) the consumers read.
        private void CopyLandmarks(List<Tasks.Components.Containers.NormalizedLandmark> src)
        {
            int n = Mathf.Min(src.Count, _mpLandmarks.Count);
            for (int i = 0; i < n; i++)
            {
                var s = src[i];
                var d = _mpLandmarks[i];
                d.X = s.x;
                d.Y = s.y;
                d.Z = s.z;
            }
        }

        // Debug preview: full-screen webcam + facemesh landmark dots. The camera shows whenever IsRendering
        // (so it hides during calibration, like the old Screen did); the dots additionally require Annotate
        // (the Gaze UI "Show/Hide FaceMesh" toggle). Camera and dots use the same flip/mirror transform so
        // they stay registered on each other.
        private void OnGUI()
        {
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
