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
        // Copied into StreamingAssets by MediaPipeAssetInstaller. It must be the WITH_BLENDSHAPES variant:
        // the 52 blendshapes come from a separate predictor packed inside the bundle, and requesting that
        // output from the smaller face_landmarker_v2 bundle fails task creation outright ("BLENDSHAPES Tag
        // and blendshapes model must be both set").
        public const string ModelFileName = "face_landmarker_v2_with_blendshapes.bytes";
        // Superseded bundle name — only used to clean it out of StreamingAssets / diagnose a stale install.
        public const string LegacyModelFileName = "face_landmarker_v2.bytes";

        // face_landmarker_v2 outputs 478 landmarks: 0..467 face mesh, then two 5-point iris blocks.
        // NOTE the Left/Right names here follow MediaPipe's own IMAGE-relative naming, which is mirrored
        // relative to the SUBJECT-relative naming used by the eye-corner constants in HomulerFunctions
        // (468.. actually lies in the eye whose corners are 33/133). That only matters when pairing an
        // iris with a specific eye — the sole consumer of these two blocks is HomulerEyeHelper's iris
        // SIZE, which takes whichever eye is bigger, so the labels do not affect any result here.
        // HomulerFunctions.Left/RightIrisCenter carry the corrected, eye-accurate mapping.
        private const int FaceLandmarkCount = 468;
        private const int LeftIrisStart = 468;
        private const int RightIrisStart = 473;
        private const int IrisCount = 5;
        private const int TotalLandmarks = 478;

        [SerializeField] private WebCamSource _webCamSource;
        [SerializeField] private bool _annotate = true;
        [SerializeField] private int maxNumFaces = 1;
        // Head pose from the FaceLandmarker's facial transformation matrix (a metric rigid fit of the
        // canonical face model) instead of the legacy atan-of-relative-Z landmark hack. The matrix pose is
        // markedly cleaner (normalized-landmark Z is the model's weakest channel and the old roll carried
        // an inherited magic /2), and it additionally provides head TRANSLATION, which no landmark-derived
        // feature carried. Kept as a toggle so the legacy behaviour remains reachable if a webcam hand-test
        // ever shows a sign/convention issue on some device.
        [Tooltip("Derive head yaw/pitch/roll (and translation) from MediaPipe's metric facial transformation matrix instead of the legacy landmark-Z approximation.")]
        [SerializeField] private bool _useMatrixHeadPose = true;
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

        // ---- Facial transformation matrix state (metric head pose + translation, in canonical-face cm).
        // NOTE the matrix is computed against MediaPipe's own assumed virtual camera, so the translation is
        // only approximately metric unless the real camera FOV matches; the ROTATION is trustworthy, and
        // for calibration features an approximately-proportional translation is exactly what's needed.
        private bool _hasTransformMatrix;
        private float _matrixYaw, _matrixPitch, _matrixRoll;   // radians
        private Vector3 _headTranslation;                      // cm, camera space

        // ---- Blendshape state. Indices resolved by category NAME once (robust to ordering), then read by
        // index every frame. EyeLook order: [inL, outL, upL, downL, inR, outR, upR, downR].
        private static readonly string[] EyeLookNames =
        {
            "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeLookDownLeft",
            "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeLookDownRight",
        };
        private readonly int[] _eyeLookIndices = new int[8];
        private int _blinkLeftIndex = -1, _blinkRightIndex = -1;
        private bool _blendshapeIndicesResolved;
        private readonly float[] _eyeLookScores = new float[8];
        private bool _hasBlendshapes;
        //Whether the landmarker was created WITH the blendshape output at all (vs _hasBlendshapes, which
        //tracks whether the current frame actually carried scores).
        private bool _blendshapesEnabled = true;
        private float _eyeBlinkLeft, _eyeBlinkRight;

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
                if (_useMatrixHeadPose && _hasTransformMatrix) return _matrixYaw;
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
                if (_useMatrixHeadPose && _hasTransformMatrix) return _matrixPitch;
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
                if (_useMatrixHeadPose && _hasTransformMatrix) return _matrixRoll;
                if (FaceLandmarks == null) return 0f;
                var l6 = FaceLandmarks[6];
                var l151 = FaceLandmarks[151];
                float roll = Mathf.Atan2(l151.X - l6.X, l6.Y - l151.Y);
                //Divide by 2 to lessen roll impact (same as the old facade)
                return roll >= 0 ? (roll - Mathf.PI) / 2 : (roll + Mathf.PI) / 2;
            }
        }

        /// <summary>True while this frame carries a facial transformation matrix (face tracked + output enabled).</summary>
        public bool HasTransformMatrix => _hasTransformMatrix && FaceLandmarks != null;
        /// <summary>Head translation from the transformation matrix, canonical-face cm in camera space.
        /// Approximately metric (see the matrix-state comment); Vector3.zero when unavailable.</summary>
        public Vector3 HeadTranslation => HasTransformMatrix ? _headTranslation : Vector3.zero;

        /// <summary>True while this frame carries face blendshapes (face tracked + output enabled).</summary>
        public bool HasBlendshapes => _hasBlendshapes && FaceLandmarks != null;
        /// <summary>The 8 eyeLook* blendshape scores [inL,outL,upL,downL,inR,outR,upR,downR] — a direct,
        /// independently-trained gaze cue fed to the calibration features. Reused buffer, zeroed when
        /// unavailable; callers that retain values must copy.</summary>
        public float[] EyeLookBlendshapes => _eyeLookScores;
        /// <summary>eyeBlink blendshape scores (0..1). Preferred over the EAR heuristic for the blink gate:
        /// the attention model separates lid closure from downward gaze, which EAR conflates.</summary>
        public float EyeBlinkLeft => _hasBlendshapes ? _eyeBlinkLeft : 0f;
        public float EyeBlinkRight => _hasBlendshapes ? _eyeBlinkRight : 0f;

        /// <summary>Time (Time.unscaledTimeAsDouble) at which the camera frame behind the CURRENT landmarks
        /// was consumed from the webcam. This is the closest observable proxy for the capture time (USB/driver
        /// latency upstream of Unity is not measurable); consumers use it to pair gaze samples with
        /// world/AOI state as it was when the user actually looked. 0 until the first detection.</summary>
        public double LastCaptureTimestamp { get; private set; }

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

        // Exposed for dataset recording (IGazeRecordingSource): landmark-to-pixel registration depends on
        // the flips applied on the way into MediaPipe, and whether the 6 gaze landmarks were smoothed is a
        // caveat anyone training on the landmark block needs. Read-only — recorded, never corrected.
        public bool FrameFlippedHorizontally => _flipHorizontally;
        public bool FrameFlippedVertically => _flipVertically;
        public bool GazeLandmarksSmoothed => _smoothGazeLandmarks;

        /// <summary>
        /// The live camera texture MediaPipe is reading, or null. Valid for the current frame only.
        /// </summary>
        public Texture CameraTexture => _webCamSource != null ? _webCamSource.GetCurrentTexture() : null;

        /// <summary>
        /// Copies the landmarks as consecutive x,y,z triplets into a caller-owned buffer, returning the
        /// number of floats written. A copy rather than an accessor on purpose: the NormalizedLandmark
        /// objects are allocated once and mutated in place every frame, so anything retaining the list reads
        /// the newest frame instead of the one it meant to record.
        /// </summary>
        public int CopyLandmarks(float[] dest)
        {
            var landmarks = FaceLandmarks;
            if (dest == null || landmarks == null) return 0;
            int n = Mathf.Min(landmarks.Count, dest.Length / 3);
            for (int i = 0; i < n; i++)
            {
                var lm = landmarks[i];
                dest[i * 3] = lm.X;
                dest[i * 3 + 1] = lm.Y;
                dest[i * 3 + 2] = lm.Z;
            }
            return n * 3;
        }

        /// <summary>
        /// Copies the 8 eyeLook* scores then eyeBlinkLeft/Right into <paramref name="dest"/> (10 floats).
        /// False when this frame carried no blendshapes — the internal buffer keeps the last good frame's
        /// values in that case, so callers must not fall back to reading it.
        /// </summary>
        public bool TryCopyEyeBlendshapes(float[] dest)
        {
            if (dest == null || dest.Length < 10 || !HasBlendshapes) return false;
            for (int i = 0; i < 8; i++) dest[i] = _eyeLookScores[i];
            dest[8] = _eyeBlinkLeft;
            dest[9] = _eyeBlinkRight;
            return true;
        }

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

            //Request the two "rich" outputs the pipeline previously discarded: the facial transformation
            //matrix (metric head rotation + translation — replaces the landmark-Z head-pose approximation
            //and closes the lateral-head-shift blindness in the calibration features) and the 52 blendshapes
            //(direct eyeLook* gaze cues + eyeBlink scores for the blink gate).
            //A project whose StreamingAssets still holds the pre-blendshape bundle would hard-fail on the
            //first request, so fall back to a blendshape-free landmarker instead of leaving the whole gaze
            //pipeline dead: everything downstream already degrades (HasBlendshapes gates the eyeLook*
            //features, and NativeGazeProvider's blink gate falls back to the EAR heuristic).
            var modelBuffer = File.ReadAllBytes(modelPath);
            _faceLandmarker = TryCreateLandmarker(modelBuffer, withBlendshapes: true);
            if (_faceLandmarker == null)
            {
                _faceLandmarker = TryCreateLandmarker(modelBuffer, withBlendshapes: false);
                if (_faceLandmarker == null)
                {
                    enabled = false;
                    return;
                }
                _blendshapesEnabled = false;
                Debug.LogWarning($"FaceMeshSolution: '{modelPath}' carries no blendshape model, continuing without " +
                                 "blendshapes (blink detection falls back to the EAR heuristic). Re-run " +
                                 "'UnitEye ▸ Install MediaPipe StreamingAssets' to get the full bundle.");
            }
            _result = FaceLandmarkerResult.Alloc(maxNumFaces, outputFaceBlendshapes: _blendshapesEnabled, outputFaceTransformationMatrixes: true);
            _stopwatch.Start();
        }

        //Returns null instead of throwing so Start can retry with blendshapes off; only the final attempt
        //reports, otherwise a recoverable first try would log an error that resolves itself a line later.
        private FaceLandmarker TryCreateLandmarker(byte[] modelBuffer, bool withBlendshapes)
        {
            //Fresh BaseOptions per attempt: CreateFromOptions takes ownership of the options it is handed.
            var options = new FaceLandmarkerOptions(
                new BaseOptions(BaseOptions.Delegate.CPU, modelAssetBuffer: modelBuffer),
                runningMode: RunningMode.VIDEO,
                numFaces: maxNumFaces,
                minFaceDetectionConfidence: 0.5f,
                minFacePresenceConfidence: 0.5f,
                minTrackingConfidence: 0.5f,
                outputFaceBlendshapes: withBlendshapes,
                outputFaceTransformationMatrixes: true);
            try
            {
                return FaceLandmarker.CreateFromOptions(options, GpuManager.GpuResources);
            }
            catch (System.Exception e)
            {
                if (!withBlendshapes)
                    Debug.LogError($"FaceMeshSolution: could not create the MediaPipe FaceLandmarker. {e.Message}");
                return null;
            }
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

            //The frame was consumed from the webcam THIS engine frame — record when, so gaze samples can be
            //paired with world/AOI state at (approximately) the moment the user actually looked.
            double captureTime = Time.unscaledTimeAsDouble;

            long timestampMs = _stopwatch.ElapsedMilliseconds;
            bool detected = _faceLandmarker.TryDetectForVideo(image, timestampMs, null, ref _result);
            image.Dispose();

            if (detected && _result.faceLandmarks != null && _result.faceLandmarks.Count > 0)
            {
                CopyLandmarks(_result.faceLandmarks[0].landmarks);
                if (_smoothGazeLandmarks)
                    SmoothGazeLandmarks(timestampMs);
                _lastLandmarkTimestampMs = timestampMs;
                LastCaptureTimestamp = captureTime;
                ExtractTransformMatrix();
                ExtractBlendshapes();
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
                _hasTransformMatrix = false;
                _hasBlendshapes = false;
                //Reset the gaze-landmark filters: after a face loss the next detection may be anywhere, and
                //filter state from the old position would drag the fresh landmarks toward it.
                for (int i = 0; i < _landmarkFilters.Length; i++)
                    _landmarkFilters[i].Reset();
                _lastLandmarkTimestampMs = -1;
            }
        }

        //Reads the facial transformation matrix (canonical face -> camera space, cm) into cached yaw/pitch/
        //roll radians + translation. Extracted once per detected frame; the head-pose properties then serve
        //cached values. The rotation is taken via the matrix columns (forward = col 2, up = col 1) and
        //converted to Tait-Bryan angles wrapped to ±π. SIGN CONVENTIONS are a webcam hand-test item like the
        //preview flips (only consistency matters for the calibration features — standardization absorbs a
        //global sign — but UnitEyeAPI.GetHeadPose consumers may see flipped signs vs the legacy path; toggle
        //_useMatrixHeadPose off to restore the old behaviour).
        private void ExtractTransformMatrix()
        {
            var matrices = _result.facialTransformationMatrixes;
            if (matrices == null || matrices.Count == 0)
            {
                _hasTransformMatrix = false;
                return;
            }

            var m = matrices[0];
            var forward = new Vector3(m.m02, m.m12, m.m22);
            var up = new Vector3(m.m01, m.m11, m.m21);
            if (forward.sqrMagnitude < 1e-8f || up.sqrMagnitude < 1e-8f)
            {
                _hasTransformMatrix = false;
                return;
            }
            var e = Quaternion.LookRotation(forward, up).eulerAngles;
            _matrixPitch = WrapAngleRad(e.x);
            _matrixYaw = WrapAngleRad(e.y);
            _matrixRoll = WrapAngleRad(e.z);
            _headTranslation = new Vector3(m.m03, m.m13, m.m23);
            _hasTransformMatrix = true;
        }

        //Degrees (0..360, Unity euler) -> radians wrapped to (-π, π], matching the legacy head-pose range.
        private static float WrapAngleRad(float degrees)
        {
            float d = Mathf.Repeat(degrees + 180f, 360f) - 180f;
            return d * Mathf.Deg2Rad;
        }

        //Copies the eyeLook*/eyeBlink* blendshape scores into reused buffers. Category indices are resolved
        //by NAME on the first frame that carries blendshapes (robust to ordering), then read by index.
        private void ExtractBlendshapes()
        {
            var blendshapes = _result.faceBlendshapes;
            if (blendshapes == null || blendshapes.Count == 0)
            {
                _hasBlendshapes = false;
                return;
            }

            var categories = blendshapes[0].categories;
            if (categories == null || categories.Count == 0)
            {
                _hasBlendshapes = false;
                return;
            }

            if (!_blendshapeIndicesResolved)
            {
                for (int i = 0; i < _eyeLookIndices.Length; i++) _eyeLookIndices[i] = -1;
                for (int c = 0; c < categories.Count; c++)
                {
                    var name = categories[c].categoryName;
                    if (string.IsNullOrEmpty(name)) continue;
                    if (name == "eyeBlinkLeft") _blinkLeftIndex = c;
                    else if (name == "eyeBlinkRight") _blinkRightIndex = c;
                    else
                        for (int i = 0; i < EyeLookNames.Length; i++)
                            if (name == EyeLookNames[i]) { _eyeLookIndices[i] = c; break; }
                }
                _blendshapeIndicesResolved = true;
            }

            for (int i = 0; i < _eyeLookIndices.Length; i++)
            {
                int idx = _eyeLookIndices[i];
                _eyeLookScores[i] = idx >= 0 && idx < categories.Count ? categories[idx].score : 0f;
            }
            _eyeBlinkLeft = _blinkLeftIndex >= 0 && _blinkLeftIndex < categories.Count ? categories[_blinkLeftIndex].score : 0f;
            _eyeBlinkRight = _blinkRightIndex >= 0 && _blinkRightIndex < categories.Count ? categories[_blinkRightIndex].score : 0f;
            _hasBlendshapes = true;
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
