// Excluded from WebGL player builds: depends on the native MediaPipe plugin + Inference Engine.
#if !UNITY_WEBGL || UNITY_EDITOR
using Mediapipe.Unity;
using Mediapipe.Unity.FaceMesh;
using Unity.InferenceEngine;
using UnityEngine;
using Screen = UnityEngine.Screen;

namespace UnitEye
{
    /// <summary>
    /// Alternative gaze backbone driven by a direction-based model such as
    /// https://github.com/yakhyo/gaze-estimation (ResNet / MobileNet / MobileOne). Unlike EyeMU (which
    /// regresses a screen point directly), these models take a FACE crop and output a gaze DIRECTION
    /// (pitch, yaw). This runner crops the face using the MediaPipe FaceMesh face rect (so no extra face
    /// detector is needed), runs the model, and turns the angles into a rough on-screen point; the
    /// per-user calibration layer (RidgeRegression / SimpleMLP) then does the real angle->screen mapping
    /// from the feature vector, exactly as it refines EyeMU's raw output.
    ///
    /// STATUS: this is a wired, compiling integration seam, but the model-specific constants below
    /// (INPUT_SIZE, normalization, I/O names, output decoding, the angle->pixel gain) are UNVERIFIED —
    /// they depend on the exact ONNX you export. See docs/GAZE-BACKBONES.md. Add your model at
    /// Resources/ONNX/GazeEstimation and hand-test with a webcam. Without the model this backbone
    /// disables itself (logs once) so the project still runs on EyeMU.
    /// </summary>
    public class GazeEstimationRunner : IGazeBackbone
    {
        // ---- Model-specific constants: ADJUST to match your exported ONNX (see the doc) ----
        const string RESOURCE_PATH = "ONNX/GazeEstimation";   // Resources/ONNX/GazeEstimation.onnx
        const int INPUT_SIZE = 448;                            // L2CS/yakhyo default; MobileOne variants differ
        // Angle (radians) -> normalized screen gain used only for the pre-calibration RawGaze. The
        // calibration model maps the real screen position from Features, so this only needs to be roughly
        // sane (bigger = more screen travel per degree of gaze).
        const float ANGLE_TO_SCREEN_GAIN = 1.2f;
        // -------------------------------------------------------------------------------------

        private readonly FaceMeshSolution _faceMesh;
        private Model _model;
        private Worker _worker;
        private bool _enabled;
        private static bool s_warnedMissing;

        private readonly RenderTexture _faceCrop = new RenderTexture(INPUT_SIZE, INPUT_SIZE, 0, RenderTextureFormat.ARGB32);
        private readonly float[] _features = new float[8];
        private Vector2 _rawGaze;

        public GazeEstimationRunner(FaceMeshSolution faceMesh)
        {
            _faceMesh = faceMesh;

            var modelAsset = Resources.Load<ModelAsset>(RESOURCE_PATH);
            if (modelAsset == null)
            {
                if (!s_warnedMissing)
                {
                    s_warnedMissing = true;
                    UnitEyeLog.Error($"GazeEstimation backbone selected but no model found at Resources/{RESOURCE_PATH}.onnx. " +
                                     "Add the ONNX (see docs/GAZE-BACKBONES.md) or switch the backbone back to EyeMU.");
                }
                return;
            }

            _model = ModelLoader.Load(modelAsset);
            _worker = new Worker(_model, BackendType.GPUCompute);
            _enabled = true;
        }

        public Vector2 RawGaze => _rawGaze;
        public float[] Features => _features;
        public RenderTexture LeftEyeTexture => _faceCrop;   // face crop doubles as the debug thumbnail
        public RenderTexture RightEyeTexture => _faceCrop;

        public bool PerformInference(WebCamSource webcam)
        {
            if (!_enabled)
                return false;

            var tex = (WebCamTexture)webcam.GetCurrentTexture();
            if (!webcam.isPrepared || tex == null)
                return false;

            var rects = _faceMesh.FaceRects;
            if (rects == null || rects.Count == 0)
                return false;

            // Crop the face on the GPU from the MediaPipe face rect (normalized, y-down) into the square
            // model-input RT. Bottom-left UV like the eye-crop path; rotation is ignored (documented).
            var r = rects[0];
            float w = Mathf.Clamp01(r.Width);
            float h = Mathf.Clamp01(r.Height);
            float cx = Mathf.Clamp01(r.XCenter);
            float cyUp = 1f - Mathf.Clamp01(r.YCenter);
            var scale = new Vector2(w, h);
            var offset = new Vector2(cx - w * 0.5f, cyUp - h * 0.5f);
            Graphics.Blit(tex, _faceCrop, scale, offset);

            // NOTE: many exported gaze models expect ImageNet-normalized NCHW input. TextureConverter
            // gives an un-normalized tensor; if your ONNX does NOT bake normalization in, apply it here
            // (a preprocess compute shader, like EyeMU's PreprocessEyeMU.compute) before ToTensor.
            using (var input = TextureConverter.ToTensor(_faceCrop, INPUT_SIZE, INPUT_SIZE, 3))
            {
                _worker.Schedule(input);
            }

            // Decode the output. ADJUST for your model: this assumes a single output whose first two
            // values are (pitch, yaw) in radians. L2CS-style models instead emit per-bin logits that need
            // a softmax + expectation; decode those here if that's what your export produces.
            using (var output = (_worker.PeekOutput() as Tensor<float>))
            {
                if (output == null)
                    return false;
                var data = output.DownloadToArray();
                if (data.Length < 2)
                    return false;

                float pitch = data[0];
                float yaw = data[1];

                // Rough pre-calibration screen point (calibration refines from Features). yaw -> x, pitch -> y.
                float nx = Mathf.Clamp01(0.5f + yaw * ANGLE_TO_SCREEN_GAIN);
                float ny = Mathf.Clamp01(0.5f - pitch * ANGLE_TO_SCREEN_GAIN);
                _rawGaze = new Vector2(nx * Screen.width, ny * Screen.height);

                // Feature vector for calibration: the gaze angles plus head pose/geometry and screen size.
                _features[0] = pitch;
                _features[1] = yaw;
                _features[2] = _faceMesh.HeadYaw;
                _features[3] = _faceMesh.HeadPitch;
                _features[4] = _faceMesh.HeadRoll;
                _features[5] = _faceMesh.HeadArea;
                _features[6] = Screen.width;
                _features[7] = Screen.height;
            }

            return true;
        }

        public void Dispose()
        {
            _worker?.Dispose();
            _worker = null;
            if (_faceCrop != null)
            {
                _faceCrop.Release();
                Object.Destroy(_faceCrop);
            }
        }
    }
}
#endif
