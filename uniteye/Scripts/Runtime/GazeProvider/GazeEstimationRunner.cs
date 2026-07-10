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
    /// Gaze backbone driven by a direction-based model from https://github.com/yakhyo/gaze-estimation
    /// (MobileOne-s0 or MobileNetV2). Unlike EyeMU (which regresses a screen point), these take a FACE crop
    /// and output a gaze DIRECTION. This runner crops the face from the MediaPipe FaceMesh rect (no extra
    /// detector), runs the ONNX, decodes the angles, and turns them into a rough on-screen point; the
    /// per-user calibration (RidgeRegression / SimpleMLP) then does the real angle->screen mapping from the
    /// feature vector, exactly as it refines EyeMU's raw output.
    ///
    /// Model I/O (introspected from the shipped ONNX via GazeModelInspector — both models are identical):
    ///   input  "input"  (1, 3, 448, 448)  NCHW, RGB, ImageNet-normalized
    ///   output "yaw"     (1, 90)  \  per-bin logits -> softmax + expectation * 4deg - 180deg -> radians
    ///   output "pitch"   (1, 90)  /  (L2CS-style Gaze360 binning)
    ///
    /// Runtime gaze accuracy still needs a webcam to confirm (crop/normalization/bin convention); if the
    /// uncalibrated dot moves the wrong way, the two knobs are BIN_WIDTH_DEG/ANGLE_OFFSET_DEG (decode) and
    /// ANGLE_TO_SCREEN_GAIN (pre-calibration only). See docs/GAZE-BACKBONES.md.
    /// </summary>
    public class GazeEstimationRunner : IGazeBackbone
    {
        const int INPUT_SIZE = 448;
        const int NUM_BINS = 90;
        const float BIN_WIDTH_DEG = 4f;      // L2CS/Gaze360 bin width
        const float ANGLE_OFFSET_DEG = 180f; // bin 0 -> -180 deg; center bin ~ straight ahead
        const string INPUT_NAME = "input";
        const string OUTPUT_YAW = "yaw";
        const string OUTPUT_PITCH = "pitch";
        // Pre-calibration RawGaze gain only (calibration maps the real screen point from Features).
        const float ANGLE_TO_SCREEN_GAIN = 1.2f;

        private readonly FaceMeshSolution _faceMesh;
        private readonly ComputeShader _preprocess;
        private Model _model;
        private Worker _worker;
        private bool _enabled;
        private static bool s_warnedMissing;

        private readonly RenderTexture _faceCrop = new RenderTexture(INPUT_SIZE, INPUT_SIZE, 0, RenderTextureFormat.ARGB32);
        private readonly RenderTexture _tensorTex = new RenderTexture(INPUT_SIZE, INPUT_SIZE, 0, RenderTextureFormat.ARGBHalf);
        private readonly TextureTransform _nchw = new TextureTransform().SetTensorLayout(TensorLayout.NCHW);
        private Tensor<float> _inputTensor;
        private readonly float[] _features = new float[8];
        private Vector2 _rawGaze;

        public GazeEstimationRunner(FaceMeshSolution faceMesh, string modelResourcePath)
        {
            _faceMesh = faceMesh;

            var modelAsset = Resources.Load<ModelAsset>(modelResourcePath);
            _preprocess = Resources.Load<ComputeShader>("PreprocessGazeEstimation");
            if (modelAsset == null || _preprocess == null)
            {
                if (!s_warnedMissing)
                {
                    s_warnedMissing = true;
                    UnitEyeLog.Error($"GazeEstimation backbone: missing model 'Resources/{modelResourcePath}.onnx' " +
                                     "or the PreprocessGazeEstimation compute shader. Switch the backbone to EyeMU " +
                                     "or fix the assets (see docs/GAZE-BACKBONES.md).");
                }
                return;
            }

            _model = ModelLoader.Load(modelAsset);
            _worker = new Worker(_model, BackendType.GPUCompute);
            _tensorTex.enableRandomWrite = true;
            _tensorTex.Create();
            _inputTensor = new Tensor<float>(new TensorShape(1, 3, INPUT_SIZE, INPUT_SIZE));
            _enabled = true;
        }

        public Vector2 RawGaze => _rawGaze;
        public float[] Features => _features;
        public RenderTexture LeftEyeTexture => _faceCrop;   // the face crop doubles as the debug thumbnail
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

            // GPU-crop the face from the MediaPipe face rect (normalized, y-down) into the square input.
            // Bottom-left UV like the eye-crop path; rotation is ignored.
            var r = rects[0];
            float w = Mathf.Clamp01(r.Width);
            float h = Mathf.Clamp01(r.Height);
            float cx = Mathf.Clamp01(r.XCenter);
            float cyUp = 1f - Mathf.Clamp01(r.YCenter);
            Graphics.Blit(tex, _faceCrop, new Vector2(w, h), new Vector2(cx - w * 0.5f, cyUp - h * 0.5f));

            // ImageNet-normalize into the tensor texture, then convert to the (1,3,448,448) NCHW tensor.
            _preprocess.SetTexture(0, "_Texture", _faceCrop);
            _preprocess.SetTexture(0, "_Tensor", _tensorTex);
            _preprocess.Dispatch(0, INPUT_SIZE / 8, INPUT_SIZE / 8, 1);
            TextureConverter.ToTensor(_tensorTex, _inputTensor, _nchw);

            _worker.SetInput(INPUT_NAME, _inputTensor);
            _worker.Schedule();

            var yawT = _worker.PeekOutput(OUTPUT_YAW) as Tensor<float>;
            var pitchT = _worker.PeekOutput(OUTPUT_PITCH) as Tensor<float>;
            if (yawT == null || pitchT == null)
                return false;

            float yaw = DecodeAngleRadians(yawT.DownloadToArray());
            float pitch = DecodeAngleRadians(pitchT.DownloadToArray());

            // Rough pre-calibration screen point (calibration refines from Features). yaw -> x, pitch -> y.
            float nx = Mathf.Clamp01(0.5f + yaw * ANGLE_TO_SCREEN_GAIN);
            float ny = Mathf.Clamp01(0.5f - pitch * ANGLE_TO_SCREEN_GAIN);
            _rawGaze = new Vector2(nx * Screen.width, ny * Screen.height);

            // Feature vector for calibration: gaze angles + head pose/geometry + screen size.
            _features[0] = pitch;
            _features[1] = yaw;
            _features[2] = _faceMesh.HeadYaw;
            _features[3] = _faceMesh.HeadPitch;
            _features[4] = _faceMesh.HeadRoll;
            _features[5] = _faceMesh.HeadArea;
            _features[6] = Screen.width;
            _features[7] = Screen.height;
            return true;
        }

        /// <summary>
        /// L2CS-style decode of one output head: softmax over the bins, take the expected bin index, then
        /// map bin -> angle (index * bin_width - offset, in degrees) and return radians. Mutates the passed
        /// array in place (it is an owned per-frame readback buffer).
        /// </summary>
        public static float DecodeAngleRadians(float[] bins)
        {
            float max = float.NegativeInfinity;
            for (int i = 0; i < bins.Length; i++)
                if (bins[i] > max) max = bins[i];

            float sum = 0f;
            for (int i = 0; i < bins.Length; i++)
            {
                bins[i] = Mathf.Exp(bins[i] - max);
                sum += bins[i];
            }

            float expectation = 0f;
            if (sum > 0f)
                for (int i = 0; i < bins.Length; i++)
                    expectation += (bins[i] / sum) * i;

            float degrees = expectation * BIN_WIDTH_DEG - ANGLE_OFFSET_DEG;
            return degrees * Mathf.Deg2Rad;
        }

        public void Dispose()
        {
            _worker?.Dispose();
            _worker = null;
            _inputTensor?.Dispose();
            _inputTensor = null;
            if (_faceCrop != null) { _faceCrop.Release(); Object.Destroy(_faceCrop); }
            if (_tensorTex != null) { _tensorTex.Release(); Object.Destroy(_tensorTex); }
        }
    }
}
#endif
