// Excluded from WebGL player builds: depends on the native MediaPipe plugin (Mediapipe.Runtime
// has no wasm library, so IL2CPP linking fails). Kept for the Editor regardless of build target.
#if !UNITY_WEBGL || UNITY_EDITOR
using Mediapipe.Unity;
using Mediapipe.Unity.FaceMesh;
using System.Collections.Generic;
using UnitEye;
using Unity.InferenceEngine;
using UnityEngine;
using static UnitEye.HomulerFunctions;
using Screen = UnityEngine.Screen;
namespace UnitEye
{

    /// <summary>
    /// Runs the EyeMU gaze model on Unity's Inference Engine (com.unity.ai.inference),
    /// using landmarks from the native homuler MediaPipe FaceMeshSolution.
    /// This is the Barracuda-free replacement for the old EyeMURunner + HolisticBarracuda path.
    /// </summary>
    public class HomulerEyeMURunner : IGazeBackbone
    {
        const int IMG_SIZE = 128;

        private FaceMeshSolution _faceMesh;
        private EyeMUResource _eyeMUResource;
        private Model _model;
        private Worker _worker;

        //The EyeMU model declares its image inputs as (batch, 128, 128, 3) = NHWC (channels-last),
        //but TextureConverter defaults to NCHW, so force NHWC to match. Barracuda's new Tensor(rt,3) was NHWC.
        private readonly TextureTransform _nhwcTransform = new TextureTransform().SetTensorLayout(TensorLayout.NHWC);

        //Actual model I/O names (from the Inference Engine ONNX import): the image inputs carry a ':0' suffix.
        const string INPUT_LEFT = "input_1:0";
        const string INPUT_RIGHT = "input_2:0";
        const string INPUT_CORNERS = "input_4";
        const string INPUT_POSE = "input_5";
        const string OUTPUT_EMBEDDING = "dense_7";
        const string OUTPUT_GAZE = "dense_8";

        public float[] Embedding2Output { get; private set; } = new float[4];
        public float[] NetworkOutput { get; private set; } = new float[2];

        //IGazeBackbone: the raw (pre-calibration) gaze in pixels. NetworkOutput already holds pixel coords.
        public Vector2 RawGaze => new Vector2(NetworkOutput[0], NetworkOutput[1]);

        #region Head pose (re-enabled so the feature vector and UnitEyeAPI.GetHeadPose match the former Holistic path)
        public float HeadYaw => _faceMesh.HeadYaw;
        public float HeadPitch => _faceMesh.HeadPitch;
        public float HeadRoll => _faceMesh.HeadRoll;
        public float HeadArea => _faceMesh.HeadArea;
        public float[] HeadGeom => _faceMesh.HeadGeom;
        #endregion

        /// <summary>Length of the calibration feature vector <see cref="Features"/> / FillEyeMUFeatures produce.</summary>
        public const int FeatureCount = 15;

        //Reused feature buffer so the per-frame Features access allocates nothing (was a fresh List<float> +
        //AddRange growth every frame). See FillEyeMUFeatures for the layout. Valid only until the next
        //frame's access; callers that retain it (the calibration capture) must copy.
        private readonly float[] _features = new float[FeatureCount];
        public float[] Features
        {
            get
            {
                //Normalize the raw gaze point to 0..1 (the model's own output before it was scaled to pixels
                //in NetworkOutput) so the polynomial terms stay in a sane range.
                float gx = Screen.width > 0 ? NetworkOutput[0] / Screen.width : 0f;
                float gy = Screen.height > 0 ? NetworkOutput[1] / Screen.height : 0f;
                FillEyeMUFeatures(_features, Embedding2Output, gx, gy,
                    _faceMesh.HeadYaw, _faceMesh.HeadPitch, _faceMesh.HeadRoll, _faceMesh.HeadArea);
                return _features;
            }
        }

        /// <summary>
        /// Fills the EyeMU calibration feature vector: the 4-value embedding, a low-order POLYNOMIAL of the
        /// normalized raw gaze point [gx, gy, gx², gy², gx·gy, gx³, gy³], and the head pose [yaw, pitch,
        /// roll, area]. EyeMU regresses a screen point trained on portrait phones, so its raw point maps
        /// NON-LINEARLY onto a desktop screen; a per-axis linear ridge over just [gx, gy] fit the centre
        /// slope and compressed the corners inward ("stuck near the middle, corners bad"). The quadratic +
        /// cross terms model the off-centre asymmetry and the x–y coupling, and the cubic terms extend corner
        /// reach — the same 2nd-order calibration polynomial that fixed the direction backbones
        /// (GazeEstimationRunner.FillGazeFeatures), here applied to EyeMU's point instead of an angle. The old
        /// constant Screen.width/height features were dropped (they standardize to zero, i.e. carry no
        /// signal). Order is irrelevant to the standardized ridge/MLP; keeping it fixed is what matters for
        /// train/predict agreement. NOTE: this changes the vector length (was 12) — EyeMU must be RECALIBRATED
        /// (a stale-length model NaNs and falls back to raw gaze).
        /// </summary>
        public static void FillEyeMUFeatures(float[] f, float[] embedding, float gx, float gy,
            float headYaw, float headPitch, float headRoll, float headArea)
        {
            f[0] = embedding[0];
            f[1] = embedding[1];
            f[2] = embedding[2];
            f[3] = embedding[3];
            f[4] = gx;
            f[5] = gy;
            f[6] = gx * gx;
            f[7] = gy * gy;
            f[8] = gx * gy;
            f[9] = gx * gx * gx;
            f[10] = gy * gy * gy;
            f[11] = headYaw;
            f[12] = headPitch;
            f[13] = headRoll;
            f[14] = headArea;
        }

        //GUI textures
        public RenderTexture LeftEyeTexture { get; private set; } = new RenderTexture(IMG_SIZE, IMG_SIZE, 0, RenderTextureFormat.ARGB32);
        public RenderTexture RightEyeTexture { get; private set; } = new RenderTexture(IMG_SIZE, IMG_SIZE, 0, RenderTextureFormat.ARGB32);

        //Tensor textures (depth 0: these are random-write compute targets and never used as a depth buffer)
        private RenderTexture _leftEyeTextureTensor = new RenderTexture(IMG_SIZE, IMG_SIZE, 0, RenderTextureFormat.ARGBHalf);
        private RenderTexture _rightEyeTextureTensor = new RenderTexture(IMG_SIZE, IMG_SIZE, 0, RenderTextureFormat.ARGBHalf);

        //Reused pose input buffer (was a fresh float[4] every inference)
        private readonly float[] _poseBuffer = new float[4];

        //Reused eye-image input tensors so inference doesn't allocate + free two ~192KB (1x128x128x3) GPU
        //tensors every frame. TextureConverter.ToTensor writes into these pre-allocated tensors, the same
        //reuse GazeEstimationRunner does with its single input tensor. Overwriting them next frame is safe:
        //DownloadToArray() below forces the scheduled inference to complete before PerformInference returns,
        //so the previous frame's inputs are done being read by the time we refill them.
        private Tensor<float> _leftTensor;
        private Tensor<float> _rightTensor;

        public HomulerEyeMURunner(FaceMeshSolution faceMesh)
        {
            _faceMesh = faceMesh;
            _eyeMUResource = Resources.Load<EyeMUResource>("EyeMU");

            _model = ModelLoader.Load(_eyeMUResource.modelAsset);
            _worker = new Worker(_model, BackendType.GPUCompute);

            //Prepare tensor textures for random write
            _leftEyeTextureTensor.enableRandomWrite = true;
            _leftEyeTextureTensor.Create();

            _rightEyeTextureTensor.enableRandomWrite = true;
            _rightEyeTextureTensor.Create();

            _leftTensor = new Tensor<float>(new TensorShape(1, IMG_SIZE, IMG_SIZE, 3));
            _rightTensor = new Tensor<float>(new TensorShape(1, IMG_SIZE, IMG_SIZE, 3));
        }

        /// <summary>
        /// Perform gaze location inference based on the homuler webcam source.
        /// </summary>
        /// <param name="webcam">homuler WebCamSource to use</param>
        /// <returns>true if execution completed, false if not</returns>
        public bool PerformInference(WebCamSource webcam)
        {
            var webcamTexture = (WebCamTexture)webcam.GetCurrentTexture();

            if (!webcam.isPrepared || webcamTexture == null)
                return false;

            if (!ComputeEyes(webcamTexture))
                return false;

            //Eye corners (8) and head pose (4) input tensors. The tensor constructor copies the data, so
            //the reused _poseBuffer is safe (each frame's tensor is consumed + disposed before the next).
            var corners = new Tensor<float>(new TensorShape(1, 8), _faceMesh.EyeCorners);
            _poseBuffer[0] = _faceMesh.HeadYaw;
            _poseBuffer[1] = _faceMesh.HeadPitch;
            _poseBuffer[2] = _faceMesh.HeadRoll;
            _poseBuffer[3] = _faceMesh.HeadArea;
            var pose = new Tensor<float>(new TensorShape(1, 4), _poseBuffer);

            //The eye crops were already cropped straight from the webcam texture into LeftEyeTexture/
            //RightEyeTexture on the GPU in ComputeEyes (no CPU GetPixels readback). Preprocess them into
            //tensor-format RenderTextures, then convert to NHWC input tensors (shape 1x128x128x3) to match
            //the model's declared image-input layout.
            //HAND-TEST NOTE: shapes/names are verified to match the model, but pixel normalization and channel
            //order (RGB vs BGR) ride on the preprocessing shader + TextureConverter and can only be confirmed
            //by looking at live gaze. The GPU crop geometry (bottom-left UV, left-eye horizontal flip) also
            //wants a live check — the eye-crop thumbnails (Show Eyecrops) should look identical to before.
            _leftEyeTextureTensor = PreprocessImage(LeftEyeTexture, _leftEyeTextureTensor, _eyeMUResource.preprocessCompute);
            TextureConverter.ToTensor(_leftEyeTextureTensor, _leftTensor, _nhwcTransform);

            _rightEyeTextureTensor = PreprocessImage(RightEyeTexture, _rightEyeTextureTensor, _eyeMUResource.preprocessCompute);
            TextureConverter.ToTensor(_rightEyeTextureTensor, _rightTensor, _nhwcTransform);

            _worker.SetInput(INPUT_LEFT, _leftTensor);
            _worker.SetInput(INPUT_RIGHT, _rightTensor);
            _worker.SetInput(INPUT_CORNERS, corners);
            _worker.SetInput(INPUT_POSE, pose);
            _worker.Schedule();

            //dense_7 -> embedding (4 values)
            var dense7 = _worker.PeekOutput(OUTPUT_EMBEDDING) as Tensor<float>;
            var dense7Data = dense7.DownloadToArray();
            for (int i = 0; i < Embedding2Output.Length && i < dense7Data.Length; i++)
                Embedding2Output[i] = dense7Data[i];

            //dense_8 -> final gaze (2 values, normalized 0..1)
            var final = _worker.PeekOutput(OUTPUT_GAZE) as Tensor<float>;
            var finalData = final.DownloadToArray();
            NetworkOutput[0] = finalData[0] * Screen.width;
            NetworkOutput[1] = finalData[1] * Screen.height;

            //Cleanup the per-frame small input tensors (the eye-image tensors are reused, disposed in Dispose).
            corners.Dispose();
            pose.Dispose();

            return true;
        }

        /// <summary>
        /// Crop the left and right eye regions from the webcam texture straight into the eye RenderTextures
        /// on the GPU (no CPU GetPixels readback / Texture2D churn / per-frame texture allocation, which is
        /// what the old GetEyeTexture + FlipTexture path did every frame).
        /// </summary>
        /// <param name="texture">webcam texture</param>
        /// <returns>true if landmarks were available, false if not</returns>
        private bool ComputeEyes(WebCamTexture texture)
        {
            var landmarks = _faceMesh.FaceLandmarks;

            if (landmarks == null)
                return false;

            int srcW = texture.width, srcH = texture.height;
            if (srcW <= 0 || srcH <= 0)
                return false;

            //Left eye (mesh corners 362,263), horizontally flipped to match EyeMU's expected orientation.
            bool leftValid = BlitEyeCrop(texture, LeftEyeTexture, GetEyeCropRect(landmarks, 362, 263, srcW, srcH), srcW, srcH, flipX: true);

            //Right eye (mesh corners 33,133), no flip.
            bool rightValid = BlitEyeCrop(texture, RightEyeTexture, GetEyeCropRect(landmarks, 33, 133, srcW, srcH), srcW, srcH, flipX: false);

            //Never infer from the previous frame's crop when a landmark moves a crop outside the image.
            //A stale crop paired with current landmarks produces a plausible but wrong gaze estimate.
            return leftValid && rightValid;
        }

        /// <summary>
        /// GPU crop: samples the sub-rectangle <paramref name="crop"/> of <paramref name="source"/> into
        /// <paramref name="dest"/> via Graphics.Blit scale/offset. Uses the same bottom-left origin the old
        /// GetPixels path used; flipX negates the horizontal scale to mirror the left eye. If the crop is
        /// (partly) off the source it is skipped and false is returned so the caller rejects the sample
        /// instead of reusing the previous frame's crop.
        /// </summary>
        private static bool BlitEyeCrop(Texture source, RenderTexture dest, RectInt crop, int srcW, int srcH, bool flipX)
        {
            if (crop.width <= 0 || crop.height <= 0 ||
                crop.x < 0 || crop.y < 0 || crop.x + crop.width > srcW || crop.y + crop.height > srcH)
                return false;

            float cw = (float)crop.width / srcW;
            float ch = (float)crop.height / srcH;
            float cx = (float)crop.x / srcW;
            float cy = (float)crop.y / srcH;

            //Blit samples source at uv*scale + offset. For the horizontal flip, negate x and start from the
            //crop's right edge.
            Vector2 scale = flipX ? new Vector2(-cw, ch) : new Vector2(cw, ch);
            Vector2 offset = flipX ? new Vector2(cx + cw, cy) : new Vector2(cx, cy);
            Graphics.Blit(source, dest, scale, offset);
            return true;
        }

        /// <summary>
        /// Dispose of the Inference Engine worker and release the GPU RenderTextures.
        /// RenderTextures are native resources that the GC does not reclaim; without this, every provider
        /// rebuild (scene reload / Editor domain reload) leaks four GPU surfaces.
        /// </summary>
        public void Dispose()
        {
            _worker?.Dispose();
            _worker = null;

            _leftTensor?.Dispose();
            _leftTensor = null;
            _rightTensor?.Dispose();
            _rightTensor = null;

            ReleaseRT(LeftEyeTexture);
            ReleaseRT(RightEyeTexture);
            ReleaseRT(_leftEyeTextureTensor);
            ReleaseRT(_rightEyeTextureTensor);
        }

        private static void ReleaseRT(RenderTexture rt)
        {
            if (rt == null) return;
            rt.Release();
            Object.Destroy(rt);
        }
    }
}
#endif
