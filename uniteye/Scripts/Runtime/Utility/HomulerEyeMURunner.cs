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

        /// <summary>
        /// Length of the calibration feature vector: FillEyeMUFeatures' 15 terms [embedding4, gaze
        /// polynomial 7, head pose 4], the 4 iris-offset features (indices 15..18,
        /// HomulerFunctions.FillIrisFeatures), then the shared 17-feature context block (head translation/
        /// depth, eyeLook blendshapes, gaze interaction terms — HomulerFunctions.FillContextFeatures).
        /// Blocks are APPENDED so the head-pose slots (11/12/13, targeted by the augmentation jitter) keep
        /// their positions. Changing this length stales saved calibrations (NaN -> raw-gaze fallback);
        /// recalibrate.
        /// </summary>
        public const int FeatureCount = 19 + HomulerFunctions.ContextFeatureCount;
        /// <summary>Index of the first iris-offset feature (see HomulerFunctions.FillIrisFeatures).</summary>
        public const int IrisFeatureStart = 15;
        /// <summary>Index of the first shared-context feature.</summary>
        public const int ContextFeatureStart = 19;

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
                //Head pose + iris come from the PUBLISHED tail (snapshotted when the published inference was
                //scheduled), not live landmarks: with async readback the outputs are a frame old, and mixing
                //them with newer head/iris values would assemble a feature vector no single frame produced.
                FillEyeMUFeatures(_features, Embedding2Output, gx, gy,
                    _tailPublished[0], _tailPublished[1], _tailPublished[2], _tailPublished[3]);
                _features[IrisFeatureStart] = _tailPublished[4];
                _features[IrisFeatureStart + 1] = _tailPublished[5];
                _features[IrisFeatureStart + 2] = _tailPublished[6];
                _features[IrisFeatureStart + 3] = _tailPublished[7];
                //Shared context block: head translation/depth, eyeLook blendshapes, interaction terms.
                //gx/gy (the normalized gaze point) are this backbone's primary gaze terms.
                HomulerFunctions.FillContextFeatures(_features, ContextFeatureStart, gx, gy,
                    _tailPublished[0], _tailPublished[1], _tailPublished, TailContextStart);
                return _features;
            }
        }

        /// <summary>Capture time (Time.unscaledTimeAsDouble) of the frame behind the published gaze.</summary>
        public double CaptureTimestamp => _timestampPublished;

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
        //in sync mode DownloadToArray forces the scheduled inference to complete before PerformInference
        //returns, and in async mode a new inference is only scheduled once the previous readback completed.
        private Tensor<float> _leftTensor;
        private Tensor<float> _rightTensor;

        //Async (pipelined) readback state: results of the scheduled inference are published on a LATER call
        //once the GPU->CPU readback completes, instead of stalling the CPU in DownloadToArray every frame.
        private readonly bool _asyncReadback;
        private bool _pendingReadback;
        private Tensor<float> _outEmbedding, _outGaze;           // worker-owned output refs (not disposed)
        private Tensor<float> _pendingCorners, _pendingPose;     // inputs kept alive until publish
        //The feature-vector TAIL (head pose 4 + iris offsets 4 + context 11) snapshotted when an inference
        //is SCHEDULED and published together with its outputs, so the assembled feature vector is internally
        //consistent (all values from the same camera frame) even when the result arrives a frame later.
        private const int TailLength = 8 + HomulerFunctions.ContextTailCount;
        private const int TailContextStart = 8;
        private readonly float[] _tailPending = new float[TailLength];
        private readonly float[] _tailPublished = new float[TailLength];
        private double _timestampPending, _timestampPublished;

        public HomulerEyeMURunner(FaceMeshSolution faceMesh, bool asyncReadback = false)
        {
            _faceMesh = faceMesh;
            _asyncReadback = asyncReadback;
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

        //Snapshot the non-inference feature tail (head pose + iris offsets + context) from CURRENT landmarks.
        private void CaptureFeatureTail(float[] dest)
        {
            dest[0] = _faceMesh.HeadYaw;
            dest[1] = _faceMesh.HeadPitch;
            dest[2] = _faceMesh.HeadRoll;
            dest[3] = _faceMesh.HeadArea;
            HomulerFunctions.FillIrisFeatures(_faceMesh.FaceLandmarks, dest, 4);
            HomulerFunctions.FillTailContext(_faceMesh, dest, TailContextStart);
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

            //Async mode: publish the PREVIOUS inference first, if its readback finished. If it hasn't, do
            //not schedule new work over the in-flight tensors — report "no new sample" and try again next
            //frame. A fresh result therefore arrives one camera frame later than in sync mode, but the CPU
            //never blocks waiting for the GPU.
            bool published = false;
            if (_asyncReadback && _pendingReadback)
            {
                if (!_outEmbedding.IsReadbackRequestDone() || !_outGaze.IsReadbackRequestDone())
                    return false;
                PublishPendingOutputs();
                published = true;
            }

            if (!ComputeEyes(webcamTexture))
                return published;

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

            if (_asyncReadback)
            {
                //Kick off the non-blocking readbacks and remember everything needed to publish later. The
                //corners/pose tensors must stay alive until the GPU finished consuming them.
                _outEmbedding = _worker.PeekOutput(OUTPUT_EMBEDDING) as Tensor<float>;
                _outGaze = _worker.PeekOutput(OUTPUT_GAZE) as Tensor<float>;
                _outEmbedding.ReadbackRequest();
                _outGaze.ReadbackRequest();
                _pendingCorners = corners;
                _pendingPose = pose;
                CaptureFeatureTail(_tailPending);
                _timestampPending = _faceMesh.LastCaptureTimestamp;
                _pendingReadback = true;
                return published;
            }

            //Sync mode: block on the results now (DownloadToArray waits for the GPU).
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

            //Same-frame tail: identical values to the old live reads, just captured once here.
            CaptureFeatureTail(_tailPublished);
            _timestampPublished = _faceMesh.LastCaptureTimestamp;

            //Cleanup the per-frame small input tensors (the eye-image tensors are reused, disposed in Dispose).
            corners.Dispose();
            pose.Dispose();

            return true;
        }

        //Publishes the completed async inference: non-blocking downloads (the readback already finished),
        //then the tail snapshot taken when that inference was scheduled.
        private void PublishPendingOutputs()
        {
            var embedding = _outEmbedding.DownloadToArray();
            for (int i = 0; i < Embedding2Output.Length && i < embedding.Length; i++)
                Embedding2Output[i] = embedding[i];

            var gaze = _outGaze.DownloadToArray();
            NetworkOutput[0] = gaze[0] * Screen.width;
            NetworkOutput[1] = gaze[1] * Screen.height;

            System.Array.Copy(_tailPending, _tailPublished, _tailPublished.Length);
            _timestampPublished = _timestampPending;

            _outEmbedding = null;
            _outGaze = null;
            _pendingCorners?.Dispose();
            _pendingCorners = null;
            _pendingPose?.Dispose();
            _pendingPose = null;
            _pendingReadback = false;
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

            //In-flight async inputs (the worker-owned output refs are disposed with the worker).
            _pendingCorners?.Dispose();
            _pendingCorners = null;
            _pendingPose?.Dispose();
            _pendingPose = null;
            _outEmbedding = null;
            _outGaze = null;
            _pendingReadback = false;

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
