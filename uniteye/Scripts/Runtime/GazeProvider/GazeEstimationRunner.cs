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
    /// and output a gaze DIRECTION. This runner crops the face from the MediaPipe FaceMesh landmark
    /// bounding box (no extra detector), runs the ONNX, decodes the angles, and maps to a rough point; the
    /// per-user calibration (RidgeRegression / SimpleMLP) then does the real angle->screen mapping from the
    /// feature vector, exactly as it refines EyeMU's raw output.
    ///
    /// Model I/O (introspected from the shipped ONNX via GazeModelInspector — both models are identical):
    ///   input  "input"  (1, 3, 448, 448)  NCHW, RGB, ImageNet-normalized
    ///   output "yaw"     (1, 90)  \  per-bin logits -> softmax + expectation * 4deg - 180deg -> radians
    ///   output "pitch"   (1, 90)  /  (L2CS-style Gaze360 binning)
    ///
    /// The preprocessing + decode above are VERIFIED against the author's reference implementation
    /// (yakhyo/uniface, uniface/gaze/models.py — the successor library that wraps these exact ONNX
    /// releases as "MobileGaze"): 448x448 RGB, ImageNet mean/std, softmax + soft-argmax * 4 - 180 ->
    /// radians, pitch/yaw. Their demo feeds the RAW rectangular detector bbox (no padding/squaring); our
    /// squared FACE_CROP_SCALE-padded landmark bbox emulates that framing without aspect stretch —
    /// FACE_CROP_SCALE stays the webcam-tuning knob. ANGLE_TO_SCREEN_GAIN affects the pre-calibration
    /// RawGaze only. See docs/GAZE-BACKBONES.md.
    /// </summary>
    public class GazeEstimationRunner : IGazeBackbone
    {
        /// <summary>
        /// Input side used when the model's own shape cannot be read. Every export shipped here is 448, but
        /// the size is DERIVED from the loaded graph rather than assumed — the reference implementation does
        /// the same (uniface `input_size = input_shape[2:4][::-1]`), and a future export at another
        /// resolution would otherwise be silently mis-preprocessed: the crop, the compute dispatch and the
        /// tensor would all stay 448 while the model expected something else, producing plausible-looking
        /// garbage rather than an error.
        /// </summary>
        const int DEFAULT_INPUT_SIZE = 448;
        const int NUM_BINS = 90;
        const float BIN_WIDTH_DEG = 4f;      // L2CS/Gaze360 bin width
        const float ANGLE_OFFSET_DEG = 180f; // bin 0 -> -180 deg; center bin ~ straight ahead
        const string INPUT_NAME = "input";
        const string OUTPUT_YAW = "yaw";
        const string OUTPUT_PITCH = "pitch";
        // Pre-calibration RawGaze gain only (calibration maps the real screen point from Features).
        const float ANGLE_TO_SCREEN_GAIN = 1.2f;
        // Square face crop side = max(landmark-bbox width, height) * this, for context around the face.
        const float FACE_CROP_SCALE = 1.4f;

        private readonly FaceMeshSolution _faceMesh;
        private readonly ComputeShader _preprocess;
        private Model _model;
        private Worker _worker;
        private bool _enabled;
        private static bool s_warnedMissing;

        /// <summary>Input side this model actually wants, read from its graph (see DEFAULT_INPUT_SIZE).</summary>
        private readonly int _inputSize;
        /// <summary>Compute-shader threadgroups per axis; ceil so a non-multiple-of-8 side still covers the
        /// image (the kernel bounds-checks the tail threads).</summary>
        private readonly int _dispatchGroups;
        //Not readonly: sized from _inputSize, which is only known after the model loads (a readonly field
        //cannot be assigned from the helper that allocates both).
        private RenderTexture _faceCrop;
        private RenderTexture _tensorTex;
        private readonly TextureTransform _nchw = new TextureTransform().SetTensorLayout(TensorLayout.NCHW);
        private Tensor<float> _inputTensor;
        // Calibration feature vector: a polynomial expansion of the gaze angles + linear head pose (see
        // FillGazeFeatures). The direction model's angle->screen map is NON-LINEAR (x ~ D*tan(yaw)) with a
        // yaw*pitch coupling, which a per-axis LINEAR ridge cannot represent — so without the polynomial
        // terms the fit tracked the centre slope and compressed the corners inward ("crosshair stuck in
        // the middle, corners bad"). The expansion gives ridge (and the MLP) the basis to reach the
        // corners. Because both calibration capture and inference read this same vector via
        // provider.GetFeatures(), train and predict always agree.
        private readonly float[] _features = new float[FeatureCount];
        private Vector2 _rawGaze;

        //Async (pipelined) readback state — same design as HomulerEyeMURunner: results publish on a later
        //call once the readback completes, with the head/iris/context feature tail snapshotted at schedule
        //time so the assembled vector stays internally consistent.
        private readonly bool _asyncReadback;
        private bool _pendingReadback;
        private Tensor<float> _outYaw, _outPitch, _outEmbedding;   // worker-owned output refs (not disposed)
        private float[] _embeddingPublished;    // latest embedding readback (worker-owned buffer copy)
        //Tail layout: [head4, iris4, context11] (see TailLength / HomulerFunctions.FillTailContext).
        private const int TailLength = 8 + HomulerFunctions.ContextTailCount;
        private const int TailContextStart = 8;
        private readonly float[] _tailPending = new float[TailLength];
        private readonly float[] _tailPublished = new float[TailLength];
        private double _timestampPending, _timestampPublished;
        //Softmax spread of the CURRENTLY PUBLISHED inference, per axis, in radians (see
        //DecodeAngleRadians(float[], out float)). NaN until the first inference publishes, which is what
        //"this backbone has no uncertainty to report" means to every consumer.
        private Vector2 _sigmaPublished = new Vector2(float.NaN, float.NaN);

        //Horizontal-flip test-time augmentation: run the crop AND its mirror, negate the mirrored yaw,
        //average — a standard 3-8% angular-error reduction for appearance models at 2x inference cost.
        //Sync mode only (the async pipeline holds one in-flight inference); OFF by default until the
        //mirror/negate convention is confirmed against a live webcam (a wrong sign would average toward 0).
        private readonly bool _flipAugmentation;

        //Roll-normalize the face crop (2D data normalization) — see PerformInference. Hand-test knob:
        //if gaze degrades on a device (sign convention), disable via HomulerGaze's inspector toggle.
        private readonly bool _rollNormalize;

        public GazeEstimationRunner(FaceMeshSolution faceMesh, string modelResourcePath, bool asyncReadback = false,
            bool flipAugmentation = false, bool rollNormalize = true)
        {
            _faceMesh = faceMesh;
            _asyncReadback = asyncReadback;
            _flipAugmentation = flipAugmentation && !asyncReadback;
            _rollNormalize = rollNormalize;

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
                //Still allocate at the default size: the debug thumbnail properties are read by the Gaze UI
                //and the dataset recorder whether or not the backbone came up, and they were never null before.
                _inputSize = DEFAULT_INPUT_SIZE;
                _dispatchGroups = Mathf.CeilToInt(_inputSize / 8f);
                AllocateTextures();
                return;
            }

            _model = ModelLoader.Load(modelAsset);
            _inputSize = ReadInputSize(_model, modelResourcePath);
            _dispatchGroups = Mathf.CeilToInt(_inputSize / 8f);
            AllocateTextures();
            _worker = new Worker(_model, BackendType.GPUCompute);
            _inputTensor = new Tensor<float>(new TensorShape(1, 3, _inputSize, _inputSize));

            //Embedding-head personalization: the shipped ONNX files carry a third output "embedding" — the
            //pre-logit GAP feature vector (512/1024/1280-d depending on the model), tapped via graph edit.
            //Regressing the per-user calibration on THIS (instead of only the 2 decoded angles) is the
            //closed-form version of "fine-tune the last layer" (the Google NatComm 2020 recipe: SVR on
            //penultimate features halved error) — the decoded angles destroy the person-specific
            //appearance information the embedding still carries. The features array is sized engineered +
            //embedding; the ridge's standardization + CV-chosen lambda handle the extra columns.
            foreach (var output in _model.outputs)
                if (output.name == OUTPUT_EMBEDDING)
                    _hasEmbeddingOutput = true;
            //The width (512/1024/1280 depending on the model) is only visible at runtime — Model.Output
            //carries no shape — so the feature array is sized lazily at the first readback
            //(EnsureEmbeddingSized), before any Features consumer sees a sample.
            _enabled = true;
        }

        private void AllocateTextures()
        {
            _faceCrop = new RenderTexture(_inputSize, _inputSize, 0, RenderTextureFormat.ARGB32);
            _tensorTex = new RenderTexture(_inputSize, _inputSize, 0, RenderTextureFormat.ARGBHalf);
            _tensorTex.enableRandomWrite = true;
            _tensorTex.Create();
        }

        /// <summary>
        /// The square input side the model declares, from its input tensor's H/W. These are NCHW image
        /// models, so the shape is (1, 3, H, W) and H == W for every export in this family; a non-square or
        /// unreadable shape falls back to <see cref="DEFAULT_INPUT_SIZE"/> with a warning rather than
        /// guessing, because every downstream size (crop RT, compute dispatch, input tensor) derives from
        /// this one number and a wrong value mis-preprocesses silently instead of failing.
        /// </summary>
        private static int ReadInputSize(Model model, string modelResourcePath)
        {
            try
            {
                foreach (var input in model.inputs)
                {
                    if (input.name != INPUT_NAME && model.inputs.Count > 1) continue;
                    var shape = input.shape;
                    //rank ASSERTS on a dynamic-rank shape, so this guard is required, not defensive.
                    if (shape.isRankDynamic || shape.rank != 4) break;
                    //Get() is the public per-axis accessor (the DynamicTensorDim indexer is internal) and
                    //returns -1 for a dynamic dim — which carries no number to size buffers from.
                    int h = shape.Get(2), w = shape.Get(3);
                    if (h <= 0 || w <= 0) break;
                    if (h != w)
                    {
                        UnitEyeLog.Warn($"GazeEstimation backbone: '{modelResourcePath}' wants a non-square " +
                                        $"{w}x{h} input; this runner crops square. Using {h}.");
                    }
                    return h;
                }
            }
            catch (System.Exception e)
            {
                UnitEyeLog.Exception(e);
            }
            UnitEyeLog.Warn($"GazeEstimation backbone: could not read an input size from '{modelResourcePath}'; " +
                            $"falling back to {DEFAULT_INPUT_SIZE}. If this model is not {DEFAULT_INPUT_SIZE}px " +
                            "square its preprocessing is wrong — see docs/GAZE-BACKBONES.md.");
            return DEFAULT_INPUT_SIZE;
        }

        const string OUTPUT_EMBEDDING = "embedding";
        private bool _hasEmbeddingOutput;
        private int _embeddingDim;
        private float[] _embeddingSigns;   // ±1 sparse-JL projection signs (null when raw dim <= 64)

        /// <summary>
        /// Width the raw embedding is compressed to before entering the calibration features. The raw
        /// 512-1280-d embedding fed to the ridge directly made calibration TRAINING pathological: the
        /// trainer's cross-validation runs hundreds of dense solves, and 1300-column normal equations turn
        /// a two-second training step into minutes of frozen UI. A fixed ±1 random projection (sparse
        /// Johnson-Lindenstrauss, Achlioptas 2003) preserves the linear-regression geometry at 64 dims —
        /// the same budget the personalization literature reaches for via PCA — at ~80k multiply-adds per
        /// frame. The signs are generated by an IN-CODE xorshift with fixed constants (not System.Random,
        /// whose sequence is an implementation detail that can change across runtimes and would silently
        /// re-shuffle every saved calibration's feature space).
        /// </summary>
        public const int EmbeddingProjectionDim = 64;

        /// <summary>Deterministic ±1/sqrt(outDim... ) projection signs for ProjectEmbedding (length rawDim * outDim).</summary>
        public static float[] BuildEmbeddingSigns(int rawDim, int outDim)
        {
            var signs = new float[rawDim * outDim];
            uint state = 0x9E3779B9u;   // fixed seed — MUST never change (saved calibrations depend on it)
            float scale = 1f / Mathf.Sqrt(rawDim);
            for (int i = 0; i < signs.Length; i++)
            {
                //xorshift32 (Marsaglia) — fully specified here, runtime-independent.
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                signs[i] = (state & 1u) == 0u ? scale : -scale;
            }
            return signs;
        }

        /// <summary>Projects the raw embedding into dest[destOffset..destOffset+outDim) using the signs.</summary>
        public static void ProjectEmbedding(float[] raw, float[] signs, float[] dest, int destOffset, int outDim)
        {
            for (int k = 0; k < outDim; k++)
            {
                float sum = 0f;
                int row = k * raw.Length;
                for (int j = 0; j < raw.Length; j++)
                    sum += raw[j] * signs[row + j];
                dest[destOffset + k] = sum;
            }
        }

        //Sizes the published feature vector once the embedding width is known (first readback). Runs
        //before PublishResult on that same call, so consumers never observe a length change mid-session.
        private void EnsureEmbeddingSized(float[] embedding)
        {
            if (_fullFeatures != null || embedding == null) return;
            if (embedding.Length > EmbeddingProjectionDim)
            {
                _embeddingDim = EmbeddingProjectionDim;
                _embeddingSigns = BuildEmbeddingSigns(embedding.Length, EmbeddingProjectionDim);
            }
            else
            {
                _embeddingDim = embedding.Length;
            }
            _fullFeatures = new float[FeatureCount + _embeddingDim];
        }
        //Engineered features + backbone embedding (see ctor). Replaces _features as the published vector
        //when an embedding output exists; the engineered block occupies the first FeatureCount slots
        //either way, so the smoke-tested layout constants stay valid.
        private float[] _fullFeatures;
        /// <summary>Embedding width appended to the feature vector (0 for unedited models).</summary>
        public int EmbeddingDim => _embeddingDim;

        public Vector2 RawGaze => _rawGaze;
        public Vector2 AngularUncertainty => _sigmaPublished;
        public float[] Features => _fullFeatures ?? _features;
        public double CaptureTimestamp => _timestampPublished;
        public RenderTexture LeftEyeTexture => _faceCrop;   // the face crop doubles as the debug thumbnail
        public RenderTexture RightEyeTexture => _faceCrop;

        public bool PerformInference(WebCamSource webcam)
        {
            if (!_enabled)
                return false;

            var tex = (WebCamTexture)webcam.GetCurrentTexture();
            if (!webcam.isPrepared || tex == null)
                return false;

            //Async mode: publish the previous inference first once its readback is done; never schedule
            //over in-flight tensors (see HomulerEyeMURunner.PerformInference for the full rationale).
            bool published = false;
            if (_asyncReadback && _pendingReadback)
            {
                if (!_outYaw.IsReadbackRequestDone() || !_outPitch.IsReadbackRequestDone() ||
                    (_outEmbedding != null && !_outEmbedding.IsReadbackRequestDone()))
                    return false;
                float pendingYaw = DecodeAngleRadians(_outYaw.DownloadToArray(), out float pendingSigmaYaw);
                float pendingPitch = DecodeAngleRadians(_outPitch.DownloadToArray(), out float pendingSigmaPitch);
                if (_outEmbedding != null)
                {
                    _embeddingPublished = _outEmbedding.DownloadToArray();
                    EnsureEmbeddingSized(_embeddingPublished);
                }
                System.Array.Copy(_tailPending, _tailPublished, _tailPublished.Length);
                _timestampPublished = _timestampPending;
                //The sigma comes out of THIS readback, so unlike the head/iris tail it needs no
                //snapshot-at-schedule-time: it is already consistent with the angles it was decoded from.
                PublishResult(pendingYaw, pendingPitch, pendingSigmaYaw, pendingSigmaPitch, _tailPublished);
                _outYaw = null;
                _outPitch = null;
                _outEmbedding = null;
                _pendingReadback = false;
                published = true;
            }

            // Crop from the FaceMesh LANDMARK bounding box, NOT FaceRects: in the (NonBlocking)Sync
            // running mode this project uses, FaceMeshSolution only populates FaceLandmarks (via
            // WaitForNextValue) and leaves FaceRects null (it's only set on the async event path), so
            // relying on FaceRects made PerformInference return false every frame (stuck crosshair, no crop).
            var landmarks = _faceMesh.FaceLandmarks;
            if (landmarks == null || landmarks.Count == 0)
                return published;

            int srcW = tex.width, srcH = tex.height;
            if (srcW <= 0 || srcH <= 0)
                return published;

            // Face landmark bounding box (normalized, y-down), cached per frame by FaceMeshSolution —
            // no need to re-loop the 468 landmarks here.
            var bbox = _faceMesh.FaceBoundsNormalized;
            float minX = bbox.xMin, minY = bbox.yMin, maxX = bbox.xMax, maxY = bbox.yMax;

            // Square crop in PIXELS (so the 448x448 input isn't stretched), centred on the face with
            // FACE_CROP_SCALE padding for context, expressed as a bottom-left-UV Graphics.Blit.
            float cxPx = (minX + maxX) * 0.5f * srcW;
            float cyPx = (minY + maxY) * 0.5f * srcH;                   // y-down
            float sidePx = Mathf.Max((maxX - minX) * srcW, (maxY - minY) * srcH) * FACE_CROP_SCALE;
            if (sidePx <= 1f)
                return published;
            // Keep the crop inside the frame: with the 1.4x padding the square leaves the source whenever
            // the face nears an edge, and out-of-range UVs sample wrap-around/clamp-smeared pixels — the
            // model then sees the opposite frame edge inside the "face". Shrink to fit if the frame is
            // smaller than the padded square, then shift the square fully inside (mirrors the bounds
            // check BlitEyeCrop does for the eye crops).
            sidePx = Mathf.Min(sidePx, Mathf.Min(srcW, srcH));
            float cyUpPx = srcH - cyPx;                                 // to y-up
            float x0 = Mathf.Clamp(cxPx - sidePx * 0.5f, 0f, srcW - sidePx);
            float y0 = Mathf.Clamp(cyUpPx - sidePx * 0.5f, 0f, srcH - sidePx);
            var scale = new Vector2(sidePx / srcW, sidePx / srcH);
            var offset = new Vector2(x0 / srcW, y0 / srcH);
            Graphics.Blit(tex, _faceCrop, scale, offset);

            // ImageNet-normalize into the tensor texture, then convert to the (1,3,448,448) NCHW tensor.
            // Roll-normalization (2D data normalization): the preprocess samples rotated coordinates so
            // the model sees an UPRIGHT face regardless of head roll — laptop users tilt constantly, and
            // a rolled face inside an axis-aligned crop is off-distribution for the model while the
            // calibration only has a linear roll feature to chase the resulting error.
            float roll = _rollNormalize ? _faceMesh.HeadRoll : 0f;
            _preprocess.SetFloat("_RotCos", Mathf.Cos(roll));
            _preprocess.SetFloat("_RotSin", Mathf.Sin(roll));
            _preprocess.SetInt("_Size", _inputSize);
            _preprocess.SetTexture(0, "_Texture", _faceCrop);
            _preprocess.SetTexture(0, "_Tensor", _tensorTex);
            _preprocess.Dispatch(0, _dispatchGroups, _dispatchGroups, 1);
            TextureConverter.ToTensor(_tensorTex, _inputTensor, _nchw);

            _worker.SetInput(INPUT_NAME, _inputTensor);
            _worker.Schedule();

            var yawT = _worker.PeekOutput(OUTPUT_YAW) as Tensor<float>;
            var pitchT = _worker.PeekOutput(OUTPUT_PITCH) as Tensor<float>;
            if (yawT == null || pitchT == null)
                return published;

            var embeddingT = _hasEmbeddingOutput ? _worker.PeekOutput(OUTPUT_EMBEDDING) as Tensor<float> : null;

            if (_asyncReadback)
            {
                //Kick off the non-blocking readbacks; results publish on a later call.
                yawT.ReadbackRequest();
                pitchT.ReadbackRequest();
                embeddingT?.ReadbackRequest();
                _outYaw = yawT;
                _outPitch = pitchT;
                _outEmbedding = embeddingT;
                CaptureFeatureTail(_tailPending);
                _timestampPending = _faceMesh.LastCaptureTimestamp;
                _pendingReadback = true;
                return published;
            }

            //Sync mode: block on the results now. The embedding is read BEFORE the flip-TTA pass below —
            //that pass re-schedules the worker with the mirrored input, which would overwrite the outputs.
            float yaw = DecodeAngleRadians(yawT.DownloadToArray(), out float sigmaYaw);
            float pitch = DecodeAngleRadians(pitchT.DownloadToArray(), out float sigmaPitch);
            if (embeddingT != null)
            {
                _embeddingPublished = embeddingT.DownloadToArray();
                EnsureEmbeddingSized(_embeddingPublished);
            }

            //Flip TTA: run the MIRRORED crop through the model too and average, negating the mirrored yaw
            //(a mirrored face looks the opposite horizontal way; pitch is mirror-invariant).
            if (_flipAugmentation)
            {
                Graphics.Blit(tex, _faceCrop, new Vector2(-scale.x, scale.y), new Vector2(offset.x + scale.x, offset.y));
                //A mirrored face carries NEGATED roll, so the roll-normalization rotates the other way.
                _preprocess.SetFloat("_RotSin", Mathf.Sin(_rollNormalize ? -_faceMesh.HeadRoll : 0f));
                _preprocess.SetTexture(0, "_Texture", _faceCrop);
                _preprocess.SetTexture(0, "_Tensor", _tensorTex);
                _preprocess.Dispatch(0, _dispatchGroups, _dispatchGroups, 1);
                TextureConverter.ToTensor(_tensorTex, _inputTensor, _nchw);
                _worker.SetInput(INPUT_NAME, _inputTensor);
                _worker.Schedule();
                var yawM = _worker.PeekOutput(OUTPUT_YAW) as Tensor<float>;
                var pitchM = _worker.PeekOutput(OUTPUT_PITCH) as Tensor<float>;
                if (yawM != null && pitchM != null)
                {
                    yaw = (yaw - DecodeAngleRadians(yawM.DownloadToArray(), out float mirrorSigmaYaw)) * 0.5f;
                    pitch = (pitch + DecodeAngleRadians(pitchM.DownloadToArray(), out float mirrorSigmaPitch)) * 0.5f;
                    //Both passes measure the same underlying breadth, so the averaged estimate carries the
                    //averaged spread. (Averaging two estimates would strictly shrink the error of the MEAN,
                    //but only if their errors were independent — two views of one frame are not.)
                    sigmaYaw = (sigmaYaw + mirrorSigmaYaw) * 0.5f;
                    sigmaPitch = (sigmaPitch + mirrorSigmaPitch) * 0.5f;
                }
                //Restore the unmirrored crop so the debug thumbnail matches what the primary pass saw.
                Graphics.Blit(tex, _faceCrop, scale, offset);
            }

            CaptureFeatureTail(_tailPublished);
            _timestampPublished = _faceMesh.LastCaptureTimestamp;
            PublishResult(yaw, pitch, sigmaYaw, sigmaPitch, _tailPublished);
            return true;
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

        //Turns decoded gaze angles + the matching feature tail into RawGaze and the calibration features.
        private void PublishResult(float yaw, float pitch, float sigmaYaw, float sigmaPitch, float[] tail)
        {
            //Published together with the features it was decoded alongside, so a consumer that reads both
            //in the same frame can never pair one inference's angles with another's confidence.
            _sigmaPublished = new Vector2(sigmaYaw, sigmaPitch);
            // Rough pre-calibration screen point (calibration refines from Features). yaw -> x, pitch -> y.
            float nx = Mathf.Clamp01(0.5f + yaw * ANGLE_TO_SCREEN_GAIN);
            float ny = Mathf.Clamp01(0.5f - pitch * ANGLE_TO_SCREEN_GAIN);
            _rawGaze = new Vector2(nx * Screen.width, ny * Screen.height);

            var f = _fullFeatures ?? _features;
            // Polynomial feature vector for calibration (see FillGazeFeatures / the field comment).
            FillGazeFeatures(f, yaw, pitch, tail[0], tail[1], tail[2], tail[3]);
            // Direct geometric gaze cue: normalized iris position within each eye (see FillIrisFeatures).
            f[IrisFeatureStart] = tail[4];
            f[IrisFeatureStart + 1] = tail[5];
            f[IrisFeatureStart + 2] = tail[6];
            f[IrisFeatureStart + 3] = tail[7];
            // Shared context block: head translation/depth, eyeLook blendshapes, gaze×pose/translation/
            // distance interaction terms (see HomulerFunctions.FillContextFeatures).
            HomulerFunctions.FillContextFeatures(f, ContextFeatureStart, yaw, pitch,
                tail[0], tail[1], tail, TailContextStart);
            // Softmax-breadth block: lets the fit undo the head-dependent soft-argmax compression
            // (see FillSigmaFeatures).
            FillSigmaFeatures(f, SigmaFeatureStart, yaw, pitch, sigmaYaw, sigmaPitch);
            // Backbone embedding tail (embedding-head personalization; zero-length for unedited models):
            // the raw 512-1280-d GAP vector compressed to 64 dims via the fixed sparse-JL projection.
            if (_embeddingDim > 0 && _embeddingPublished != null)
            {
                if (_embeddingSigns != null)
                    ProjectEmbedding(_embeddingPublished, _embeddingSigns, f, FeatureCount, _embeddingDim);
                else
                    System.Array.Copy(_embeddingPublished, 0, f, FeatureCount,
                        Mathf.Min(_embeddingDim, _embeddingPublished.Length));
            }
        }

        /// <summary>
        /// Length of the calibration feature vector: FillGazeFeatures' 11 terms [gaze-angle polynomial 7,
        /// head pose 4], the 4 iris-offset features (indices 11..14, HomulerFunctions.FillIrisFeatures),
        /// the shared 17-feature context block (head translation/depth, eyeLook blendshapes, gaze
        /// interaction terms — HomulerFunctions.FillContextFeatures), then the 4 softmax-breadth terms
        /// (FillSigmaFeatures). Blocks are APPENDED so the head-pose slots (7/8/9, targeted by the
        /// augmentation jitter) keep their positions.
        /// Changing this length stales saved calibrations (NaN -> raw fallback); recalibrate.
        /// </summary>
        public const int FeatureCount = 15 + HomulerFunctions.ContextFeatureCount + SigmaFeatureCount;
        /// <summary>Number of leading gaze-angle polynomial terms (used by the ensemble backbone).</summary>
        public const int GazeAngleTermCount = 7;
        /// <summary>Index of the first iris-offset feature.</summary>
        public const int IrisFeatureStart = 11;
        /// <summary>Index of the first shared-context feature.</summary>
        public const int ContextFeatureStart = 15;
        /// <summary>Index of the first softmax-breadth feature.</summary>
        public const int SigmaFeatureStart = 15 + HomulerFunctions.ContextFeatureCount;
        /// <summary>[sigmaYaw, sigmaPitch, sigmaYaw*yaw, sigmaPitch*pitch] — see FillSigmaFeatures.</summary>
        public const int SigmaFeatureCount = 4;

        /// <summary>
        /// Softmax-breadth block: the per-axis spread of the model's own output distribution, and its
        /// interaction with the decoded angle.
        ///
        /// WHY THESE ARE NOT REDUNDANT WITH THE POLYNOMIAL. The full-range soft-argmax decode is
        /// systematically COMPRESSED: probability mass in the far bins pulls the expectation toward the
        /// centre bin, so a true +80 deg reads +43.7 deg on a broad head. Compression alone would be
        /// harmless — it is a gain error, and fitting gains is exactly what the calibration does. The
        /// problem is that the compression factor is a function of the BREADTH, and the breadth moves
        /// frame to frame with crop framing, head pose and lighting. A polynomial in (yaw, pitch) is a
        /// FIXED map and cannot represent a gain that changes underneath it; given sigma it can, because
        /// sigma*angle is the first-order term of exactly that varying gain. This is the same reasoning as
        /// the gaze x head-pose cross terms in FillContextFeatures, applied to the decode instead of the
        /// geometry.
        ///
        /// The 64-d projected embedding is pre-logit and so carries some of this information already, but
        /// lossily (a random projection of a 512-1280-d vector) and without the explicit product term.
        /// Four extra standardized columns are cheap next to that; ridge's CV-lambda drops them if they do
        /// not pay.
        /// </summary>
        public static void FillSigmaFeatures(float[] f, int start, float yaw, float pitch,
            float sigmaYaw, float sigmaPitch)
        {
            //A NaN here would propagate into the fit and poison every prediction, so an unavailable spread
            //enters as 0 — which standardizes to "the session's mean breadth", i.e. contributes nothing.
            if (float.IsNaN(sigmaYaw) || float.IsInfinity(sigmaYaw)) sigmaYaw = 0f;
            if (float.IsNaN(sigmaPitch) || float.IsInfinity(sigmaPitch)) sigmaPitch = 0f;
            f[start] = sigmaYaw;
            f[start + 1] = sigmaPitch;
            f[start + 2] = sigmaYaw * yaw;
            f[start + 3] = sigmaPitch * pitch;
        }

        /// <summary>
        /// Fills the calibration feature vector with a low-order polynomial of the gaze angles plus linear
        /// head-pose context. The gaze-angle block [yaw, pitch, yaw², pitch², yaw·pitch, yaw³, pitch³] lets
        /// a per-axis LINEAR model represent the non-linear angle→screen map: the even/cross terms handle
        /// the off-centre asymmetry and the yaw–pitch coupling, and the odd cubic terms are a pole-free
        /// stand-in for tan (tan x ≈ x + x³/3) that extends reach at the corners — the classic 2nd-order
        /// eye-tracking calibration polynomial with a cubic reach term. Head pose stays linear (it shifts
        /// the mapping but the eye angle is the dominant signal). The old constant Screen.width/height
        /// features were dropped (they standardize to zero, i.e. carry no signal). Order is irrelevant to
        /// the standardized ridge/MLP; keeping it fixed is what matters for train/predict agreement.
        /// </summary>
        public static void FillGazeFeatures(float[] f, float yaw, float pitch,
            float headYaw, float headPitch, float headRoll, float headArea)
        {
            f[0] = yaw;
            f[1] = pitch;
            f[2] = yaw * yaw;
            f[3] = pitch * pitch;
            f[4] = yaw * pitch;
            f[5] = yaw * yaw * yaw;
            f[6] = pitch * pitch * pitch;
            f[7] = headYaw;
            f[8] = headPitch;
            f[9] = headRoll;
            f[10] = headArea;
        }

        //Softmax-expectation window: bins outside argmax ± this take no part in the expectation.
        //NO LONGER THE DEFAULT — see DecodeAngleRadians. Kept because the property it was written for is
        //real on a PEAKED head, and because reverting is then one call away.
        //
        //The original reasoning: desktop gaze spans ~±25° (~6 of the 90 4°-bins); the other ~80 bins carry
        //only softmax noise, and any probability mass there pulls the full-range expectation toward the
        //centre — the classic soft-argmax compression, worst at the screen corners. All true. The step that
        //does not follow is the conclusion that the window is therefore safer. What the window actually
        //does is make the output depend on WHICH LOBE HOLDS THE ARGMAX, and the MobileGaze heads are not
        //peaked: measured against mobilenetv2_gaze.onnx on a well-framed, stable face crop, the softmax
        //maximum runs about 6x uniform (0.065 against 0.011) spread over several lobes. The argmax then
        //hops between lobes on noise, and the decoded angle hops the whole distance between them.
        //
        //Measured, same face, ten consecutive frames, no deliberate eye movement:
        //  windowed          +158, +19, -20, +18, +158, +158, +158, +17, -20, +159
        //  full expectation  +3.3, -2.4, +1.3, +7.4, +9.6, +1.1, +5.2, +1.9, +2.5, +12.7
        //Within one 12-sample fixation the windowed decode reached a standard deviation of 49°. No
        //calibration can fit that; the compression the window was avoiding is a SCALE error and the
        //calibration absorbs scale errors by construction.
        public const int DecodeWindowBins = 5;

        /// <summary>
        /// Decode of one output head: softmax over ALL bins, expected bin index, then map bin -> angle
        /// (index * bin_width - offset, in degrees) and return radians. Mutates the passed array in place
        /// (it is an owned per-frame readback buffer).
        ///
        /// This is what the reference implementation these exports come from does
        /// (yakhyo/gaze-estimation, "MobileGaze": <c>sum(softmax * idx) * 4 - 180</c>). UnitEye previously
        /// took the expectation over a ±<see cref="DecodeWindowBins"/> window around the argmax; see the
        /// comment on that constant for the measurements that changed it back.
        ///
        /// CAVEAT ON THE EVIDENCE: verified on one machine, one camera, one participant, with
        /// mobilenetv2_gaze.onnx. The failure it fixes is stark enough to act on, but it has not been
        /// re-measured across the other exports or other users. <see cref="DecodeAngleRadiansWindowed"/>
        /// is still there if this turns out to be narrower than it looks.
        /// </summary>
        public static float DecodeAngleRadians(float[] bins) => DecodeAngleRadians(bins, out _);

        /// <summary>
        /// As <see cref="DecodeAngleRadians(float[])"/>, and additionally reports the SPREAD of the softmax
        /// distribution (its standard deviation, converted from bins to radians) — free, because the
        /// expectation loop already has the probabilities.
        ///
        /// This number is the head's own confidence and it is worth more than it looks. These heads are
        /// broad (the measurement on the <see cref="DecodeWindowBins"/> comment: max ~6x uniform, several
        /// lobes), and crucially they are VARIABLY broad — breadth moves with crop framing, head pose and
        /// lighting. Two consequences the rest of the pipeline consumes:
        ///  - It is a per-frame QUALITY signal, where everything downstream previously had only static
        ///    uncertainty (the AOI error ellipse is measured once at evaluation time). See GazeConfidence.
        ///  - It parameterizes the soft-argmax COMPRESSION. A broad distribution pulls the expectation
        ///    toward the centre bin; that is why the full expectation reads bin 65 as +43.7 deg rather than
        ///    +80. Calibration absorbs a CONSTANT scale error by construction, but this one is not constant
        ///    — it varies with the breadth — so sigma (and its interaction with the angle) enters the
        ///    calibration feature vector to give the fit a basis for undoing it. See FillGazeFeatures.
        /// </summary>
        public static float DecodeAngleRadians(float[] bins, out float sigmaRadians)
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

            float expectation = 0f, expectationSq = 0f;
            if (sum > 0f)
                for (int i = 0; i < bins.Length; i++)
                {
                    float p = bins[i] / sum;
                    expectation += p * i;
                    expectationSq += p * i * i;
                }

            //Var[i] = E[i^2] - E[i]^2, clamped at 0 (float cancellation can make it slightly negative when
            //the distribution is a single spike). Bins -> degrees -> radians via the same bin width as the
            //mean; the OFFSET does not apply, a spread is a difference of angles.
            float varianceBins = Mathf.Max(0f, expectationSq - expectation * expectation);
            sigmaRadians = Mathf.Sqrt(varianceBins) * BIN_WIDTH_DEG * Mathf.Deg2Rad;

            float degrees = expectation * BIN_WIDTH_DEG - ANGLE_OFFSET_DEG;
            return degrees * Mathf.Deg2Rad;
        }

        /// <summary>
        /// The previous decode: softmax and expectation restricted to ±<see cref="DecodeWindowBins"/> bins
        /// around the argmax. Retained for comparison and for the regression test that shows why it is not
        /// the default. Mutates the passed array in place.
        /// </summary>
        public static float DecodeAngleRadiansWindowed(float[] bins) => DecodeAngleRadiansWindowed(bins, out _);

        /// <summary>
        /// Windowed decode with the same spread output as <see cref="DecodeAngleRadians(float[], out float)"/>,
        /// so the two are drop-in interchangeable at the call sites. Note the sigma this reports is the
        /// spread WITHIN THE WINDOW and is therefore bounded by it — it cannot see the far-bin mass that
        /// makes the windowed decode hop, so it is not a usable confidence signal. Another reason the full
        /// expectation is the default.
        /// </summary>
        public static float DecodeAngleRadiansWindowed(float[] bins, out float sigmaRadians)
        {
            float max = float.NegativeInfinity;
            int argmax = 0;
            for (int i = 0; i < bins.Length; i++)
                if (bins[i] > max) { max = bins[i]; argmax = i; }

            int lo = Mathf.Max(0, argmax - DecodeWindowBins);
            int hi = Mathf.Min(bins.Length - 1, argmax + DecodeWindowBins);

            float sum = 0f;
            for (int i = lo; i <= hi; i++)
            {
                bins[i] = Mathf.Exp(bins[i] - max);
                sum += bins[i];
            }

            float expectation = 0f, expectationSq = 0f;
            if (sum > 0f)
                for (int i = lo; i <= hi; i++)
                {
                    float p = bins[i] / sum;
                    expectation += p * i;
                    expectationSq += p * i * i;
                }

            float varianceBins = Mathf.Max(0f, expectationSq - expectation * expectation);
            sigmaRadians = Mathf.Sqrt(varianceBins) * BIN_WIDTH_DEG * Mathf.Deg2Rad;

            float degrees = expectation * BIN_WIDTH_DEG - ANGLE_OFFSET_DEG;
            return degrees * Mathf.Deg2Rad;
        }

        public void Dispose()
        {
            _worker?.Dispose();
            _worker = null;
            _outYaw = null;
            _outPitch = null;
            _outEmbedding = null;
            _pendingReadback = false;
            _inputTensor?.Dispose();
            _inputTensor = null;
            if (_faceCrop != null) { _faceCrop.Release(); Object.Destroy(_faceCrop); }
            if (_tensorTex != null) { _tensorTex.Release(); Object.Destroy(_tensorTex); }
        }
    }
}
#endif
