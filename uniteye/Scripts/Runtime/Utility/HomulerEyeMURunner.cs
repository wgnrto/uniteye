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

/// <summary>
/// Runs the EyeMU gaze model on Unity's Inference Engine (com.unity.ai.inference),
/// using landmarks from the native homuler MediaPipe FaceMeshSolution.
/// This is the Barracuda-free replacement for the old EyeMURunner + HolisticBarracuda path.
/// </summary>
public class HomulerEyeMURunner
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

    #region Head pose (re-enabled so the feature vector and UnitEyeAPI.GetHeadPose match the former Holistic path)
    public float HeadYaw => _faceMesh.HeadYaw;
    public float HeadPitch => _faceMesh.HeadPitch;
    public float HeadRoll => _faceMesh.HeadRoll;
    public float HeadArea => _faceMesh.HeadArea;
    public float[] HeadGeom => _faceMesh.HeadGeom;
    #endregion

    public List<float> Features
    {
        get
        {
            var features = new List<float>();
            features.AddRange(Embedding2Output);
            features.AddRange(NetworkOutput);
            //Head pose re-enabled -> 12-feature vector, matching the shipped default calibration files
            features.AddRange(HeadGeom);
            features.Add(Screen.width);
            features.Add(Screen.height);

            return features;
        }
    }

    //GUI textures
    public RenderTexture LeftEyeTexture { get; private set; } = new RenderTexture(IMG_SIZE, IMG_SIZE, 0, RenderTextureFormat.ARGB32);
    public RenderTexture RightEyeTexture { get; private set; } = new RenderTexture(IMG_SIZE, IMG_SIZE, 0, RenderTextureFormat.ARGB32);

    //Tensor textures
    private RenderTexture _leftEyeTextureTensor = new RenderTexture(IMG_SIZE, IMG_SIZE, 24, RenderTextureFormat.ARGBHalf);
    private RenderTexture _rightEyeTextureTensor = new RenderTexture(IMG_SIZE, IMG_SIZE, 24, RenderTextureFormat.ARGBHalf);

    //Textures to handle GetEyeTexture()
    private Texture _leftEyeTexture;
    private Texture _rightEyeTexture;

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

        //Eye corners (8) and head pose (4) input tensors
        var corners = new Tensor<float>(new TensorShape(1, 8), _faceMesh.EyeCorners);
        var pose = new Tensor<float>(new TensorShape(1, 4),
            new float[] { _faceMesh.HeadYaw, _faceMesh.HeadPitch, _faceMesh.HeadRoll, _faceMesh.HeadArea });

        //Preprocess eye crops into tensor-format RenderTextures, then convert to NHWC input tensors
        //(shape 1x128x128x3) to match the model's declared image-input layout.
        //HAND-TEST NOTE: shapes/names are now verified to match the model, but pixel normalization and
        //channel order (RGB vs BGR) still ride on the preprocessing shader + TextureConverter and can only
        //be confirmed by looking at live gaze. This is the remaining thing to verify at runtime.
        Graphics.Blit(_leftEyeTexture, LeftEyeTexture);
        _leftEyeTextureTensor = PreprocessImage(LeftEyeTexture, _leftEyeTextureTensor, _eyeMUResource.preprocessCompute);
        var leftTensor = new Tensor<float>(new TensorShape(1, IMG_SIZE, IMG_SIZE, 3));
        TextureConverter.ToTensor(_leftEyeTextureTensor, leftTensor, _nhwcTransform);

        Graphics.Blit(_rightEyeTexture, RightEyeTexture);
        _rightEyeTextureTensor = PreprocessImage(RightEyeTexture, _rightEyeTextureTensor, _eyeMUResource.preprocessCompute);
        var rightTensor = new Tensor<float>(new TensorShape(1, IMG_SIZE, IMG_SIZE, 3));
        TextureConverter.ToTensor(_rightEyeTextureTensor, rightTensor, _nhwcTransform);

        _worker.SetInput(INPUT_LEFT, leftTensor);
        _worker.SetInput(INPUT_RIGHT, rightTensor);
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

        //Cleanup input tensors
        leftTensor.Dispose();
        rightTensor.Dispose();
        corners.Dispose();
        pose.Dispose();

        return true;
    }

    /// <summary>
    /// Compute the left and right eye crop textures from the current face landmarks.
    /// </summary>
    /// <param name="texture">webcam texture</param>
    /// <returns>true if landmarks were available, false if not</returns>
    private bool ComputeEyes(WebCamTexture texture)
    {
        var landmarks = _faceMesh.FaceLandmarks;

        if (landmarks == null)
            return false;

        //Left Eye Texture needs to be flipped to match EyeMU
        //263 and 362 are the mesh vertex indices for the eye corners of the left eye
        _leftEyeTexture = FlipTexture(GetEyeTexture(landmarks, texture, 362, 263));

        //133 and 33 are the mesh vertex indices for the eye corners of the right eye
        _rightEyeTexture = GetEyeTexture(landmarks, texture, 33, 133);

        return true;
    }

    /// <summary>
    /// Dispose of the Inference Engine worker.
    /// </summary>
    public void Dispose()
    {
        _worker?.Dispose();
        _worker = null;
    }
}
#endif
