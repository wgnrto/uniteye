using Mediapipe.Unity;
using Mediapipe.Unity.FaceMesh;
using System.Collections.Generic;
using UnitEye;
using Unity.Barracuda;
using UnityEngine;
using static UnitEye.HomulerFunctions;
using Screen = UnityEngine.Screen;

public class HomulerEyeMURunner
{
    const int IMG_SIZE = 128;
    private FaceMeshSolution _faceMesh;
    private EyeMUResource _eyeMUResource;
    private Model _model;
    private IWorker _worker;

    public float[] Embedding1Output { get; private set; } = new float[8];
    public float[] Embedding2Output { get; private set; } = new float[4];
    public float[] NetworkOutput { get; private set; } = new float[2];

    public List<float> Features
    {
        get
        {
            var features = new List<float>();
            features.AddRange(Embedding2Output);
            features.AddRange(NetworkOutput);
            //features.AddRange(HeadGeom);
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
        _worker = _model.CreateWorker();

        //Prepare tensor textures for random write
        _leftEyeTextureTensor.enableRandomWrite = true;
        _leftEyeTextureTensor.Create();

        _rightEyeTextureTensor.enableRandomWrite = true;
        _rightEyeTextureTensor.Create();
    }

    /// <summary>
    /// Perform gaze location inference based on WebCamInput.
    /// </summary>
    /// <param name="webCamInput">WebCamInput to use</param>
    /// <returns>true if execution completed, false if not</returns>
    public bool PerformInference(WebCamSource webcam)
    {
        var webcamTexture = (WebCamTexture) webcam.GetCurrentTexture();

        if (!webcam.isPrepared || webcamTexture == null)
            return false;

        if (!ComputeEyes(webcamTexture))
            return false;

        var corners = new Tensor(1, 1, 1, 8);
        for (int i = 0; i < 8; i++)
        {
            corners[0, 0, 0, i] = _faceMesh.EyeCorners[i];
        }

        var pose = new Tensor(1, 1, 1, 4);
        pose[0, 0, 0, 0] = _faceMesh.HeadYaw;
        pose[0, 0, 0, 1] = _faceMesh.HeadPitch;
        pose[0, 0, 0, 2] = _faceMesh.HeadRoll;
        pose[0, 0, 0, 3] = _faceMesh.HeadArea;

        //Use unique textures to avoid memory leak
        Graphics.Blit(_leftEyeTexture, LeftEyeTexture);
        _leftEyeTextureTensor = PreprocessImage(LeftEyeTexture, _leftEyeTextureTensor, _eyeMUResource.preprocessCompute);
        var leftTensor = new Tensor(_leftEyeTextureTensor, 3);

        Graphics.Blit(_rightEyeTexture, RightEyeTexture);
        _rightEyeTextureTensor = PreprocessImage(RightEyeTexture, _rightEyeTextureTensor, _eyeMUResource.preprocessCompute);
        var rightTensor = new Tensor(_rightEyeTextureTensor, 3);

        var inputs = new Dictionary<string, Tensor>();
        inputs["input_1"] = leftTensor;
        inputs["input_2"] = rightTensor;
        inputs["input_4"] = corners;
        inputs["input_5"] = pose;

        _worker.Execute(inputs);

        // Get network outputs
        Tensor dense6 = _worker.PeekOutput("dense_6");
        for (int i = 0; i < dense6.channels; i++)
        {
            // Division based on: https://github.com/FIGLAB/EyeMU/blob/master/flask2/static/svr.js#L109
            Embedding1Output[i] = dense6[0, 0, 0, i];// / 100.0f;
        }

        Tensor dense7 = _worker.PeekOutput("dense_7");
        for (int i = 0; i < dense7.channels; i++)
        {
            // Division based on: https://github.com/FIGLAB/EyeMU/blob/master/flask2/static/svr.js#L110
            Embedding2Output[i] = dense7[0, 0, 0, i];// / 10.0f;
        }

        Tensor final = _worker.PeekOutput("dense_8");
        for (int i = 0; i < final.channels; i++)
        {
            // NetworkOutput[i] = final[0, 0, 0, i];
        }

        NetworkOutput[0] = final[0, 0, 0, 0] * Screen.width;
        NetworkOutput[1] = final[0, 0, 0, 1] * Screen.height;

        // Cleanup memory
        foreach (var tensor in inputs.Values)
        {
            tensor.Dispose();
        }

        // Clear Dictionary
        inputs.Clear();

        return true;
    }

    /// <summary>
    /// Perform HolisticBarracuda inference based on WebCamInput.
    /// </summary>
    /// <param name="webCamInput">WebCamInput to use</param>
    /// <returns>true if execution completed, false if not</returns>
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
    /// Dispose of all neccessary ComputeBuffers or Barracuda workers.
    /// </summary>
    public void Dispose()
    {
        _worker?.Dispose();
        _worker = null;
    }
}
