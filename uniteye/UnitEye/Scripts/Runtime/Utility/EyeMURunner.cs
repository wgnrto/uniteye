using MediaPipe.Holistic;
using System.Collections.Generic;
using UnitEye;
using Unity.Barracuda;
using UnityEngine;

public class EyeMURunner
{
    const int IMG_SIZE = 128;

    private HolisticPipeline _holisticPipeline;
    private EyeMUResource _eyeMUResource;
    private Model _model;
    private IWorker _worker;

    private float _detectionThreshold = 0.5f;

    #region MediaPipe head data
    public float[] HeadEyeCorners => _holisticPipeline.facePipeline.GetEyeCorners();
    public float HeadYaw => _holisticPipeline.facePipeline.GetFaceRegionYaw();
    public float HeadPitch => _holisticPipeline.facePipeline.GetFaceRegionPitch();
    public float HeadRoll => _holisticPipeline.facePipeline.GetFaceRegionRoll();
    public float HeadArea => _holisticPipeline.facePipeline.GetFaceRegionArea();
    public float[] HeadGeom  { get { return new float[4] { HeadYaw, HeadPitch, HeadRoll, HeadArea }; } }
    #endregion

    public float[] Embedding1Output { get; private set; } = new float[8];
    public float[] Embedding2Output { get; private set; } = new float[4];
    public float[] NetworkOutput { get; private set; } = new float[2];

    public float DetectionThreshold
    { 
        get => _detectionThreshold; 
        set { 
            _detectionThreshold = value;
            _holisticPipeline.facePipeline.SetFaceDetectionThreshold(_detectionThreshold);
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

    public EyeMURunner(HolisticPipeline holisticPipeline, float detectionThreshold)
    {
        _holisticPipeline = holisticPipeline;
        DetectionThreshold = detectionThreshold;
        _eyeMUResource = Resources.Load<EyeMUResource>("EyeMU");

        _model = ModelLoader.Load(_eyeMUResource.modelAsset);
        _worker = _model.CreateWorker();

        //Prepare tensor textures for random write
        _leftEyeTextureTensor.enableRandomWrite = true;
        _leftEyeTextureTensor.Create();

        _rightEyeTextureTensor.enableRandomWrite = true;
        _rightEyeTextureTensor.Create();
    }

    public List<float> GetFeatures()
    {
        List<float> features = new List<float>();

        if (_holisticPipeline == null) return features;

        //features.AddRange(Embedding1Output);
        features.AddRange(Embedding2Output);
        features.AddRange(NetworkOutput);
        //features.AddRange(HeadEyeCorners);
        features.AddRange(HeadGeom);

        features.Add(Screen.width);
        features.Add(Screen.height);

        return features;
    }

    /// <summary>
    /// Perform HolisticBarracuda inference based on WebCamInput.
    /// </summary>
    /// <param name="webCamInput">WebCamInput to use</param>
    /// <returns>true if execution completed, false if not</returns>
    private bool ComputeEyes(WebCamInput webCamInput)
    {
        _holisticPipeline.ProcessImage(webCamInput.inputImageTexture, HolisticInferenceType.face_only);
        if (!_holisticPipeline.facePipeline.IsUserPresent()) return false;

        //Left Eye Texture needs to be flipped to match EyeMU
        //263 and 362 are the mesh vertex indices for the eye corners of the left eye
        _leftEyeTexture = Functions.FlipTexture(Functions.GetEyeTexture(_holisticPipeline, webCamInput.webCamTexture, 263, 362));

        //133 and 33 are the mesh vertex indices for the eye corners of the right eye
        _rightEyeTexture = Functions.GetEyeTexture(_holisticPipeline, webCamInput.webCamTexture, 133, 33);

        return true;
    }

    /// <summary>
    /// Perform gaze location inference based on WebCamInput.
    /// </summary>
    /// <param name="webCamInput">WebCamInput to use</param>
    /// <returns>true if execution completed, false if not</returns>
    public bool PerformInference(WebCamInput webCamInput) 
    {
        if (_holisticPipeline == null) return false;

        if (!ComputeEyes(webCamInput)) return false;

        var corners = new Tensor(1, 1, 1, 8);
        for (int i = 0; i < 8; i++)
        {
            corners[0, 0, 0, i] = HeadEyeCorners[i];
        }

        var pose = new Tensor(1, 1, 1, 4);
        pose[0, 0, 0, 0] = HeadYaw;
        pose[0, 0, 0, 1] = HeadPitch;
        pose[0, 0, 0, 2] = HeadRoll;
        pose[0, 0, 0, 3] = HeadArea;

        //Use unique textures to avoid memory leak
        Graphics.Blit(_leftEyeTexture, LeftEyeTexture);
        _leftEyeTextureTensor = Functions.PreprocessImage(LeftEyeTexture, _leftEyeTextureTensor, _eyeMUResource.preprocessCompute);
        var leftTensor = new Tensor(_leftEyeTextureTensor, 3);

        Graphics.Blit(_rightEyeTexture, RightEyeTexture);
        _rightEyeTextureTensor = Functions.PreprocessImage(RightEyeTexture, _rightEyeTextureTensor, _eyeMUResource.preprocessCompute);
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
            Embedding2Output[i] = dense7[0, 0, 0, i] ;// / 10.0f;
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
    /// Dispose of all neccessary ComputeBuffers or Barracuda workers.
    /// </summary>
    public void Dispose()
    {
        _worker?.Dispose();
        _worker = null;
    }
}
