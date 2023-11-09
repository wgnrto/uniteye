using BrightWire;
using BrightWire.ExecutionGraph;
using BrightWire.Models;
using System.IO;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using UnitEye;
using Newtonsoft.Json;

/// <summary>
/// This class implements a lightweight multi-layer-perceptron.
/// Using the MLP allows for a more robust calibration.
/// </summary>
public class MLP
{
    private IGraphEngine _executionEngine;
    private GraphFactory _graph;
    private GraphModel _graphModel;
    private ILinearAlgebraProvider _lap;

    /// <summary>
    /// Train a MLP based on data.
    /// </summary>
    /// <param name="x">Input data</param>
    /// <param name="y">Target data</param>
    /// <returns>Calibration accuracy message</returns>
    public string Train(float[][] x, Vector2[] y)
    {
        _lap = BrightWireProvider.CreateLinearAlgebra(false);
        _graph = new GraphFactory(_lap);

        Debug.Log("Loading data");

        var dataTable = ConvertToDataTable(_graph, x, y);
        //Split data for training and test data, default is 80% training data
        var data = dataTable.Split();

        Debug.Log("Data loaded");

        var errorMetric = _graph.ErrorMetric.Quadratic;

        //Configure the network properties
        _graph.CurrentPropertySet
            .Use(_graph.GradientDescent.Adam)
            //.Use(graph.GaussianWeightInitialisation(false, 0.1f, GaussianVarianceCalibration.SquareRoot2N));
            .Use(_graph.XavierWeightInitialisation());

        //Create training and test data
        var trainingData = _graph.CreateDataSource(data.Training);
        var testData = trainingData.CloneWith(data.Test);

        //Create the training engine and schedule a training rate change
        const float LEARNING_RATE = 0.02f;
        var engine = _graph.CreateTrainingEngine(trainingData, LEARNING_RATE, 32);

        //Create the network
        var network = _graph.Connect(engine)
            .AddFeedForward(outputSize: 32)
            .Add(_graph.ReluActivation())
            .AddFeedForward(outputSize: 16)
            .Add(_graph.ReluActivation())
            .AddFeedForward(outputSize: 2)
            .AddBackpropagation(errorMetric);

        Debug.Log("Training started");

        //Train the network for twenty iterations, saving the model on each improvement
        ExecutionGraph bestGraph = null;
        engine.Train(100, testData, errorMetric, model => { _graphModel = model; bestGraph = model.Graph; });

        //Export the final model and execute it on the training set
        _executionEngine = _graph.CreateEngine(bestGraph ?? engine.Graph);

        var output = _executionEngine.Execute(testData, batchSize: 1);
        var error = Test(output);

        var errorXInCm = Functions.PixelsToMm(error.x) * 0.1f;
        var errorYInCm = Functions.PixelsToMm(error.y) * 0.1f;

        //Return accuracy message
        return $"MLP Training done. RMSE X: {errorXInCm}cm | RMSE Y: {errorYInCm}cm.";
    }

    /// <summary>
    /// Saves MLP to json file.
    /// </summary>
    /// <param name="fileName">Filename</param>
    public void Save(string fileName)
    {
        string filepath = Application.streamingAssetsPath + $"/Calibration Files/MLP/";
        //Create directory if it doesn't exist
        if (!Directory.Exists(filepath))
            Directory.CreateDirectory(filepath);
        filepath += fileName;

        var json = JsonConvert.SerializeObject(_graphModel);
        File.WriteAllText(filepath, json);
    }

    /// <summary>
    /// Loads MLP from json file, uses default if no custom calibration is found.
    /// </summary>
    /// <param name="fileName">Filename</param>
    public static MLP Load(string fileName)
    {
        var mlp = new MLP();

        string filepath = Application.streamingAssetsPath + $"/Calibration Files/MLP/{fileName}";
        string jsonString;

        mlp._lap = BrightWireProvider.CreateLinearAlgebra(false);
        mlp._graph = new GraphFactory(mlp._lap);
        
        try
        {
            jsonString = File.ReadAllText(filepath);
        } 
        catch
        {
            //Default file fallback
            Debug.LogWarning("Calibrated MLP file not found, using default file! Please run a MLCalibration!");
            var calibrations = Resources.Load<CalibrationResource>("CalibrationDefaultFiles");
            jsonString = calibrations.mlpAsset.ToString();
        }

        //Create graph execution engine
        mlp._graphModel = JsonConvert.DeserializeObject<GraphModel>(jsonString);
        mlp._executionEngine = mlp._graph.CreateEngine(mlp._graphModel.Graph);

        return mlp;
    }

    private (float Min, float Max) CalculateMinMax(Vector<float> vector)
    {
        float min = 0f, max = 0f;
        foreach (var val in vector.Enumerate(Zeros.AllowSkip))
        {
            if (val > max)
                max = val;
            if (val < min)
                min = val;
        }
        return (min, max);
    }

    /// <summary>
    /// Predict with the MLP.
    /// </summary>
    /// <param name="features">Network feature outputs</param>
    /// <returns>Predicted gaze location</returns>
    public Vector2 Predict(float[] features)
    {
        var result = Vector2.zero;

        if (_executionEngine == null) return result;

        var featVector = Vector<float>.Build.Dense(features);

        // No normalization for now
        //var (min, max) = CalculateMinMax(featVector);
        //var range = max - min;
        //if (range > 0)
        //    featVector.MapInplace(v => (v - min) / range);

        var data = _graph.CreateDataSource(new[] { FloatVector.Create(featVector.ToArray()) });
        var output = _executionEngine.Execute(data, batchSize: 1);
        result.Set(output[0].Output[0].Data[0], output[0].Output[0].Data[1]);

        return result;
    }

    /// <summary>
    /// Test MLP accuracy.
    /// </summary>
    /// <param name="output">MLP output</param>
    /// <returns>X and Y errors</returns>
    public (float x, float y) Test(IReadOnlyList<ExecutionResult> output)
    {
        float errorX = 0.0f, errorY = 0.0f;
        for (int i = 0; i < output.Count; i++)
        {
            var predX = output[i].Output[0].Data[0];
            var predY = output[i].Output[0].Data[1];

            var targX = output[i].Target[0].Data[0];
            var targY = output[i].Target[0].Data[1];

            errorX += Mathf.Pow(predX - targX, 2);
            errorY += Mathf.Pow(predY - targY, 2);
        }

        return (Mathf.Sqrt(errorX / output.Count), Mathf.Sqrt(errorY / output.Count));
    }

    private IDataTable ConvertToDataTable(GraphFactory graph, float[][] X, Vector2[] Y)
    {
        var featureTable = BrightWireProvider.CreateDataTableBuilder();
        var targetTable = BrightWireProvider.CreateDataTableBuilder();

        featureTable.AddColumn(ColumnType.Vector, "Features");
        targetTable.AddColumn(ColumnType.Vector, "Target", isTarget: true);

        for (int i = 0; i < X.Length; i++)
        {
            var featVec = new FloatVector { Data = X[i] };
            var targetVec = new FloatVector { Data = new float[] { Y[i].x, Y[i].y } };
            featureTable.Add(featVec);
            targetTable.Add(targetVec);
        }

        // Only normalize the features not the target output
        // No normalization for now
        var feat = featureTable.Build();//.Normalise(NormalisationType.FeatureScale);
        return feat.Zip(targetTable.Build());
    }
}