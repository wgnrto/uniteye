using MathNet.Numerics.LinearAlgebra;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

/// <summary>
/// Custom converter to turn a serialized Vector<float> into a DenseVector
/// </summary>
public class VectorConverter : JsonConverter
{
    public override bool CanConvert(Type objectType)
    {
        return (objectType == typeof(Vector<float>));
    }

    public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer
    )
    {
        JArray jArray = JArray.Load(reader);
        var target = Vector<float>.Build.Dense(jArray.ToObject<float[]>(serializer));
        serializer.Populate(jArray.CreateReader(), target);
        return target;
    }

    public override bool CanWrite
    {
        get { return false; }
    }

    public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer)
    {
        throw new NotImplementedException();
    }
}

/// <summary>
/// This class implements RidgeRegression.
/// This is the most basic calibration type used.
/// </summary>
public class RidgeRegression
{
    [JsonConverter(typeof(VectorConverter))]
    public Vector<float> W { get; set; }

    public float B { get; set; }

    public float Lambda { get; set; }

    public bool Affine { get; set; }

    public RidgeRegression(float lambda, bool affine = true)
    {
        Lambda = lambda;
        Affine = affine;
    }

    /// <summary>
    /// Private Load to be able to load both X and Y RidgeRegression default files correctly.
    /// </summary>
    /// <param name="filename">Filename without a path</param>
    /// <param name="defaultFilename">Default filename without a path</param>
    /// <returns>RidgeRegression instance</returns>
    private static RidgeRegression Load(string filename, string defaultFilename)
    {
        string filepath = Application.streamingAssetsPath + $"/Calibration Files/RidgeRegression/{filename}";
        string jsonString;

        try
        {
            jsonString = File.ReadAllText(filepath);
        }
        catch
        {
            //Default file fallback
            Debug.LogWarning("Calibrated RidgeRegression files not found, using default files! Please run a RidgeRegression calibration!");
            var calibrations = Resources.Load<CalibrationResource>("CalibrationDefaultFiles");
            switch (defaultFilename)
            {
                case "Reg_X.json":
                    jsonString = calibrations.regXAsset.ToString();
                    break;
                case "Reg_Y.json":
                    jsonString = calibrations.regYAsset.ToString();
                    break;
                default:
                    jsonString = "";
                    break;
            }
        }

        RidgeRegression ridgeRegression = JsonConvert.DeserializeObject<RidgeRegression>(
            jsonString
        );

        return ridgeRegression;
    }

    /// <summary>
    /// Load an instance of a X RidgeRegression from a .json file in the StreamingAssets/CalibrationFiles/RidgeRegression folder.
    /// </summary>
    /// <param name="filename">Filename without a path</param>
    /// <returns>X RidgeRegression instance</returns>
    public static RidgeRegression LoadX(string filename)
    {
        return Load(filename, "Reg_X.json");
    }

    /// <summary>
    /// Load an instance of a Y RidgeRegression from a .json file in the StreamingAssets/CalibrationFiles/RidgeRegression folder.
    /// </summary>
    /// <param name="filename">Filename without a path</param>
    /// <returns>Y RidgeRegression instance</returns>
    public static RidgeRegression LoadY(string filename)
    {
        return Load(filename, "Reg_Y.json");
    }

    /// <summary>
    /// Save to .json file in the StreamingAssets/CalibrationFiles/RidgeRegression folder.
    /// </summary>
    /// <param name="filename">Filename without a path</param>
    public void Save(string filename)
    {
        string filepath = Application.streamingAssetsPath + $"/Calibration Files/RidgeRegression/";
        if (!Directory.Exists(filepath))
            Directory.CreateDirectory(filepath);
        filepath += filename;

        var json = JsonConvert.SerializeObject(this);
        File.WriteAllText(filepath, json);
    }

    /// <summary>
    /// Trains this Ridge Regression model.
    /// </summary>
    /// <param name="x">Input values</param>
    /// <param name="y">Expected output values</param>
    /// <returns>The mean squared error</returns>
    public float Train(float[][] x, float[] y)
    {
        var input = Matrix<float>.Build.DenseOfRowArrays(x);

        if (Affine)
        {
            input = input.InsertColumn(
                0,
                Vector<float>.Build.Dense(input.RowCount, Vector<float>.One)
            );
        }

        var output = Vector<float>.Build.Dense(y);
        var A = input.TransposeThisAndMultiply(input);

        Matrix<float> I = Matrix<float>.Build.DenseIdentity(A.RowCount, A.RowCount);
        I *= Lambda;
        A += I;

        W = A.QR().Solve(input.TransposeThisAndMultiply(output));
        B = W[0];

        return Test(x, y);
    }

    /// <summary>
    /// Predicts a value for a certain input.
    /// </summary>
    /// <param name="x">Input features</param>
    public float Predict(float[] x)
    {
        List<float> xs = new List<float>();
        // append one extra entry to for the bias
        if (Affine)
        {
            xs.Add(1.0f);
        }
        xs.AddRange(x);

        var input = Vector<float>.Build.Dense(xs.ToArray());
        var y = W * input;

        return y;
    }

    /// <summary>
    /// Tests the accuracy of the regression model
    /// </summary>
    /// <param name="x">Input data</param>
    /// <param name="y">Groundtruth data</param>
    /// <returns>The mean squared error</returns>
    public float Test(float[][] x, float[] y)
    {
        var error = 0.0f;
        for (int i = 0; i < y.Length; i++)
        {
            var yhat = Predict(x[i]);
            error += MathF.Pow(y[i] - yhat, 2);
        }

        return error / y.Length;
    }
}
