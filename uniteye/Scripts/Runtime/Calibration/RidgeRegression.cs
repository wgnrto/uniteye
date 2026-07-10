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
        if (reader.TokenType == JsonToken.Null) return null;

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

    //Per feature mean and standard deviation computed on the training data.
    //When set, features are standardized before training and prediction so all features
    //share one scale and the regularization treats them equally.
    //Models saved by older versions do not contain these fields and are loaded with null,
    //which skips standardization and keeps them fully backwards compatible.
    [JsonConverter(typeof(VectorConverter))]
    public Vector<float> FeatureMean { get; set; }

    [JsonConverter(typeof(VectorConverter))]
    public Vector<float> FeatureStd { get; set; }

    //Tracks which default files we have already warned about, so the "no calibration" warning is
    //logged once per session instead of on every OnValidate/Start reload (which spams the console).
    private static readonly System.Collections.Generic.HashSet<string> _warnedDefaults = new System.Collections.Generic.HashSet<string>();

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
            //No personal calibration file for this user. Fall back to a shipped default if one exists,
            //otherwise return null so the caller can fall back to raw (uncalibrated) gaze. We no longer
            //ship a default fit: the old one-person defaults ignored the eye-gaze signal and extrapolated
            //off-screen for anyone else (Reg_Y regularized to a near-constant top edge, Reg_X driven by
            //that person's head geometry), so the crosshair corner-locked and looked broken before the
            //first calibration.
            var calibrations = Resources.Load<CalibrationResource>("CalibrationDefaultFiles");
            TextAsset defaultAsset = null;
            if (calibrations != null)
            {
                if (defaultFilename == "Reg_X.json") defaultAsset = calibrations.regXAsset;
                else if (defaultFilename == "Reg_Y.json") defaultAsset = calibrations.regYAsset;
            }

            if (defaultAsset == null)
            {
                //Warn only once per default file per session
                if (_warnedDefaults.Add(defaultFilename))
                    Debug.LogWarning("No RidgeRegression calibration found; using raw (uncalibrated) gaze. Please run a RidgeRegression calibration for accurate results.");
                return null;
            }

            if (_warnedDefaults.Add(defaultFilename))
                Debug.LogWarning("Calibrated RidgeRegression files not found, using default files! Please run a RidgeRegression calibration!");
            jsonString = defaultAsset.ToString();
        }

        return JsonConvert.DeserializeObject<RidgeRegression>(jsonString);
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

        ComputeStandardization(input);
        input = StandardizeMatrix(input);

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
    /// Computes per feature mean and standard deviation over the training data.
    /// Constant features get a standard deviation of one so they become zero after
    /// centering instead of causing a division by zero.
    /// </summary>
    /// <param name="input">Training data matrix, one sample per row</param>
    private void ComputeStandardization(Matrix<float> input)
    {
        var mean = Vector<float>.Build.Dense(input.ColumnCount);
        var std = Vector<float>.Build.Dense(input.ColumnCount);

        for (int c = 0; c < input.ColumnCount; c++)
        {
            var column = input.Column(c);

            double sum = 0.0;
            for (int r = 0; r < column.Count; r++)
                sum += column[r];
            var columnMean = (float)(sum / column.Count);

            double squaredSum = 0.0;
            for (int r = 0; r < column.Count; r++)
                squaredSum += (column[r] - columnMean) * (column[r] - columnMean);
            var columnStd = (float)Math.Sqrt(squaredSum / column.Count);

            mean[c] = columnMean;
            std[c] = columnStd < 1e-6f ? 1.0f : columnStd;
        }

        FeatureMean = mean;
        FeatureStd = std;
    }

    /// <summary>
    /// Applies the stored standardization to a data matrix, one sample per row.
    /// </summary>
    private Matrix<float> StandardizeMatrix(Matrix<float> input)
    {
        if (FeatureMean == null || FeatureStd == null) return input;

        var standardized = Matrix<float>.Build.Dense(input.RowCount, input.ColumnCount);
        for (int r = 0; r < input.RowCount; r++)
            for (int c = 0; c < input.ColumnCount; c++)
                standardized[r, c] = (input[r, c] - FeatureMean[c]) / FeatureStd[c];

        return standardized;
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

        // apply the stored standardization, models from older versions have none
        if (FeatureMean != null && FeatureStd != null && FeatureMean.Count == x.Length && FeatureStd.Count == x.Length)
        {
            for (int i = 0; i < x.Length; i++)
                xs.Add((x[i] - FeatureMean[i]) / FeatureStd[i]);
        }
        else
        {
            xs.AddRange(x);
        }

        //Never throw on a model/feature dimensionality mismatch (e.g. a provider that supplies no
        //feature vector, or a calibration trained for a different feature set) — return NaN so the
        //caller's NaN handling / raw-gaze fallback applies instead of a per-frame exception.
        if (W == null || xs.Count != W.Count)
            return float.NaN;

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
