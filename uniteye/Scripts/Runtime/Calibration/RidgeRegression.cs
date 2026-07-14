using MathNet.Numerics.LinearAlgebra;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
namespace UnitEye
{

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
        //Robust (IRLS/Huber) fitting parameters.
        private const int RobustIterations = 4;      // ordinary ridge on pass 0, then reweight a few times
        private const int MinSamplesForRobust = 8;   // below this a robust scale is unstable -> plain ridge
        private const float HuberTuning = 1.345f;    // 95% efficiency vs least squares under normal residuals

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

            //Iteratively reweighted least squares with Huber weights: a blink, a saccade caught mid-dwell,
            //or a momentary tracking glitch produces a calibration sample whose residual is far larger than
            //the rest, and plain least squares chases those outliers. Pass 0 is ordinary ridge (unit
            //weights); each later pass down-weights samples by their residual (w = 1 within a robust band,
            //falling off as delta/|r| beyond it) and refits. On clean data the weights stay ~1 so the result
            //matches ordinary ridge; tiny sets (no stable robust scale) stay ordinary least squares.
            int rows = input.RowCount;
            var weights = Vector<float>.Build.Dense(rows, 1f);
            int iterations = rows >= MinSamplesForRobust ? RobustIterations : 1;
            for (int iter = 0; iter < iterations; iter++)
            {
                W = SolveWeightedRidge(input, output, weights);
                if (iter < iterations - 1)
                    weights = HuberWeights(input, output, W);
            }
            B = W[0];

            return Test(x, y);
        }

        /// <summary>
        /// Solves the (row-)weighted ridge normal equations. Scaling each row of the design matrix and
        /// target by sqrt(weight) turns the ordinary ridge solve into a weighted least squares solve; the
        /// intercept column is left unpenalized exactly as in the unweighted path.
        /// </summary>
        private Vector<float> SolveWeightedRidge(Matrix<float> input, Vector<float> output, Vector<float> weights)
        {
            int rows = input.RowCount, cols = input.ColumnCount;
            var weightedInput = Matrix<float>.Build.Dense(rows, cols);
            var weightedOutput = Vector<float>.Build.Dense(rows);
            for (int r = 0; r < rows; r++)
            {
                float sw = (float)Math.Sqrt(Math.Max(0f, weights[r]));
                for (int c = 0; c < cols; c++)
                    weightedInput[r, c] = input[r, c] * sw;
                weightedOutput[r] = output[r] * sw;
            }

            var A = weightedInput.TransposeThisAndMultiply(weightedInput);
            Matrix<float> I = Matrix<float>.Build.DenseIdentity(A.RowCount, A.RowCount);
            I *= Lambda;
            //Standard ridge does NOT penalize the intercept (see the note that used to live in Train): with
            //standardized zero-mean features the bias column is orthogonal to the rest, so penalizing it just
            //shrinks every prediction toward screen coordinate 0.
            if (Affine)
                I[0, 0] = 0f;
            A += I;

            return A.QR().Solve(weightedInput.TransposeThisAndMultiply(weightedOutput));
        }

        /// <summary>
        /// Huber sample weights from the current residuals: w = 1 for residuals within delta = 1.345*sigma
        /// of zero, then delta/|r| beyond it, where sigma = MAD/0.6745 is a robust (outlier-resistant) scale.
        /// Returns all-ones (no down-weighting) when the residuals are essentially identical.
        /// </summary>
        private static Vector<float> HuberWeights(Matrix<float> input, Vector<float> output, Vector<float> coefficients)
        {
            int rows = input.RowCount, cols = input.ColumnCount;
            var residual = new float[rows];
            for (int r = 0; r < rows; r++)
            {
                double predicted = 0.0;
                for (int c = 0; c < cols; c++)
                    predicted += input[r, c] * coefficients[c];
                residual[r] = (float)(output[r] - predicted);
            }

            float median = Median(residual);
            var absoluteDeviation = new float[rows];
            for (int i = 0; i < rows; i++)
                absoluteDeviation[i] = Math.Abs(residual[i] - median);
            float sigma = Median(absoluteDeviation) / 0.6745f;

            var weights = Vector<float>.Build.Dense(rows, 1f);
            if (sigma < 1e-6f)
                return weights; // residuals essentially identical -> nothing to down-weight

            float delta = HuberTuning * sigma;
            for (int i = 0; i < rows; i++)
            {
                float magnitude = Math.Abs(residual[i]);
                if (magnitude > delta)
                    weights[i] = delta / magnitude;
            }
            return weights;
        }

        private static float Median(float[] values)
        {
            int n = values.Length;
            if (n == 0) return 0f;
            var sorted = (float[])values.Clone();
            Array.Sort(sorted);
            return (n & 1) == 1 ? sorted[n / 2] : 0.5f * (sorted[n / 2 - 1] + sorted[n / 2]);
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
            //Hot path (called for X and Y every frame under the default RidgeRegression calibration).
            //Compute W·[bias?, standardized-x...] directly instead of building a List, a ToArray and a
            //DenseVector each call — same arithmetic and summation order, zero per-frame allocation.

            //Never throw on a model/feature dimensionality mismatch (e.g. a provider that supplies no
            //feature vector, or a calibration trained for a different feature set) — return NaN so the
            //caller's NaN handling / raw-gaze fallback applies instead of a per-frame exception.
            int bias = Affine ? 1 : 0;
            if (W == null || bias + x.Length != W.Count)
                return float.NaN;

            // apply the stored standardization, models from older versions have none
            bool standardize = FeatureMean != null && FeatureStd != null &&
                               FeatureMean.Count == x.Length && FeatureStd.Count == x.Length;

            float y = 0f;
            // bias term first (W[0] * 1.0), matching the old [1, x...] ordering
            if (Affine)
                y = W[0];
            for (int i = 0; i < x.Length; i++)
            {
                float xi = standardize ? (x[i] - FeatureMean[i]) / FeatureStd[i] : x[i];
                y += W[bias + i] * xi;
            }

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
}
