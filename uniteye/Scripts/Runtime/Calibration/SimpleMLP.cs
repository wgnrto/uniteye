using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using UnitEye;
using UnityEngine;
namespace UnitEye
{

    /// <summary>
    /// Dependency-free multilayer perceptron for the ML gaze calibration (replaces the BrightWire-based
    /// implementation; same 12→32→16→2 architecture with ReLU hidden layers and Adam, trained at runtime
    /// on the captured calibration samples).
    ///
    /// Improvements over the BrightWire version: per-feature standardization computed on the training
    /// data and stored with the model (the old implementation had normalization disabled and its
    /// train/predict paths disagreed), targets standardized internally the same way, and the reported
    /// RMSE comes from a random untouched holdout instead of per-epoch checkpoint selection on the test
    /// set. Pure C#: no BrightWire/protobuf DLLs, IL2CPP- and WebGL-safe, and deterministic when a seed
    /// is supplied.
    /// </summary>
    public class SimpleMLP
    {
        // Architecture (kept from the original implementation)
        const int HIDDEN1 = 32;
        const int HIDDEN2 = 16;
        const int OUTPUTS = 2;

        // Training hyperparameters (mirroring the BrightWire configuration where sensible)
        const float LEARNING_RATE = 0.01f;
        const int BATCH_SIZE = 32;
        const int EPOCHS = 200;
        const float HOLDOUT_FRACTION = 0.2f;

        // Adam
        const float BETA1 = 0.9f;
        const float BETA2 = 0.999f;
        const float EPSILON = 1e-8f;

        //Holdout RMSE (cm) per axis of the most recent Train() call on THIS instance; -1 before any. Train
        //already computes these for its return string — publishing them lets the caller feed the calibration
        //validation gate (HomulerGazeCalibration.LastHoldoutRmseCm) instead of parsing prose. [JsonIgnore]:
        //they describe a training run, not the model, and must not land in MLP.json.
        [JsonIgnore] public float LastHoldoutRmseXCm { get; private set; } = -1f;
        [JsonIgnore] public float LastHoldoutRmseYCm { get; private set; } = -1f;

        //Serialized model state. Weights are stored row-major: W1[h * inputCount + i].
        public int InputCount { get; set; }
        public float[] W1 { get; set; }
        public float[] B1 { get; set; }
        public float[] W2 { get; set; }
        public float[] B2 { get; set; }
        public float[] W3 { get; set; }
        public float[] B3 { get; set; }
        public float[] FeatureMean { get; set; }
        public float[] FeatureStd { get; set; }
        public float[] TargetMean { get; set; }
        public float[] TargetStd { get; set; }

        [JsonIgnore] private readonly System.Random _rng;

        public SimpleMLP() : this(-1) { }

        /// <param name="seed">Random seed for weight init and batch shuffling; -1 for time-seeded</param>
        public SimpleMLP(int seed)
        {
            _rng = seed < 0 ? new System.Random() : new System.Random(seed);
        }

        /// <summary>
        /// Trains the network on calibration samples (targets in pixels, matching the old MLP API).
        /// Returns the accuracy message with the holdout RMSE per axis in cm.
        /// </summary>
        public string Train(float[][] x, Vector2[] y,
            CalibrationFeatureAugmentationSettings augmentation = null)
        {
            if (x == null || x.Length < 10)
                throw new ArgumentException("Not enough calibration samples to train the MLP.");

            InputCount = x[0].Length;

            //Random holdout split (honest evaluation), reusing the shared Fisher-Yates permutation
            var permutation = RidgeCalibrationTrainer.RandomPermutation(x.Length, _rng);
            var testCount = (int)Math.Floor(x.Length * HOLDOUT_FRACTION);
            var trainCount = x.Length - testCount;

            var xTrain = new float[trainCount][];
            var yTrain = new Vector2[trainCount];
            for (int i = 0; i < trainCount; i++) { var idx = permutation[testCount + i]; xTrain[i] = x[idx]; yTrain[i] = y[idx]; }
            var xTest = new float[testCount][];
            var yTest = new Vector2[testCount];
            for (int i = 0; i < testCount; i++) { var idx = permutation[i]; xTest[i] = x[idx]; yTest[i] = y[idx]; }

            //Standardization stats from the training portion only
            ComputeStandardization(xTrain, yTrain);

            //Augment only the fitting partition after its statistics are established. The untouched
            //original holdout below remains the reported accuracy measure.
            xTrain = CalibrationFeatureAugmentation.Augment(xTrain, augmentation);
            yTrain = CalibrationFeatureAugmentation.DuplicateTargets(yTrain, augmentation);
            trainCount = xTrain.Length;

            //Pre-standardize the training set
            var xs = new float[trainCount][];
            var ts = new float[trainCount][];
            for (int i = 0; i < trainCount; i++)
            {
                xs[i] = StandardizeInput(xTrain[i]);
                ts[i] = new float[]
                {
                    (yTrain[i].x - TargetMean[0]) / TargetStd[0],
                    (yTrain[i].y - TargetMean[1]) / TargetStd[1],
                };
            }

            InitializeWeights();

            //Adam state
            var mW = new[] { new float[W1.Length], new float[W2.Length], new float[W3.Length] };
            var vW = new[] { new float[W1.Length], new float[W2.Length], new float[W3.Length] };
            var mB = new[] { new float[B1.Length], new float[B2.Length], new float[B3.Length] };
            var vB = new[] { new float[B1.Length], new float[B2.Length], new float[B3.Length] };
            var weights = new[] { W1, W2, W3 };
            var biases = new[] { B1, B2, B3 };
            int adamStep = 0;

            //Scratch buffers
            var z1 = new float[HIDDEN1]; var a1 = new float[HIDDEN1];
            var z2 = new float[HIDDEN2]; var a2 = new float[HIDDEN2];
            var yHat = new float[OUTPUTS];
            var d3 = new float[OUTPUTS]; var d2 = new float[HIDDEN2]; var d1 = new float[HIDDEN1];
            var gW = new[] { new float[W1.Length], new float[W2.Length], new float[W3.Length] };
            var gB = new[] { new float[B1.Length], new float[B2.Length], new float[B3.Length] };
            var order = new int[trainCount];
            for (int i = 0; i < trainCount; i++) order[i] = i;

            for (int epoch = 0; epoch < EPOCHS; epoch++)
            {
                //Shuffle sample order each epoch
                for (int i = trainCount - 1; i > 0; i--)
                {
                    int j = _rng.Next(i + 1);
                    (order[i], order[j]) = (order[j], order[i]);
                }

                for (int start = 0; start < trainCount; start += BATCH_SIZE)
                {
                    int batch = Math.Min(BATCH_SIZE, trainCount - start);
                    Array.Clear(gW[0], 0, gW[0].Length); Array.Clear(gW[1], 0, gW[1].Length); Array.Clear(gW[2], 0, gW[2].Length);
                    Array.Clear(gB[0], 0, gB[0].Length); Array.Clear(gB[1], 0, gB[1].Length); Array.Clear(gB[2], 0, gB[2].Length);

                    for (int b = 0; b < batch; b++)
                    {
                        var sample = xs[order[start + b]];
                        var target = ts[order[start + b]];

                        Forward(sample, z1, a1, z2, a2, yHat);

                        //Output delta (MSE): d3 = (yHat - t) / batch
                        for (int o = 0; o < OUTPUTS; o++) d3[o] = (yHat[o] - target[o]) / batch;

                        //Backprop layer 3
                        for (int o = 0; o < OUTPUTS; o++)
                        {
                            for (int h = 0; h < HIDDEN2; h++) gW[2][o * HIDDEN2 + h] += d3[o] * a2[h];
                            gB[2][o] += d3[o];
                        }
                        //d2 = (W3^T d3) ⊙ relu'(z2)
                        for (int h = 0; h < HIDDEN2; h++)
                        {
                            float sum = 0;
                            for (int o = 0; o < OUTPUTS; o++) sum += W3[o * HIDDEN2 + h] * d3[o];
                            d2[h] = z2[h] > 0 ? sum : 0;
                        }
                        for (int h = 0; h < HIDDEN2; h++)
                        {
                            for (int k = 0; k < HIDDEN1; k++) gW[1][h * HIDDEN1 + k] += d2[h] * a1[k];
                            gB[1][h] += d2[h];
                        }
                        //d1 = (W2^T d2) ⊙ relu'(z1)
                        for (int k = 0; k < HIDDEN1; k++)
                        {
                            float sum = 0;
                            for (int h = 0; h < HIDDEN2; h++) sum += W2[h * HIDDEN1 + k] * d2[h];
                            d1[k] = z1[k] > 0 ? sum : 0;
                        }
                        for (int k = 0; k < HIDDEN1; k++)
                        {
                            for (int i = 0; i < InputCount; i++) gW[0][k * InputCount + i] += d1[k] * sample[i];
                            gB[0][k] += d1[k];
                        }
                    }

                    //Adam update
                    adamStep++;
                    float corr1 = 1f - (float)Math.Pow(BETA1, adamStep);
                    float corr2 = 1f - (float)Math.Pow(BETA2, adamStep);
                    for (int layer = 0; layer < 3; layer++)
                    {
                        AdamUpdate(weights[layer], gW[layer], mW[layer], vW[layer], corr1, corr2);
                        AdamUpdate(biases[layer], gB[layer], mB[layer], vB[layer], corr1, corr2);
                    }
                }
            }

            //Honest holdout RMSE per axis, in pixels (falls back to the training set when the data is tiny)
            var evalX = testCount > 0 ? xTest : xTrain;
            var evalY = testCount > 0 ? yTest : yTrain;
            double sumSqX = 0, sumSqY = 0;
            for (int i = 0; i < evalX.Length; i++)
            {
                var p = Predict(evalX[i]);
                sumSqX += (p.x - evalY[i].x) * (p.x - evalY[i].x);
                sumSqY += (p.y - evalY[i].y) * (p.y - evalY[i].y);
            }
            var rmseX = (float)Math.Sqrt(sumSqX / evalX.Length);
            var rmseY = (float)Math.Sqrt(sumSqY / evalX.Length);

            var errorXInCm = Functions.PixelsToMm(rmseX) * 0.1f;
            var errorYInCm = Functions.PixelsToMm(rmseY) * 0.1f;
            LastHoldoutRmseXCm = errorXInCm;
            LastHoldoutRmseYCm = errorYInCm;
            return $"MLP Training done. RMSE X: {errorXInCm}cm | RMSE Y: {errorYInCm}cm.";
        }

        private static void AdamUpdate(float[] param, float[] grad, float[] m, float[] v, float corr1, float corr2)
        {
            for (int i = 0; i < param.Length; i++)
            {
                m[i] = BETA1 * m[i] + (1f - BETA1) * grad[i];
                v[i] = BETA2 * v[i] + (1f - BETA2) * grad[i] * grad[i];
                float mHat = m[i] / corr1;
                float vHat = v[i] / corr2;
                param[i] -= LEARNING_RATE * mHat / ((float)Math.Sqrt(vHat) + EPSILON);
            }
        }

        private void Forward(float[] xs, float[] z1, float[] a1, float[] z2, float[] a2, float[] yHat)
        {
            for (int h = 0; h < HIDDEN1; h++)
            {
                float sum = B1[h];
                int row = h * InputCount;
                for (int i = 0; i < InputCount; i++) sum += W1[row + i] * xs[i];
                z1[h] = sum;
                a1[h] = sum > 0 ? sum : 0;
            }
            for (int h = 0; h < HIDDEN2; h++)
            {
                float sum = B2[h];
                int row = h * HIDDEN1;
                for (int k = 0; k < HIDDEN1; k++) sum += W2[row + k] * a1[k];
                z2[h] = sum;
                a2[h] = sum > 0 ? sum : 0;
            }
            for (int o = 0; o < OUTPUTS; o++)
            {
                float sum = B3[o];
                int row = o * HIDDEN2;
                for (int h = 0; h < HIDDEN2; h++) sum += W3[row + h] * a2[h];
                yHat[o] = sum;
            }
        }

        private void InitializeWeights()
        {
            W1 = XavierInit(HIDDEN1, InputCount);
            B1 = new float[HIDDEN1];
            W2 = XavierInit(HIDDEN2, HIDDEN1);
            B2 = new float[HIDDEN2];
            W3 = XavierInit(OUTPUTS, HIDDEN2);
            B3 = new float[OUTPUTS];
        }

        private float[] XavierInit(int fanOut, int fanIn)
        {
            var limit = (float)Math.Sqrt(6.0 / (fanIn + fanOut));
            var w = new float[fanOut * fanIn];
            for (int i = 0; i < w.Length; i++)
                w[i] = ((float)_rng.NextDouble() * 2f - 1f) * limit;
            return w;
        }

        private void ComputeStandardization(float[][] x, Vector2[] y)
        {
            int d = InputCount, n = x.Length;
            FeatureMean = new float[d];
            FeatureStd = new float[d];
            for (int j = 0; j < d; j++)
            {
                double sum = 0;
                for (int i = 0; i < n; i++) sum += x[i][j];
                FeatureMean[j] = (float)(sum / n);
                double sq = 0;
                for (int i = 0; i < n; i++) { var v = x[i][j] - FeatureMean[j]; sq += v * v; }
                var std = (float)Math.Sqrt(sq / n);
                FeatureStd[j] = std < 1e-6f ? 1f : std;
            }

            TargetMean = new float[2];
            TargetStd = new float[2];
            double sx = 0, sy = 0;
            for (int i = 0; i < n; i++) { sx += y[i].x; sy += y[i].y; }
            TargetMean[0] = (float)(sx / n); TargetMean[1] = (float)(sy / n);
            double qx = 0, qy = 0;
            for (int i = 0; i < n; i++)
            {
                qx += (y[i].x - TargetMean[0]) * (y[i].x - TargetMean[0]);
                qy += (y[i].y - TargetMean[1]) * (y[i].y - TargetMean[1]);
            }
            TargetStd[0] = Math.Max(1e-6f, (float)Math.Sqrt(qx / n));
            TargetStd[1] = Math.Max(1e-6f, (float)Math.Sqrt(qy / n));
        }

        private float[] StandardizeInput(float[] x)
        {
            var xs = new float[InputCount];
            for (int i = 0; i < InputCount; i++) xs[i] = (x[i] - FeatureMean[i]) / FeatureStd[i];
            return xs;
        }

        //Reusable Predict scratch buffers so the per-frame ML calibration path allocates nothing.
        //Predict is not re-entrant (called sequentially on the gaze thread), so one set per instance is safe.
        [JsonIgnore] private float[] _predXs, _predZ1, _predA1, _predZ2, _predA2, _predYHat;

        /// <summary>
        /// Predicts the gaze location in pixels. Returns NaN on a feature/model mismatch so callers'
        /// raw-gaze fallback applies (same contract as RidgeRegression.Predict).
        /// </summary>
        public Vector2 Predict(float[] features)
        {
            if (W1 == null || features == null || features.Length != InputCount)
                return new Vector2(float.NaN, float.NaN);

            if (_predXs == null || _predXs.Length != InputCount)
            {
                _predXs = new float[InputCount];
                _predZ1 = new float[HIDDEN1]; _predA1 = new float[HIDDEN1];
                _predZ2 = new float[HIDDEN2]; _predA2 = new float[HIDDEN2];
                _predYHat = new float[OUTPUTS];
            }

            //Standardize into the reused buffer (identical to StandardizeInput, without the allocation)
            for (int i = 0; i < InputCount; i++)
                _predXs[i] = (features[i] - FeatureMean[i]) / FeatureStd[i];

            Forward(_predXs, _predZ1, _predA1, _predZ2, _predA2, _predYHat);

            return new Vector2(
                _predYHat[0] * TargetStd[0] + TargetMean[0],
                _predYHat[1] * TargetStd[1] + TargetMean[1]);
        }

        /// <summary>Saves to StreamingAssets/Calibration Files/MLP/ (same location as the old implementation).</summary>
        public void Save(string fileName)
        {
            string dir = Application.streamingAssetsPath + "/Calibration Files/MLP/";
            if (!Directory.Exists(dir))
                Directory.CreateDirectory(dir);
            File.WriteAllText(dir + fileName, JsonConvert.SerializeObject(this));
        }

        private static bool s_warnedMissing;

        /// <summary>
        /// Loads a trained model, or returns null when no calibration exists yet (callers fall back to the
        /// raw gaze). Old BrightWire-format files are not loadable; run an ML calibration to create a new one.
        /// </summary>
        public static SimpleMLP Load(string fileName)
        {
            string filepath = Application.streamingAssetsPath + $"/Calibration Files/MLP/{fileName}";
            if (!File.Exists(filepath))
            {
                if (!s_warnedMissing)
                {
                    s_warnedMissing = true;
                    Debug.LogWarning("No ML calibration file found, the raw gaze location will be used. Please run an MLCalibration!");
                }
                return null;
            }

            var mlp = JsonConvert.DeserializeObject<SimpleMLP>(File.ReadAllText(filepath));
            if (mlp == null || mlp.W1 == null || mlp.FeatureMean == null)
            {
                if (!s_warnedMissing)
                {
                    s_warnedMissing = true;
                    Debug.LogWarning("ML calibration file is from an older UnitEye version and cannot be loaded. Please run an MLCalibration to retrain it.");
                }
                return null;
            }
            return mlp;
        }
    }
}
