using System;
using System.Collections.Generic;
namespace UnitEye
{

    /// <summary>
    /// Shared training routine for the RidgeRegression calibration used by GazeCalibration and HomulerGazeCalibration.
    /// The samples are split into a random training and holdout set (Fisher-Yates permutation),
    /// the regularization strength lambda is selected with k-fold cross validation on the training set only,
    /// the final models are refitted on the full training set and the reported RMSE comes from the untouched holdout set.
    /// </summary>
    public static class RidgeCalibrationTrainer
    {
        public static readonly float[] DefaultLambdas = { 0.01f, 0.05f, 0.1f, 1.0f, 5.0f, 10.0f };
        /// <summary>Three cells per axis distinguish corners, edges, and centre without sparse bins.</summary>
        public const int SpatialBalanceCells = 3;
        /// <summary>Caps each cell at 250 samples to bound training time while retaining fixation data.</summary>
        public const int MaxSamplesPerSpatialCell = 250;

        public class Result
        {
            public RidgeRegression XModel;
            public RidgeRegression YModel;
            //Holdout RMSE scaled by the given factors, usually screen size in cm
            public float XRmse;
            public float YRmse;
            public float BestLambdaX;
            public float BestLambdaY;
            public int TrainCount;
            public int TestCount;
            //The untouched holdout split itself (empty when TestCount == 0). Exposed so a post-fit
            //correction (the thin-plate-spline warp) can be VALIDATED on data the ridge never trained on
            //and discarded when it does not generalize.
            public float[][] HoldoutFeatures;
            public float[] HoldoutTargetsX;
            public float[] HoldoutTargetsY;
        }

        /// <summary>
        /// Returns a uniformly random permutation of the indices 0 to count - 1 (Fisher-Yates shuffle).
        /// </summary>
        public static int[] RandomPermutation(int count, Random rng)
        {
            var permutation = new int[count];
            for (int i = 0; i < count; i++)
                permutation[i] = i;

            for (int i = count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (permutation[i], permutation[j]) = (permutation[j], permutation[i]);
            }

            return permutation;
        }

        /// <summary>
        /// Returns a shuffled, interleaved permutation of 3x3 target cells. This makes holdout folds
        /// representative of the screen extremes instead of allowing a chronological sweep to dominate them.
        /// </summary>
        public static int[] StratifiedRandomPermutation(IReadOnlyList<float> targetsX,
            IReadOnlyList<float> targetsY, Random rng)
        {
            if (targetsX == null || targetsY == null || targetsX.Count != targetsY.Count)
                throw new ArgumentException("Target counts do not match.");

            var buckets = CreateSpatialBuckets(targetsX, targetsY);
            foreach (var bucket in buckets)
                Shuffle(bucket, rng);

            var permutation = new List<int>(targetsX.Count);
            for (var offset = 0; permutation.Count < targetsX.Count; offset++)
            {
                foreach (var bucket in buckets)
                {
                    if (offset < bucket.Count)
                        permutation.Add(bucket[offset]);
                }
            }
            return permutation.ToArray();
        }

        /// <summary>
        /// Selects an equal number of samples from each occupied 3x3 screen cell, with replacement for
        /// sparse cells. Dense sweep data therefore cannot drown out deliberate corner fixations.
        /// </summary>
        public static int[] SpatiallyBalancedIndices(IReadOnlyList<float> targetsX,
            IReadOnlyList<float> targetsY, Random rng, int maxSamplesPerCell = MaxSamplesPerSpatialCell)
        {
            if (targetsX == null || targetsY == null || targetsX.Count != targetsY.Count)
                throw new ArgumentException("Target counts do not match.");

            var buckets = CreateSpatialBuckets(targetsX, targetsY);
            var targetCount = 0;
            foreach (var bucket in buckets)
                targetCount = Math.Max(targetCount, bucket.Count);
            targetCount = Clamp(targetCount, 1, maxSamplesPerCell);

            var indices = new List<int>(buckets.Length * targetCount);
            foreach (var bucket in buckets)
            {
                if (bucket.Count == 0)
                    continue;
                Shuffle(bucket, rng);
                for (var i = 0; i < targetCount; i++)
                    indices.Add(bucket[i % bucket.Count]);
            }
            Shuffle(indices, rng);
            return indices.ToArray();
        }

        private static List<int>[] CreateSpatialBuckets(IReadOnlyList<float> targetsX,
            IReadOnlyList<float> targetsY)
        {
            var buckets = new List<int>[SpatialBalanceCells * SpatialBalanceCells];
            for (var i = 0; i < buckets.Length; i++)
                buckets[i] = new List<int>();

            for (var i = 0; i < targetsX.Count; i++)
            {
                var x = ClampToCellIndex(targetsX[i]);
                var y = ClampToCellIndex(targetsY[i]);
                buckets[y * SpatialBalanceCells + x].Add(i);
            }
            return buckets;
        }

        private static int ClampToCellIndex(float coordinate)
            => Clamp((int)(coordinate * SpatialBalanceCells), 0, SpatialBalanceCells - 1);

        private static int Clamp(int value, int minimum, int maximum)
            => Math.Max(minimum, Math.Min(maximum, value));

        private static void Shuffle<T>(IList<T> values, Random rng)
        {
            for (var i = values.Count - 1; i > 0; i--)
            {
                var j = rng.Next(i + 1);
                (values[i], values[j]) = (values[j], values[i]);
            }
        }

        /// <summary>
        /// Trains one RidgeRegression model per axis from the captured calibration samples.
        /// </summary>
        /// <param name="features">Feature vector per sample</param>
        /// <param name="targetsX">Normalized x target per sample</param>
        /// <param name="targetsY">Normalized y target per sample</param>
        /// <param name="rmseScaleX">Factor applied to the x RMSE, usually the screen width in cm</param>
        /// <param name="rmseScaleY">Factor applied to the y RMSE, usually the screen height in cm</param>
        /// <param name="testFraction">Fraction of samples held out for the reported RMSE</param>
        /// <param name="folds">Number of cross validation folds for the lambda selection</param>
        /// <param name="lambdas">Lambda candidates, DefaultLambdas when null</param>
        /// <param name="rng">Random source for the split, time seeded when null</param>
        public static Result Train(
            IReadOnlyList<float[]> features,
            IReadOnlyList<float> targetsX,
            IReadOnlyList<float> targetsY,
            float rmseScaleX,
            float rmseScaleY,
            float testFraction = 0.2f,
            int folds = 5,
            float[] lambdas = null,
            Random rng = null,
            CalibrationFeatureAugmentationSettings augmentation = null,
            bool balanceTraining = false)
        {
            if (features == null || features.Count == 0)
                throw new ArgumentException("No calibration samples were captured, cannot train.");
            if (features.Count != targetsX.Count || features.Count != targetsY.Count)
                throw new ArgumentException("Feature and target counts do not match.");

            lambdas ??= DefaultLambdas;
            rng ??= new Random();

            var sampleCount = features.Count;
            var testCount = (int)Math.Floor(sampleCount * testFraction);
            var trainCount = sampleCount - testCount;
            if (trainCount <= 0)
                throw new ArgumentException("The test fraction leaves no training samples.");

            //Stratified split on unique samples BEFORE any oversampling so holdout never shares duplicates
            //with training folds.
            var permutation = StratifiedRandomPermutation(targetsX, targetsY, rng);

            var xTrain = new float[trainCount][];
            var yXTrain = new float[trainCount];
            var yYTrain = new float[trainCount];
            for (int i = 0; i < trainCount; i++)
            {
                var index = permutation[testCount + i];
                xTrain[i] = features[index];
                yXTrain[i] = targetsX[index];
                yYTrain[i] = targetsY[index];
            }

            var xTest = new float[testCount][];
            var yXTest = new float[testCount];
            var yYTest = new float[testCount];
            for (int i = 0; i < testCount; i++)
            {
                var index = permutation[i];
                xTest[i] = features[index];
                yXTest[i] = targetsX[index];
                yYTest[i] = targetsY[index];
            }

            //Select lambda per axis on the training set only (sharing pre-augmented CV folds across X, Y, and lambdas)
            SelectLambdas(xTrain, yXTrain, yYTrain, lambdas, folds, augmentation, balanceTraining,
                out var bestLambdaX, out var bestLambdaY);

            //Balance strictly within the training partition, then apply training-only feature jitter.
            var xRefit = xTrain;
            var yXRefit = yXTrain;
            var yYRefit = yYTrain;
            if (balanceTraining)
                BalancePartition(ref xRefit, ref yXRefit, ref yYRefit, CalibrationSampleBalancer.SpatialBalancingSeed);

            var augmentedTrain = CalibrationFeatureAugmentation.Augment(xRefit, augmentation);
            var augmentedXTargets = CalibrationFeatureAugmentation.DuplicateTargets(yXRefit, augmentation);
            var augmentedYTargets = CalibrationFeatureAugmentation.DuplicateTargets(yYRefit, augmentation);
            var xModel = new RidgeRegression(bestLambdaX);
            xModel.TrainCore(augmentedTrain, augmentedXTargets);
            var yModel = new RidgeRegression(bestLambdaY);
            yModel.TrainCore(augmentedTrain, augmentedYTargets);

            //Report the error on the untouched holdout set, fall back to the training set
            //when there are too few samples for a holdout
            var evalFeatures = testCount > 0 ? xTest : xTrain;
            var evalTargetsX = testCount > 0 ? yXTest : yXTrain;
            var evalTargetsY = testCount > 0 ? yYTest : yYTrain;

            return new Result
            {
                XModel = xModel,
                YModel = yModel,
                XRmse = Rmse(xModel, evalFeatures, evalTargetsX) * rmseScaleX,
                YRmse = Rmse(yModel, evalFeatures, evalTargetsY) * rmseScaleY,
                BestLambdaX = bestLambdaX,
                BestLambdaY = bestLambdaY,
                TrainCount = trainCount,
                TestCount = testCount,
                HoldoutFeatures = xTest,
                HoldoutTargetsX = yXTest,
                HoldoutTargetsY = yYTest,
            };
        }

        private static void BalancePartition(ref float[][] x, ref float[] yX, ref float[] yY, int seed)
        {
            var selected = SpatiallyBalancedIndices(yX, yY, new Random(seed));
            var bx = new float[selected.Length][];
            var byX = new float[selected.Length];
            var byY = new float[selected.Length];
            for (int i = 0; i < selected.Length; i++)
            {
                int idx = selected[i];
                bx[i] = x[idx];
                byX[i] = yX[idx];
                byY[i] = yY[idx];
            }
            x = bx;
            yX = byX;
            yY = byY;
        }

        /// <summary>
        /// Selects the lambda with the lowest k-fold cross validation MSE per axis.
        /// Fold splits, optional spatial balancing (inside xFit only), and deterministic Box-Muller feature
        /// augmentation are precomputed ONCE per fold and shared across both axes and all lambda candidates.
        /// </summary>
        private static void SelectLambdas(
            float[][] x, float[] yX, float[] yY, float[] lambdas, int folds,
            CalibrationFeatureAugmentationSettings augmentation, bool balanceTraining,
            out float bestLambdaX, out float bestLambdaY)
        {
            var sampleCount = x.Length;

            //Too few samples for meaningful folds, use the middle lambda as a safe default
            if (lambdas.Length == 1 || sampleCount < 4)
            {
                bestLambdaX = bestLambdaY = lambdas[lambdas.Length / 2];
                return;
            }

            folds = Math.Max(2, Math.Min(folds, sampleCount));

            var augmentationRandom = augmentation != null ? new Random(augmentation.seed) : null;
            var foldSplits = new List<(float[][] xFit, float[] yXFit, float[] yYFit, float[][] xVal, float[] yXVal, float[] yYVal)>(folds);
            for (int f = 0; f < folds; f++)
            {
                var foldStart = f * sampleCount / folds;
                var foldEnd = (f + 1) * sampleCount / folds;
                var validationCount = foldEnd - foldStart;
                var fitCount = sampleCount - validationCount;

                var xFit = new float[fitCount][];
                var yXFit = new float[fitCount];
                var yYFit = new float[fitCount];
                var xVal = new float[validationCount][];
                var yXVal = new float[validationCount];
                var yYVal = new float[validationCount];

                int fitIndex = 0;
                for (int i = 0; i < sampleCount; i++)
                {
                    if (i >= foldStart && i < foldEnd)
                    {
                        xVal[i - foldStart] = x[i];
                        yXVal[i - foldStart] = yX[i];
                        yYVal[i - foldStart] = yY[i];
                    }
                    else
                    {
                        xFit[fitIndex] = x[i];
                        yXFit[fitIndex] = yX[i];
                        yYFit[fitIndex] = yY[i];
                        fitIndex++;
                    }
                }

                if (balanceTraining && fitCount > 0)
                    BalancePartition(ref xFit, ref yXFit, ref yYFit, CalibrationSampleBalancer.SpatialBalancingSeed + 1 + f);

                var augXFit = CalibrationFeatureAugmentation.Augment(xFit, augmentation, augmentationRandom);
                var augYXFit = CalibrationFeatureAugmentation.DuplicateTargets(yXFit, augmentation);
                var augYYFit = CalibrationFeatureAugmentation.DuplicateTargets(yYFit, augmentation);
                foldSplits.Add((augXFit, augYXFit, augYYFit, xVal, yXVal, yYVal));
            }

            bestLambdaX = lambdas[0];
            bestLambdaY = lambdas[0];
            var bestMseX = double.MaxValue;
            var bestMseY = double.MaxValue;

            foreach (var lambda in lambdas)
            {
                double sqErrX = 0.0, sqErrY = 0.0;
                long errorCount = 0;

                foreach (var (xFit, yXFit, yYFit, xVal, yXVal, yYVal) in foldSplits)
                {
                    var modelX = new RidgeRegression(lambda);
                    modelX.TrainCore(xFit, yXFit);
                    var modelY = new RidgeRegression(lambda);
                    modelY.TrainCore(xFit, yYFit);

                    for (int i = 0; i < xVal.Length; i++)
                    {
                        var ex = (double)modelX.Predict(xVal[i]) - yXVal[i];
                        var ey = (double)modelY.Predict(xVal[i]) - yYVal[i];
                        sqErrX += ex * ex;
                        sqErrY += ey * ey;
                        errorCount++;
                    }
                }

                var mseX = sqErrX / errorCount;
                if (mseX < bestMseX) { bestMseX = mseX; bestLambdaX = lambda; }
                var mseY = sqErrY / errorCount;
                if (mseY < bestMseY) { bestMseY = mseY; bestLambdaY = lambda; }
            }
        }

        private static float Rmse(RidgeRegression model, float[][] x, float[] y)
        {
            double squaredErrorSum = 0.0;
            for (int i = 0; i < y.Length; i++)
            {
                var error = (double)model.Predict(x[i]) - y[i];
                squaredErrorSum += error * error;
            }

            return (float)Math.Sqrt(squaredErrorSum / y.Length);
        }
    }
}
