using System;
using System.Collections.Generic;

/// <summary>
/// Shared training routine for the RidgeRegression calibration used by GazeCalibration and HomulerGazeCalibration.
/// The samples are split into a random training and holdout set (Fisher-Yates permutation),
/// the regularization strength lambda is selected with k-fold cross validation on the training set only,
/// the final models are refitted on the full training set and the reported RMSE comes from the untouched holdout set.
/// </summary>
public static class RidgeCalibrationTrainer
{
    public static readonly float[] DefaultLambdas = { 0.01f, 0.05f, 0.1f, 1.0f, 5.0f, 10.0f };

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
        Random rng = null)
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

        //Random split, the first testCount permuted indices form the holdout set
        var permutation = RandomPermutation(sampleCount, rng);

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

        //Select lambda per axis on the training set only
        var bestLambdaX = SelectLambda(xTrain, yXTrain, lambdas, folds);
        var bestLambdaY = SelectLambda(xTrain, yYTrain, lambdas, folds);

        //Refit the final models on the full training set
        var xModel = new RidgeRegression(bestLambdaX);
        xModel.Train(xTrain, yXTrain);
        var yModel = new RidgeRegression(bestLambdaY);
        yModel.Train(xTrain, yYTrain);

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
        };
    }

    /// <summary>
    /// Selects the lambda with the lowest k-fold cross validation MSE.
    /// The samples must already be in random order, folds are contiguous chunks.
    /// </summary>
    private static float SelectLambda(float[][] x, float[] y, float[] lambdas, int folds)
    {
        var sampleCount = x.Length;

        //Too few samples for meaningful folds, use the middle lambda as a safe default
        if (lambdas.Length == 1 || sampleCount < 4)
            return lambdas[lambdas.Length / 2];

        folds = Math.Max(2, Math.Min(folds, sampleCount));

        //Precompute the fold splits once, they are reused for every lambda
        var foldSplits = new List<(float[][] xFit, float[] yFit, float[][] xVal, float[] yVal)>();
        for (int f = 0; f < folds; f++)
        {
            var foldStart = f * sampleCount / folds;
            var foldEnd = (f + 1) * sampleCount / folds;
            var validationCount = foldEnd - foldStart;
            var fitCount = sampleCount - validationCount;

            var xFit = new float[fitCount][];
            var yFit = new float[fitCount];
            var xVal = new float[validationCount][];
            var yVal = new float[validationCount];

            int fitIndex = 0;
            for (int i = 0; i < sampleCount; i++)
            {
                if (i >= foldStart && i < foldEnd)
                {
                    xVal[i - foldStart] = x[i];
                    yVal[i - foldStart] = y[i];
                }
                else
                {
                    xFit[fitIndex] = x[i];
                    yFit[fitIndex] = y[i];
                    fitIndex++;
                }
            }

            foldSplits.Add((xFit, yFit, xVal, yVal));
        }

        var bestLambda = lambdas[0];
        var bestMse = double.MaxValue;

        foreach (var lambda in lambdas)
        {
            double squaredErrorSum = 0.0;
            long errorCount = 0;

            foreach (var (xFit, yFit, xVal, yVal) in foldSplits)
            {
                var model = new RidgeRegression(lambda);
                model.Train(xFit, yFit);

                for (int i = 0; i < xVal.Length; i++)
                {
                    var error = (double)model.Predict(xVal[i]) - yVal[i];
                    squaredErrorSum += error * error;
                    errorCount++;
                }
            }

            var mse = squaredErrorSum / errorCount;
            if (mse < bestMse)
            {
                bestMse = mse;
                bestLambda = lambda;
            }
        }

        return bestLambda;
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
