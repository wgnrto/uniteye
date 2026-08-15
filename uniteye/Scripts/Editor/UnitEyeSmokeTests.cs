using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using Newtonsoft.Json;
using Unity.InferenceEngine;
using UnitEye;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

/// <summary>
/// Lightweight smoke tests for the pure logic parts of UnitEye.
/// Run from the command line with:
/// Unity.exe -batchmode -projectPath [host project] -executeMethod UnitEyeSmokeTests.Run -logFile [log]
/// In batch mode the editor exits with code 0 when all checks pass and 1 when any check fails.
/// Can also be run from the menu via UnitEye > Run Smoke Tests.
/// </summary>
public static class UnitEyeSmokeTests
{
    private static readonly List<string> _failures = new List<string>();
    private static int _checks;

    [MenuItem("UnitEye/Run Smoke Tests")]
    public static void Run()
    {
        _failures.Clear();
        _checks = 0;

        try
        {
            TestRandomPermutation();
            TestRidgeStandardization();
            TestRidgeSerializationRoundTrip();
            TestRidgeOldFormatCompatibility();
            TestRidgeInterceptNotPenalized();
            TestRobustRidge();
            TestNoShippedDefaultCalibration();
            TestTrainerOnSyntheticData();
            TestSpatiallyBalancedCalibrationSamples();
            TestFeatureAugmentation();
            TestHeadRotationPreset();
            TestCalibrationCaptureHelpers();
            TestSimpleMLP();
            TestGazeGridQuantizer();
            TestOneEuroFilter();
            TestOneEuroFilterVector2FastPath();
            TestEyeCropRect();
            TestIrisFeatures();
            TestScenesAndPrefabsHaveNoMissingScripts();
            TestScenesWireTheMediaPipeGameObject();
            TestFaceLandmarkerBundleSupportsRequestedOutputs();
            TestConsentWordingIsPinned();
            TestConsentRecordAndTokens();
            TestRecordingTierOrderingIsPrivacyMonotonic();
            TestRecorderWritesInvariantNumbers();
            TestRecordingRuntimeHasNoNetworkCode();
            TestScreenGeometryWarning();
            TestBenchmarkRoundTripAndSplit();
            TestEyeMUModelLoadsAndRuns();
            TestGazeEstimationDecode();
            TestGazeFeaturePolynomial();
            TestEyeMUFeaturePolynomial();
            TestGazeModelsLoadAndRun();
            TestCalibrationFileNames();
            TestCalibrationProfiles();
            TestThinPlateSplineWarp();
            TestGazeStatistics();
            TestDriftCorrector();
            TestFixationAggregator();
            TestPursuitCorrelator();
            TestAOIProbability();
            TestGazeErrorModel();
            TestInteriorPreset();
            TestEmbeddingProjection();
        }
        catch (Exception e)
        {
            _failures.Add($"Unhandled exception: {e}");
        }

        if (_failures.Count == 0)
        {
            Debug.Log($"UNITEYE_SMOKE_TESTS_PASSED ({_checks} checks)");
            if (Application.isBatchMode)
                EditorApplication.Exit(0);
        }
        else
        {
            foreach (var failure in _failures)
                Debug.LogError($"UNITEYE_SMOKE_TEST_FAILED: {failure}");
            Debug.LogError($"UNITEYE_SMOKE_TESTS_FAILED ({_failures.Count} failures out of {_checks} checks)");
            if (Application.isBatchMode)
                EditorApplication.Exit(1);
        }
    }

    private static void Check(bool condition, string description)
    {
        _checks++;
        if (!condition)
            _failures.Add(description);
    }

    private static void CheckClose(float actual, float expected, float tolerance, string description)
    {
        Check(Mathf.Abs(actual - expected) <= tolerance,
            $"{description} (expected {expected}, got {actual}, tolerance {tolerance})");
    }

    #region RidgeCalibrationTrainer

    private static void TestRandomPermutation()
    {
        var rng = new System.Random(12345);
        var permutation = RidgeCalibrationTrainer.RandomPermutation(1000, rng);

        Check(permutation.Length == 1000, "Permutation should contain all 1000 entries");

        var seen = new bool[1000];
        var allValid = true;
        foreach (var index in permutation)
        {
            if (index < 0 || index >= 1000 || seen[index]) { allValid = false; break; }
            seen[index] = true;
        }
        Check(allValid, "Permutation should contain every index exactly once");

        var isIdentity = true;
        for (int i = 0; i < permutation.Length; i++)
        {
            if (permutation[i] != i) { isIdentity = false; break; }
        }
        Check(!isIdentity, "Permutation should actually shuffle (the old split always used the chronological head)");
    }

    private static (List<float[]> features, List<float> yX, List<float> yY) MakeSyntheticData(int count, System.Random rng, float noise)
    {
        //Mimics the real feature vector shape: mixed scales plus constant screen dimensions
        var features = new List<float[]>(count);
        var yX = new List<float>(count);
        var yY = new List<float>(count);

        for (int i = 0; i < count; i++)
        {
            var a = (float)rng.NextDouble();            //small scale
            var b = (float)rng.NextDouble() * 1000f;    //large scale
            var c = (float)rng.NextDouble() * 0.01f;    //tiny scale
            var sample = new float[] { a, b, c, 1920f, 1080f };

            features.Add(sample);
            yX.Add(0.4f * a + 0.0003f * b + 20f * c + 0.05f + (float)(rng.NextDouble() - 0.5) * 2f * noise);
            yY.Add(-0.2f * a + 0.0001f * b - 10f * c + 0.30f + (float)(rng.NextDouble() - 0.5) * 2f * noise);
        }

        return (features, yX, yY);
    }

    private static void TestRidgeStandardization()
    {
        var (features, yX, _) = MakeSyntheticData(200, new System.Random(1), noise: 0f);

        var model = new RidgeRegression(0.01f);
        var trainMse = model.Train(ToArray(features), yX.ToArray());

        Check(model.FeatureMean != null && model.FeatureStd != null, "Training should compute standardization stats");
        Check(model.FeatureMean.Count == 5, "Standardization stats should cover all features");
        CheckClose(model.FeatureMean[3], 1920f, 0.001f, "Mean of a constant feature should be the constant");
        CheckClose(model.FeatureStd[3], 1f, 0.001f, "Std of a constant feature should fall back to one");
        Check(trainMse < 1e-3f, $"Ridge should fit noiseless linear data closely, train MSE was {trainMse}");

        //Prediction on a fresh sample
        var fresh = new float[] { 0.5f, 500f, 0.005f, 1920f, 1080f };
        var expected = 0.4f * 0.5f + 0.0003f * 500f + 20f * 0.005f + 0.05f;
        CheckClose(model.Predict(fresh), expected, 0.02f, "Ridge prediction on a fresh sample");
    }

    private static void TestRidgeSerializationRoundTrip()
    {
        var (features, yX, _) = MakeSyntheticData(100, new System.Random(2), noise: 0.001f);

        var model = new RidgeRegression(0.05f);
        model.Train(ToArray(features), yX.ToArray());

        //Same serialization calls as RidgeRegression.Save/Load, without touching the file system
        var json = JsonConvert.SerializeObject(model);
        var loaded = JsonConvert.DeserializeObject<RidgeRegression>(json);

        Check(loaded.FeatureMean != null, "Standardization stats should survive the serialization round trip");

        var sample = features[0];
        CheckClose(loaded.Predict(sample), model.Predict(sample), 1e-4f, "Prediction should be identical after save and load");
    }

    private static void TestRidgeOldFormatCompatibility()
    {
        //A file in the pre standardization format, W = bias + two feature weights
        const string oldJson = "{\"W\":[0.1,0.2,0.3],\"B\":0.1,\"Lambda\":0.01,\"Affine\":true}";
        var model = JsonConvert.DeserializeObject<RidgeRegression>(oldJson);

        Check(model.FeatureMean == null && model.FeatureStd == null, "Old files should load without standardization stats");
        //0.1 * 1 + 0.2 * 1 + 0.3 * 2 = 0.9
        CheckClose(model.Predict(new float[] { 1f, 2f }), 0.9f, 1e-5f, "Old format prediction should be plain affine weights");
    }

    private static void TestRidgeInterceptNotPenalized()
    {
        //Ridge must NOT penalize the intercept: with standardized (zero-mean) features the intercept
        //carries the whole target mean, so even a LARGE lambda must reproduce a constant target exactly.
        //The old Train added lambda over the full identity (including the bias column), shrinking the
        //intercept to mean(y) * N / (N + lambda) — for N=60, lambda=10 that is ~14% low, i.e. a
        //systematic offset of every prediction toward screen coordinate 0.
        const int n = 60;
        var rng = new System.Random(7);
        var x = new float[n][];
        var y = new float[n];
        for (int i = 0; i < n; i++)
        {
            x[i] = new float[] { (float)rng.NextDouble(), (float)rng.NextDouble() * 100f, (float)rng.NextDouble() };
            y[i] = 0.7f;
        }

        var model = new RidgeRegression(10f);
        model.Train(x, y);
        CheckClose(model.Predict(x[0]), 0.7f, 1e-3f,
            "Large-lambda ridge must still recover a constant target exactly (unpenalized intercept)");
    }

    private static void TestRobustRidge()
    {
        //Robust (IRLS/Huber) fitting: outlier calibration samples (a blink or saccade caught mid-dwell)
        //must not drag the fit. Build clean linear data, fit it, then corrupt 10% of the TARGETS with large
        //errors and refit — the robust fit should stay close to both the truth and the clean-data fit,
        //whereas a plain least-squares fit would be pulled ~0.5 off by +5 outliers on a ~[0,1] target.
        var rng = new System.Random(123);
        const int n = 200;
        var x = new float[n][];
        var yClean = new float[n];
        for (int i = 0; i < n; i++)
        {
            float a = (float)rng.NextDouble();
            float b = (float)rng.NextDouble();
            x[i] = new float[] { a, b };
            yClean[i] = 0.3f + 0.5f * a - 0.2f * b;
        }

        var clean = new RidgeRegression(0.01f);
        clean.Train(x, (float[])yClean.Clone());

        var yCorrupt = (float[])yClean.Clone();
        for (int i = 0; i < n / 10; i++)
            yCorrupt[rng.Next(n)] += 5f; // gross target outliers

        var robust = new RidgeRegression(0.01f);
        robust.Train(x, yCorrupt);

        var probe = new float[] { 0.7f, 0.3f };
        float truth = 0.3f + 0.5f * 0.7f - 0.2f * 0.3f;
        float pRobust = robust.Predict(probe);
        Check(Mathf.Abs(pRobust - truth) < 0.15f,
            $"Robust ridge should resist target outliers (pred {pRobust:F3} vs truth {truth:F3})");
        Check(Mathf.Abs(pRobust - clean.Predict(probe)) < 0.15f,
            "Robust ridge on corrupted data should stay near the clean-data fit");

        var robust2 = new RidgeRegression(0.01f);
        robust2.Train(x, yCorrupt);
        Check(Mathf.Abs(robust2.Predict(probe) - pRobust) < 1e-4f, "Robust ridge training should be deterministic");
    }

    private static void TestNoShippedDefaultCalibration()
    {
        //We deliberately no longer ship a default ridge/MLP fit. The old one-person defaults ignored
        //the eye-gaze signal and extrapolated off-screen for anyone else (Reg_Y regularized to a
        //near-constant top edge, Reg_X driven by that person's head geometry), so the crosshair
        //corner-locked and looked broken before the first calibration. A fresh user must instead get
        //raw (uncalibrated) gaze via HomulerGaze's null-model fallback.
        var defaults = Resources.Load<CalibrationResource>("CalibrationDefaultFiles");
        Check(defaults != null, "CalibrationDefaultFiles asset should load from Resources");
        if (defaults != null)
        {
            Check(defaults.regXAsset == null, "No shipped default Reg_X (raw gaze until the user calibrates)");
            Check(defaults.regYAsset == null, "No shipped default Reg_Y (raw gaze until the user calibrates)");
            Check(defaults.mlpAsset == null, "No shipped default MLP (raw gaze until the user calibrates)");
        }

        //The degenerate default JSON files must be gone from Resources
        var oldX = AssetDatabase.LoadAssetAtPath<TextAsset>(
            "Packages/de.uniulm.uniteye/Resources/Calibration/Default/Reg_X.json");
        var oldY = AssetDatabase.LoadAssetAtPath<TextAsset>(
            "Packages/de.uniulm.uniteye/Resources/Calibration/Default/Reg_Y.json");
        Check(oldX == null, "Old degenerate default Reg_X.json should be deleted");
        Check(oldY == null, "Old degenerate default Reg_Y.json should be deleted");

        //Fallback contract: a model with no weights predicts NaN, which HomulerGaze.RefineGazeLocation
        //treats as 'use raw gaze'. (This is what a null default now resolves to.)
        var empty = JsonConvert.DeserializeObject<RidgeRegression>("{\"W\":null,\"B\":0,\"Lambda\":1,\"Affine\":true}");
        Check(empty != null, "RidgeRegression should still deserialize");
        var features = new float[12];
        Check(float.IsNaN(empty.Predict(features)), "A model with no weights must predict NaN (the raw-gaze fallback signal)");
    }

    private static void TestTrainerOnSyntheticData()
    {
        var (features, yX, yY) = MakeSyntheticData(500, new System.Random(3), noise: 0.005f);

        var result = RidgeCalibrationTrainer.Train(
            features, yX, yY,
            rmseScaleX: 10f, rmseScaleY: 10f,
            rng: new System.Random(7));

        Check(result.TrainCount == 400, $"Expected 400 training samples, got {result.TrainCount}");
        Check(result.TestCount == 100, $"Expected 100 holdout samples, got {result.TestCount}");
        Check(result.XRmse < 0.3f, $"Holdout RMSE X should be near the noise level, was {result.XRmse}");
        Check(result.YRmse < 0.3f, $"Holdout RMSE Y should be near the noise level, was {result.YRmse}");
        Check(Array.IndexOf(RidgeCalibrationTrainer.DefaultLambdas, result.BestLambdaX) >= 0, "Selected lambda X should come from the candidate list");
        Check(Array.IndexOf(RidgeCalibrationTrainer.DefaultLambdas, result.BestLambdaY) >= 0, "Selected lambda Y should come from the candidate list");

        //Fresh sample prediction through both refitted models
        var fresh = new float[] { 0.25f, 250f, 0.0025f, 1920f, 1080f };
        var expectedX = 0.4f * 0.25f + 0.0003f * 250f + 20f * 0.0025f + 0.05f;
        var expectedY = -0.2f * 0.25f + 0.0001f * 250f - 10f * 0.0025f + 0.30f;
        CheckClose(result.XModel.Predict(fresh), expectedX, 0.02f, "Trainer X model prediction");
        CheckClose(result.YModel.Predict(fresh), expectedY, 0.02f, "Trainer Y model prediction");

        //Same seed must give the same split and therefore the same result
        var repeat = RidgeCalibrationTrainer.Train(
            features, yX, yY,
            rmseScaleX: 10f, rmseScaleY: 10f,
            rng: new System.Random(7));
        Check(repeat.XRmse == result.XRmse && repeat.YRmse == result.YRmse, "Training with the same seed should be deterministic");
    }

    private static void TestSpatiallyBalancedCalibrationSamples()
    {
        var x = new List<float>();
        var y = new List<float>();
        //A dense centre sweep and sparse corner fixation must contribute equally after balancing.
        for (var i = 0; i < 100; i++) { x.Add(0.5f); y.Add(0.5f); }
        x.Add(0.08f); y.Add(0.08f);
        x.Add(0.08f); y.Add(0.08f);

        var indices = RidgeCalibrationTrainer.SpatiallyBalancedIndices(
            x, y, new System.Random(9), maxSamplesPerCell: 10);
        Check(indices.Length == 20, "Spatial balancing should retain ten samples from each occupied cell");

        var cornerCount = 0;
        foreach (var index in indices)
            if (x[index] < 1f / 3f && y[index] < 1f / 3f) cornerCount++;
        Check(cornerCount == 10, "Spatial balancing should upweight sparse corner fixation samples");

        var permutation = RidgeCalibrationTrainer.StratifiedRandomPermutation(x, y, new System.Random(9));
        Check(permutation.Length == x.Count, "Stratified permutation should retain every sample");
        var firstIsCorner = x[permutation[0]] < 1f / 3f && y[permutation[0]] < 1f / 3f;
        var secondIsCorner = x[permutation[1]] < 1f / 3f && y[permutation[1]] < 1f / 3f;
        Check(firstIsCorner != secondIsCorner, "Stratified permutation should interleave target cells");
    }

    private static void TestFeatureAugmentation()
    {
        var features = new[]
        {
            new[] { 1f, 10f, 5f },
            new[] { 3f, 30f, 5f },
            new[] { 5f, 50f, 5f },
        };
        var targets = new[] { 0.1f, 0.5f, 0.9f };

        //Feature augmentation is now the DEFAULT calibration approach: a freshly constructed settings object
        //must be enabled so calibration augments unless the user deliberately turns it off.
        Check(new CalibrationFeatureAugmentationSettings().enabled,
            "Feature augmentation should be enabled by default");
        Check(CalibrationFeatureAugmentation.IsEnabled(new CalibrationFeatureAugmentationSettings()),
            "Default feature augmentation settings should report as enabled");

        var disabled = new CalibrationFeatureAugmentationSettings { enabled = false };
        Check(ReferenceEquals(CalibrationFeatureAugmentation.Augment(features, disabled), features),
            "Disabled augmentation should leave training features unchanged");

        var settings = new CalibrationFeatureAugmentationSettings
        {
            enabled = true,
            copiesPerSample = 2,
            standardDeviationScale = 0.1f,
            maximumStandardDeviations = 2f,
            seed = 17,
        };
        var augmented = CalibrationFeatureAugmentation.Augment(features, settings);
        var repeated = CalibrationFeatureAugmentation.Augment(features, settings);
        var augmentedTargets = CalibrationFeatureAugmentation.DuplicateTargets(targets, settings);
        Check(augmented.Length == features.Length * 3, "Augmentation should add the requested training copies");
        Check(augmentedTargets.Length == augmented.Length, "Augmented features and labels should remain aligned");
        for (var copy = 0; copy <= settings.copiesPerSample; copy++)
            for (var i = 0; i < features.Length; i++)
            {
                var index = copy * features.Length + i;
                Check(augmentedTargets[index] == targets[i], "Feature augmentation must not change calibration labels");
                for (var feature = 0; feature < features[i].Length; feature++)
                    Check(augmented[index][feature] == repeated[index][feature],
                        "Fixed augmentation seed should produce deterministic feature jitter");
            }
        Check(augmented[features.Length][2] == features[0][2],
            "Zero-variance features should not receive synthetic jitter");

        //Extra head-pose jitter: a head-pose slot with zero captured variance still receives synthetic
        //jitter when named (so the fit is not tied to the single calibration head pose), while an unnamed
        //zero-variance feature stays fixed and the original (copy 0) samples are never jittered.
        var headSettings = new CalibrationFeatureAugmentationSettings
        {
            enabled = true,
            copiesPerSample = 1,
            standardDeviationScale = 0.1f,
            maximumStandardDeviations = 2f,
            headPoseJitterDegrees = 3f,
            seed = 17,
            headPoseFeatureIndices = new[] { 1 }, // feature 1 is the "head-pose" slot; feature 2 is not
        };
        var headFeatures = new[]
        {
            new[] { 1f, 5f, 7f },
            new[] { 3f, 5f, 7f },
            new[] { 5f, 5f, 7f },
        };
        var headAug = CalibrationFeatureAugmentation.Augment(headFeatures, headSettings);
        var headRepeat = CalibrationFeatureAugmentation.Augment(headFeatures, headSettings);
        Check(headAug.Length == headFeatures.Length * 2, "Head-pose augmentation should still add the requested copies");
        var anyHeadJitter = false;
        for (var i = 0; i < headFeatures.Length; i++)
        {
            Check(headAug[i][1] == headFeatures[i][1], "The original (copy 0) samples must never be jittered");
            var copyIndex = headFeatures.Length + i;
            if (Mathf.Abs(headAug[copyIndex][1] - headFeatures[i][1]) > 1e-6f)
                anyHeadJitter = true;
            //Bound = proportional (0 for a zero-variance feature) + (jitterDegrees in radians) * maxStd.
            //Head-pose features are radians, so the 3-degree jitter is converted before being applied.
            Check(Mathf.Abs(headAug[copyIndex][1] - headFeatures[i][1]) <= 3f * Mathf.Deg2Rad * 2f + 1e-4f,
                "Head-pose jitter must stay within its bound");
            Check(headAug[copyIndex][2] == headFeatures[i][2],
                "An unnamed zero-variance feature must not receive head-pose jitter");
            Check(headAug[copyIndex][1] == headRepeat[copyIndex][1],
                "Head-pose jitter should be deterministic for a fixed seed");
        }
        Check(anyHeadJitter, "A named zero-variance head-pose feature should receive synthetic jitter");

        var (ridgeFeatures, yX, yY) = MakeSyntheticData(500, new System.Random(31), noise: 0.005f);
        var result = RidgeCalibrationTrainer.Train(ridgeFeatures, yX, yY, 10f, 10f,
            rng: new System.Random(7), augmentation: settings);
        Check(result.TrainCount == 400 && result.TestCount == 100,
            "Ridge augmentation must not duplicate the reported train or holdout counts");
        Check(result.XRmse < 0.4f && result.YRmse < 0.4f,
            "Ridge feature jitter should retain clean synthetic-data accuracy");
    }

    private static void TestHeadRotationPreset()
    {
        //Ordinary presets are not a head-movement stage; the new preset is, and dwells at its waypoints.
        Check(!new CornerPreset(20f).IsHeadMovement, "Ordinary presets must not be flagged as head-movement");

        var preset = new HeadRotationPreset(20f, 5f, 0.08f);
        Check(preset.IsHeadMovement, "HeadRotationPreset should be a head-movement stage");
        Check(preset.StopAtWaypoints, "HeadRotationPreset should dwell at its waypoints");
        Check(Mathf.Approximately(preset.DwellSeconds, 5f), "HeadRotationPreset should honor its dwell seconds");
        //The dwell floor keeps a too-short request usable for a full head roll.
        Check(Mathf.Approximately(new HeadRotationPreset(20f, 0.5f).DwellSeconds, 2f),
            "HeadRotationPreset should clamp the dwell to a sensible floor");

        //centre approach + centre dwell + four corners + closing centre. The calibration dwells at
        //points[1..n-2], i.e. the centre and the four corners, which is the head-pose coverage this stage
        //is for. The head-pose feature indices the calibration jitters must match the runner layouts.
        var points = preset.GetPoints();
        Check(points.Count == 7, "HeadRotationPreset should produce centre + four corners with approach/close points");
    }

    private static void TestCalibrationCaptureHelpers()
    {
        //Pursuit-lag lookup: the label for a sweep sample is the newest dot position at least lag seconds old.
        var trail = new List<(float time, Vector2 pos)>
        {
            (1.00f, new Vector2(100, 0)),
            (1.05f, new Vector2(110, 0)),
            (1.10f, new Vector2(120, 0)),
            (1.15f, new Vector2(130, 0)),
        };
        var lagged = HomulerGazeCalibration.DelayedDotPosition(trail, 1.15f, 0.1f, new Vector2(-1, -1));
        Check(lagged == new Vector2(110, 0), "Pursuit-lag label is the newest dot position at least lag seconds old");
        var fallback = HomulerGazeCalibration.DelayedDotPosition(trail, 1.05f, 0.5f, new Vector2(-1, -1));
        Check(fallback == new Vector2(-1, -1), "Pursuit-lag lookup falls back when the trail has no entry that old");

        //Fixation gate: tight cluster passes, spread window and too-few samples fail.
        var tight = new List<(float time, Vector2 pos)>
            { (0f, new Vector2(500, 500)), (0.1f, new Vector2(505, 498)), (0.2f, new Vector2(498, 503)) };
        Check(HomulerGazeCalibration.IsFixationStable(tight, 20f), "A tight gaze cluster counts as a fixation");
        var spread = new List<(float time, Vector2 pos)>
            { (0f, new Vector2(500, 500)), (0.1f, new Vector2(700, 500)), (0.2f, new Vector2(500, 700)) };
        Check(!HomulerGazeCalibration.IsFixationStable(spread, 20f), "A spread-out window is not a fixation");
        var few = new List<(float time, Vector2 pos)> { (0f, new Vector2(500, 500)), (0.1f, new Vector2(500, 500)) };
        Check(!HomulerGazeCalibration.IsFixationStable(few, 20f), "Fewer than 3 samples never count as a fixation");
    }

    private static float[][] ToArray(List<float[]> list) => list.ToArray();

    private static void TestSimpleMLP()
    {
        //Nonlinear synthetic data (something ridge cannot fit) with pixel-scale targets,
        //mimicking the calibration setup
        var rng = new System.Random(99);
        int n = 800;
        var x = new float[n][];
        var y = new Vector2[n];
        for (int i = 0; i < n; i++)
        {
            float a = (float)rng.NextDouble() * 2f - 1f;
            float b = (float)rng.NextDouble() * 2f - 1f;
            x[i] = new float[] { a, b, a * b, 1920f };  //includes a constant feature
            y[i] = new Vector2(
                960f + 400f * a + 250f * (float)Math.Sin(2.5 * b),
                540f + 300f * b + 200f * a * b);
        }

        var mlp = new SimpleMLP(seed: 42);
        Check(mlp.LastHoldoutRmseXCm < 0f, "SimpleMLP holdout RMSE should be the -1 sentinel before training");
        var message = mlp.Train(x, y);
        Check(message.Contains("MLP Training done"), "SimpleMLP.Train should return the accuracy message");

        //Train must PUBLISH the holdout RMSE it already computes, not only embed it in the message:
        //HomulerGazeCalibration.ProcessDataNeural feeds these into the static LastHoldoutRmseCm that drives
        //the calibration validation gate. While they were private, an MLCalibration run left that static
        //describing the last RidgeRegression fit (or stuck at -1 on a fresh install, silencing the gate).
        Check(mlp.LastHoldoutRmseXCm >= 0f && mlp.LastHoldoutRmseYCm >= 0f,
            $"SimpleMLP.Train should publish its holdout RMSE (got {mlp.LastHoldoutRmseXCm},{mlp.LastHoldoutRmseYCm})");
        Check(message.Contains($"RMSE X: {mlp.LastHoldoutRmseXCm}") && message.Contains($"RMSE Y: {mlp.LastHoldoutRmseYCm}"),
            "SimpleMLP's published holdout RMSE should be the same number it reports in the message");
        //It describes a training run, not the model, so it must stay out of the persisted JSON.
        Check(!JsonConvert.SerializeObject(mlp).Contains("LastHoldoutRmse"),
            "SimpleMLP holdout RMSE should not be serialized into MLP.json");

        //Holdout accuracy: target std is ~470px/380px, an MLP that learned should be far below that
        double sx = 0, sy = 0;
        var probeRng = new System.Random(7);
        int probes = 200;
        for (int i = 0; i < probes; i++)
        {
            float a = (float)probeRng.NextDouble() * 2f - 1f;
            float b = (float)probeRng.NextDouble() * 2f - 1f;
            var truth = new Vector2(960f + 400f * a + 250f * (float)Math.Sin(2.5 * b), 540f + 300f * b + 200f * a * b);
            var p = mlp.Predict(new float[] { a, b, a * b, 1920f });
            sx += (p.x - truth.x) * (p.x - truth.x);
            sy += (p.y - truth.y) * (p.y - truth.y);
        }
        var rmseX = (float)Math.Sqrt(sx / probes);
        var rmseY = (float)Math.Sqrt(sy / probes);
        Check(rmseX < 100f && rmseY < 100f, $"SimpleMLP should fit nonlinear data (probe RMSE {rmseX:F1},{rmseY:F1}px)");

        //Round-trip via the same JSON path Save/Load use
        var json = JsonConvert.SerializeObject(mlp);
        var loaded = JsonConvert.DeserializeObject<SimpleMLP>(json);
        var probe = new float[] { 0.3f, -0.4f, -0.12f, 1920f };
        var p1 = mlp.Predict(probe); var p2 = loaded.Predict(probe);
        Check(Mathf.Abs(p1.x - p2.x) < 1e-3f && Mathf.Abs(p1.y - p2.y) < 1e-3f, "SimpleMLP prediction should survive the serialization round trip");

        //Determinism with a seed
        var mlp2 = new SimpleMLP(seed: 42);
        mlp2.Train(x, y);
        var q1 = mlp.Predict(probe); var q2 = mlp2.Predict(probe);
        Check(q1 == q2, "SimpleMLP training should be deterministic for a fixed seed");

        var augmentation = new CalibrationFeatureAugmentationSettings
        {
            enabled = true,
            copiesPerSample = 1,
            standardDeviationScale = 0.01f,
            maximumStandardDeviations = 2f,
            seed = 42,
        };
        var augmentedMlp = new SimpleMLP(seed: 42);
        augmentedMlp.Train(x, y, augmentation);
        var augmentedPrediction = augmentedMlp.Predict(probe);
        Check(Mathf.Abs(augmentedPrediction.x - p1.x) < 100f && Mathf.Abs(augmentedPrediction.y - p1.y) < 100f,
            "MLP feature jitter should retain clean synthetic-data accuracy");

        //Feature/model mismatch returns NaN (raw-gaze fallback contract)
        var mismatch = mlp.Predict(new float[] { 1f, 2f });
        Check(float.IsNaN(mismatch.x), "SimpleMLP.Predict should return NaN on a dimensionality mismatch");
    }

    #endregion

    #region GazeGridQuantizer

    private static void TestGazeGridQuantizer()
    {
        var quantizer = new GazeGridQuantizer(columns: 3, rows: 3, hysteresisMargin: 0.15f, dwellSeconds: 0.1f);

        Check(quantizer.CurrentCell == -1, "Quantizer should start without an active cell");

        //First sample is adopted immediately
        Check(quantizer.Update(new Vector2(0.10f, 0.10f), 0.00f), "First sample should activate a cell");
        Check(quantizer.CurrentCell == 0 && quantizer.CurrentColumn == 0 && quantizer.CurrentRow == 0, "First sample should map to cell 0");

        //Jitter across the border but inside the hysteresis margin must not switch
        Check(!quantizer.Update(new Vector2(0.34f, 0.10f), 0.02f), "Jitter inside the hysteresis margin should not switch");
        Check(quantizer.CurrentCell == 0, "Cell should still be 0 after margin jitter");

        //A clear move switches only after the dwell time
        Check(!quantizer.Update(new Vector2(0.50f, 0.10f), 0.04f), "A new cell should not be adopted instantly");
        Check(!quantizer.Update(new Vector2(0.50f, 0.10f), 0.09f), "A new cell should not be adopted before the dwell time");
        Check(quantizer.Update(new Vector2(0.50f, 0.10f), 0.15f), "A new cell should be adopted after the dwell time");
        Check(quantizer.CurrentCell == 1, "Cell should be 1 after the dwell switch");

        //Returning to the active cell resets the dwell candidate
        Check(!quantizer.Update(new Vector2(0.90f, 0.10f), 2.00f), "Excursion sample should only start a candidate");
        Check(!quantizer.Update(new Vector2(0.50f, 0.10f), 2.05f), "Returning to the active cell should not switch");
        Check(!quantizer.Update(new Vector2(0.90f, 0.10f), 2.10f), "Second excursion should restart the candidate");
        Check(!quantizer.Update(new Vector2(0.90f, 0.10f), 2.15f), "The restarted candidate should not use the old dwell start");
        Check(quantizer.Update(new Vector2(0.90f, 0.10f), 2.21f), "The restarted candidate should switch after a full dwell");
        Check(quantizer.CurrentCell == 2, "Cell should be 2 after the second dwell switch");

        //Out of range samples are clamped onto the grid
        quantizer.Update(new Vector2(1.5f, 1.5f), 5.0f);
        Check(quantizer.Update(new Vector2(1.5f, 1.5f), 5.2f), "Clamped out of range samples should switch after dwell");
        Check(quantizer.CurrentCell == 8, "Out of range samples should clamp to the last cell");

        //NaN samples are ignored
        Check(!quantizer.Update(new Vector2(float.NaN, 0.5f), 6.0f), "NaN samples should be ignored");
        Check(quantizer.CurrentCell == 8, "NaN samples should not change the cell");

        //Zero dwell switches immediately once outside the margin
        var immediate = new GazeGridQuantizer(columns: 2, rows: 2, hysteresisMargin: 0.1f, dwellSeconds: 0f);
        immediate.Update(new Vector2(0.2f, 0.2f), 0f);
        Check(immediate.Update(new Vector2(0.9f, 0.9f), 1f), "Zero dwell should switch immediately");
        Check(immediate.CurrentCell == 3, "Zero dwell switch should land in the sampled cell");

        //Geometry helpers
        Check(quantizer.CellAt(new Vector2(0.99f, 0.99f)) == 8, "CellAt should map the bottom right corner to the last cell");
        var rect = quantizer.GetCellRect(4);
        CheckClose(rect.x, 1f / 3f, 1e-5f, "Cell rect x of the center cell");
        CheckClose(rect.y, 1f / 3f, 1e-5f, "Cell rect y of the center cell");
    }

    #endregion

    #region Scenes and prefabs

    private static void TestScenesAndPrefabsHaveNoMissingScripts()
    {
        //All scenes/prefabs are now expected to have zero missing scripts. (The HomulerGazeCalibration
        //scene previously had five dangling landmark-annotation references; the MediaPipe 0.16.3 Task-API
        //migration's cleanup pass (MigrationCleanup) stripped them along with the deleted Solution-era
        //components.)
        var knownMissingScripts = new Dictionary<string, int>();

        //Validates that all scenes and prefabs shipped with the package still resolve their
        //script and asset references, which guards against GUID breakage from restructuring
        var sceneGuids = AssetDatabase.FindAssets("t:SceneAsset", new[] { "Packages/de.uniulm.uniteye/Scenes" });
        Check(sceneGuids.Length > 0, "Package scenes should be found");

        foreach (var guid in sceneGuids)
        {
            var path = AssetDatabase.GUIDToAssetPath(guid);
            var scene = EditorSceneManager.OpenScene(path, OpenSceneMode.Single);

            var missingScripts = 0;
            foreach (var root in scene.GetRootGameObjects())
                missingScripts += CountMissingScriptsRecursive(root);

            var expected = knownMissingScripts.TryGetValue(path, out var known) ? known : 0;
            Check(missingScripts == expected, $"Scene {path} has {missingScripts} missing scripts (expected {expected})");
        }

        var prefabGuids = AssetDatabase.FindAssets("t:Prefab", new[] { "Packages/de.uniulm.uniteye/Prefabs" });
        Check(prefabGuids.Length > 0, "Package prefabs should be found");

        foreach (var guid in prefabGuids)
        {
            var path = AssetDatabase.GUIDToAssetPath(guid);
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
            Check(prefab != null, $"Prefab {path} should be loadable");
            if (prefab == null) continue;

            var missingScripts = CountMissingScriptsRecursive(prefab);
            Check(missingScripts == 0, $"Prefab {path} has {missingScripts} missing scripts");
        }

        //Leave a fresh empty scene behind so no package scene stays open
        EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
    }

    private static void TestScenesWireTheMediaPipeGameObject()
    {
        //"No missing scripts" is not the same as "still works": the MediaPipe 0.16.3 Task-API migration
        //stripped the dead Solution-era components from the Mediapipe GameObject, and the calibration scene
        //never got the replacements (FaceMeshSolution + WebCamSource) re-added. That scene then threw an NRE
        //out of the NativeGazeProvider constructor and one per frame from LateUpdate afterwards, while the
        //missing-script test above stayed green. So assert the wiring every HomulerGaze actually needs.
        var sceneGuids = AssetDatabase.FindAssets("t:SceneAsset", new[] { "Packages/de.uniulm.uniteye/Scenes" });
        Check(sceneGuids.Length > 0, "Package scenes should be found");

        var checkedComponents = 0;
        foreach (var guid in sceneGuids)
        {
            var path = AssetDatabase.GUIDToAssetPath(guid);
            var scene = EditorSceneManager.OpenScene(path, OpenSceneMode.Single);

            foreach (var root in scene.GetRootGameObjects())
            {
                foreach (var gaze in root.GetComponentsInChildren<UnitEye.HomulerGaze>(true))
                {
                    checkedComponents++;
                    //_mediaPipeGO is private; SerializedObject reads it without widening the runtime API.
                    var go = new SerializedObject(gaze).FindProperty("_mediaPipeGO")
                        .objectReferenceValue as GameObject;
                    Check(go != null, $"{path}: HomulerGaze on '{gaze.name}' has _mediaPipeGO assigned");
                    if (go == null) continue;

                    Check(go.GetComponent<Mediapipe.Unity.FaceMesh.FaceMeshSolution>() != null,
                        $"{path}: '{go.name}' (HomulerGaze._mediaPipeGO) has a FaceMeshSolution");
                    Check(go.GetComponent<Mediapipe.Unity.WebCamSource>() != null,
                        $"{path}: '{go.name}' (HomulerGaze._mediaPipeGO) has a WebCamSource");
                }
            }
        }
        Check(checkedComponents > 0, "At least one package scene should contain a HomulerGaze");

        EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
    }

    private static void TestConsentWordingIsPinned()
    {
        //A consent record stores the SHA-256 of the wording it was collected under. That is only worth
        //anything if the wording cannot drift silently, so the hash is pinned here: editing any consent
        //string fails this test until someone updates the pin DELIBERATELY (and bumps WordingVersion).
        //If this fails and you did change the text on purpose: bump GazeConsentTexts.WordingVersion, then
        //paste the "actual" hash below. Do not paste it without bumping the version — old records would
        //then claim a version whose text no longer exists.
        const string pinnedVersion = "1.0.0";
        const string pinnedHash = "6f3b1c7dbced07b406eb8a68e19a15d6cd971e025434f52cf3df34bb767a9a29";
        Check(GazeConsentTexts.WordingVersion == pinnedVersion,
            $"Consent wording version should be {pinnedVersion} (got {GazeConsentTexts.WordingVersion})");
        var actual = GazeConsentTexts.WordingHash();
        Check(actual.Length == 64, "Consent wording hash should be a 64-char SHA-256 hex string");
        if (pinnedHash != "PIN_ME")
            Check(actual == pinnedHash, $"Consent wording changed without a version bump (hash {actual})");
        else
            Debug.Log($"UNITEYE_CONSENT_WORDING_HASH: {actual}");

        //The text must not promise anonymity: a 478-point face mesh is a biometric template and the imagery
        //tiers are plainly identifying. Saying otherwise would make the consent false, not just imprecise.
        foreach (var text in new[] { GazeConsentTexts.Intro, GazeConsentTexts.TierChoice, GazeConsentTexts.Publication })
            Check(text.IndexOf("anonym", StringComparison.OrdinalIgnoreCase) < 0,
                "Consent wording must not claim anonymity");

        //It must state plainly that publication cannot be fully undone; without that it is not informed.
        Check(GazeConsentTexts.Publication.Contains("permanent"),
            "Publication consent must warn that publication is effectively permanent");
    }

    private static void TestConsentRecordAndTokens()
    {
        var utc = new DateTime(2026, 3, 1, 12, 0, 0, DateTimeKind.Utc);
        var rec = GazeConsentRecord.Create(GazeRecordingTier.EyeCrops, mayPublish: true, utc, "a@b.c", "study");

        Check(rec.consentedOnUtcDate == "2026-03-01", "Consent stores the calendar date it was given");
        //Date only, never a time of day: the consent text says so, and time-of-day is a routine signal.
        Check(!rec.consentedOnUtcDate.Contains(":"), "Consent must not store a time of day");
        Check(rec.publicationHoldUntilUtcDate == "2026-03-15",
            $"Publication hold should be {GazeConsentRecord.PublicationHoldDays} days out (got {rec.publicationHoldUntilUtcDate})");

        //The hold is the only thing standing between "agreed" and "published", so the predicate must be strict.
        Check(!rec.PublishableOn(utc), "Not publishable on the day consent was given");
        Check(!rec.PublishableOn(utc.AddDays(13)), "Not publishable before the hold elapses");
        Check(rec.PublishableOn(utc.AddDays(14)), "Publishable once the hold has elapsed");
        var noPublish = GazeConsentRecord.Create(GazeRecordingTier.Features, mayPublish: false, utc, "a@b.c", "s");
        Check(!noPublish.PublishableOn(utc.AddDays(365)),
            "A session that never consented to publication is never publishable, however old");

        //Tokens are the only participant identifier and the only handle for withdrawal: they must be unique
        //and transcribable. Ambiguous glyphs (I/L/O/U) are excluded so a code read off a screen onto paper
        //cannot become a different valid code.
        var seen = new HashSet<string>();
        string ambiguous = null;
        for (int i = 0; i < 2000; i++)
        {
            var t = GazeConsentRecord.NewParticipantToken();
            seen.Add(t);
            if (ambiguous == null && t.IndexOfAny(new[] { 'I', 'L', 'O', 'U' }) >= 0) ambiguous = t;
        }
        Check(seen.Count == 2000, $"Participant tokens should not collide (got {seen.Count} distinct of 2000)");
        Check(ambiguous == null, $"Participant tokens must avoid ambiguous characters (got {ambiguous})");
        Check(GazeConsentRecord.NewParticipantToken().Length == 14, "Participant token should be grouped 4-4-4");
    }

    private static void TestRecorderWritesInvariantNumbers()
    {
        //The dataset is comma-separated JSON. Under a comma-decimal culture the ambient formatter renders
        //0.4193f as "0,4193", which silently turns one number into two fields — unreadable on any other
        //machine. This repo already formats CSV floats with the ambient culture elsewhere, so the recorder
        //pinning InvariantCulture is worth a test rather than a comment.
        var previous = System.Threading.Thread.CurrentThread.CurrentCulture;
        try
        {
            System.Threading.Thread.CurrentThread.CurrentCulture = new CultureInfo("de-DE");
            //Sanity-check the hazard is real on this runtime before asserting the fix.
            Check((0.4193f).ToString() == "0,4193", "de-DE should format a float with a decimal comma (hazard check)");

            var consent = GazeConsentRecord.Create(GazeRecordingTier.Features, false,
                new DateTime(2026, 3, 1, 0, 0, 0, DateTimeKind.Utc), "a@b.c", "smoke");
            var recorder = new GazeSessionRecorder(consent, GazeRecordingTier.Features, 1);
            var folder = recorder.SessionFolder;
            try
            {
                //0.4193 goes in as a SCALAR (iris disagreement) so it lands in the JSON row; the feature
                //vector is a binary blob and would not exercise the formatter at all.
                recorder.RecordSample(0, new[] { 0.4193f, -1.5f }, new Vector2(12.5f, 7.25f), new Vector2(12.5f, 7.25f),
                    1.5, true, false, 0, 0, 512.5f, new Vector3(0.1f, 0.2f, 0.3f), 0.4193f, false, null, null);
                recorder.Finish("test", 1.25f);

                var rows = File.ReadAllText(Path.Combine(folder, "samples.jsonl"));
                Check(rows.Contains("0.4193"), $"Recorded floats must use a decimal point under de-DE (row: {rows.Trim()})");
                Check(!rows.Contains("0,4193"), "Recorded floats must never use a decimal comma");
                //A decimal comma would also split one JSON number into two fields; count the commas that
                //separate real fields to catch that even if the value above ever changes.
                Check(!rows.Contains(",\"i\":") || rows.IndexOf("{\"i\":", StringComparison.Ordinal) == 0,
                    "Recorded rows should be one JSON object per line");
                Check(File.Exists(Path.Combine(folder, "consent.json")),
                    "consent.json must be written beside the data it governs");

                //Feature blob is raw float32 and must round-trip bit-exactly — it is the model's actual input.
                var blob = File.ReadAllBytes(Path.Combine(folder, "features.f32"));
                Check(blob.Length == 8, $"Feature blob should hold 2 float32s (got {blob.Length} bytes)");
                Check(Mathf.Abs(BitConverter.ToSingle(blob, 0) - 0.4193f) < 1e-9f, "Feature blob should round-trip exactly");
            }
            finally
            {
                try { Directory.Delete(folder, true); } catch { }
            }
        }
        finally { System.Threading.Thread.CurrentThread.CurrentCulture = previous; }
    }

    private static void TestRecordingRuntimeHasNoNetworkCode()
    {
        //The consent screen tells participants "we never send anything over the internet". That promise is
        //only as good as the code, so assert it over the source rather than trusting review. Scoped to the
        //recording/consent files: the wider runtime legitimately reaches the vendored MediaPipe, which uses
        //UnityWebRequest to load model files from StreamingAssets.
        var dir = Path.GetFullPath("Packages/de.uniulm.uniteye/Scripts/Runtime/Recording");
        Check(Directory.Exists(dir), $"Recording source folder should exist at {dir}");
        if (!Directory.Exists(dir)) return;

        string[] forbidden = { "UnityWebRequest", "System.Net", "HttpClient", "WebClient", "Socket", "UploadHandler" };
        var files = Directory.GetFiles(dir, "*.cs", SearchOption.AllDirectories);
        Check(files.Length > 0, "Recording folder should contain source files");
        foreach (var file in files)
        {
            var text = File.ReadAllText(file);
            foreach (var needle in forbidden)
                Check(text.IndexOf(needle, StringComparison.Ordinal) < 0,
                    $"{Path.GetFileName(file)} must contain no networking code (found '{needle}')");
        }
    }

    private static void TestRecordingTierOrderingIsPrivacyMonotonic()
    {
        //Downstream code gates imagery with `tier >= EyeCrops` and face geometry with `tier >= Landmarks`.
        //That only stays correct while the enum is ordered least- to most-identifying, so pin the order.
        Check((int)GazeRecordingTier.Off == 0, "Off must be 0 so a default-constructed tier records nothing");
        Check(GazeRecordingTier.Features < GazeRecordingTier.Landmarks, "Features is less identifying than Landmarks");
        Check(GazeRecordingTier.Landmarks < GazeRecordingTier.EyeCrops, "Landmarks is less identifying than EyeCrops");
        Check(GazeRecordingTier.EyeCrops < GazeRecordingTier.FaceVideo, "EyeCrops is less identifying than FaceVideo");
        Check(GazeRecordingTier.FaceVideo < GazeRecordingTier.FullFrames, "FaceVideo is less identifying than FullFrames");
    }

    private static void TestScreenGeometryWarning()
    {
        //Runs in batch mode, so Application.isEditor is true and the render surface is not a real display -
        //exactly the condition the warning exists for. Asserting it FIRES here is the meaningful direction:
        //a warning that silently never appears is worse than none, because it reads as an all-clear.
        var warning = ScreenGeometry.PhysicalScaleWarning();
        Check(warning.Length > 0, "A non-fullscreen / Editor render surface produces a physical-scale warning");
        Check(warning.Contains("centim", StringComparison.OrdinalIgnoreCase) || warning.Contains("CENTIM"),
            "The warning says which figures are affected, not merely that something is wrong");
        Check(ScreenGeometry.DisplayWidth > 0 && ScreenGeometry.DisplayHeight > 0,
            "Display resolution is readable");
    }

    private static void TestBenchmarkRoundTripAndSplit()
    {
        //Writes a synthetic session, reads it back and benchmarks it. Covers the recorder/reader seam (two
        //pieces of hand-rolled JSON written at different times - exactly where silent corruption lives) and
        //the two properties that make the benchmark's number mean anything.
        var consent = GazeConsentRecord.Create(GazeRecordingTier.Features, false,
            new DateTime(2026, 3, 1, 0, 0, 0, DateTimeKind.Utc), "a@b.c", "bench-smoke");
        var recorder = new GazeSessionRecorder(consent, GazeRecordingTier.Features, 1);
        var folder = recorder.SessionFolder;
        try
        {
            const int W = 1920, H = 1080;
            recorder.WriteSessionHeader("EyeMU", W, H, 52f, 29f, 640, 480, false, true, true, 478, true, false, "not_asked");

            //13 dwell locations on a 3x3 + interior layout, 40 samples each. Features are a linear function
            //of the target plus tiny noise, so a ridge fit should recover it and the reported error should be
            //small but non-zero - enough to tell "it ran" from "it silently returned garbage".
            var rng = new System.Random(7);
            var locs = new List<Vector2>();
            foreach (var fy in new[] { 0.1f, 0.5f, 0.9f })
                foreach (var fx in new[] { 0.1f, 0.5f, 0.9f })
                    locs.Add(new Vector2(fx * W, fy * H));
            foreach (var fy in new[] { 0.25f, 0.75f })
                foreach (var fx in new[] { 0.25f, 0.75f })
                    locs.Add(new Vector2(fx * W, fy * H));

            int i = 0;
            foreach (var loc in locs)
                for (var k = 0; k < 40; k++)
                {
                    float nx = loc.x / W, ny = loc.y / H;
                    var f = new[]
                    {
                        nx + (float)(rng.NextDouble() - 0.5) * 0.004f,
                        ny + (float)(rng.NextDouble() - 0.5) * 0.004f,
                        1f,
                    };
                    recorder.RecordSample(i++, f, loc, loc, i * 0.03, atDwell: true, headRotation: false,
                        preset: 0, round: 0, distanceMm: 600f, headPose: new Vector3(0.01f, 0.02f, 0f),
                        irisDisagreement: 0.01f, blinking: false, source: null, provider: null);
                }
            recorder.Finish("completed", 0.9f);

            //--- reader inverts the writer ---
            var session = UnitEye.Benchmark.GazeDatasetReader.Load(folder);
            Check(session.Usable, $"A recorded session reads back as usable (status {session.Status})");
            Check(session.Samples.Count == locs.Count * 40,
                $"Reader recovers every row (got {session.Samples.Count}, expected {locs.Count * 40})");
            Check(session.ScreenWidthPx == W && session.ScreenHeightPx == H, "Reader recovers screen geometry");
            Check(session.Samples[0].Features.Length == 3, "Reader recovers the feature vector length");
            Check(Mathf.Abs(session.Samples[0].LabelPx.x - locs[0].x) < 0.01f, "Reader recovers the label");
            Check(!float.IsNaN(session.Samples[0].DistanceMm), "Reader recovers an optional field that was present");

            //--- the benchmark runs and produces a sane number ---
            var result = UnitEye.Benchmark.GazeBenchmark.BenchmarkSession(session,
                new UnitEye.Benchmark.GazeBenchmark.Config { Name = "t", Augmentation = false });
            Check(result.Status == "ok", $"Benchmark completes on a synthetic session (status {result.Status})");
            Check(result.Locations == locs.Count, $"Benchmark finds every dwell location (got {result.Locations})");
            Check(!double.IsNaN(result.RmsePctDiag), "Benchmark reports an error figure");
            Check(result.RmsePctDiag > 0.0 && result.RmsePctDiag < 25.0,
                $"Benchmark error is in a sane range (got {result.RmsePctDiag:F3}% of diagonal)");

            //Degrees are SUPPRESSED here, and that is the point. This session is written from the Editor, so
            //its centimetre figures come from Screen.dpi describing a monitor while Screen.width describes a
            //Game view - the ratio is unknown and unrecoverable. % of diagonal survives (same units top and
            //bottom); degrees would be confidently wrong, which is worse than absent.
            Check(!session.PhysicalScaleTrustworthy,
                "A session recorded in the Editor is flagged as having untrustworthy physical scale");
            Check(double.IsNaN(result.RmseDegrees),
                "Degrees are withheld when the session's centimetre scale is not trustworthy");
            Check(!double.IsNaN(result.RmsePctDiag),
                "% of diagonal is still reported when the physical scale is untrustworthy - it is a ratio");

            //--- the split actually withholds ---
            //Every fold trains WITHOUT the location it scores, so the error cannot be zero however clean the
            //data is. A zero here would mean the held-out rows leaked into training - the exact failure the
            //whole design exists to prevent, and one that looks like a great result.
            Check(result.RmsePctDiag > 1e-6,
                "A leave-one-target-out benchmark cannot report zero error - that would mean the split leaked");

            //--- the MLP head is scored in the same space as the ridge head ---
            //The two heads natively disagree on units: the ridge pair predicts normalized, SimpleMLP predicts
            //pixels. If the benchmark ever scored one in the wrong space the error would be off by a factor of
            //~W, so assert the MLP lands in the same plausible band rather than merely "runs".
            var mlp = UnitEye.Benchmark.GazeBenchmark.BenchmarkSession(session,
                new UnitEye.Benchmark.GazeBenchmark.Config
                {
                    Name = "t-mlp",
                    Model = UnitEye.Benchmark.GazeBenchmark.Head.Mlp,
                    Augmentation = false,
                });
            Check(mlp.Status == "ok", $"Benchmark completes with the MLP head (status {mlp.Status})");
            Check(!double.IsNaN(mlp.RmsePctDiag), "MLP head reports an error figure");
            Check(double.IsNaN(mlp.RmseDegrees), "The MLP head withholds degrees on the same grounds as ridge");
            Check(mlp.RmsePctDiag > 1e-6 && mlp.RmsePctDiag < 25.0,
                $"MLP error is in the same sane band as ridge, i.e. scored in pixels not normalized units " +
                $"(got {mlp.RmsePctDiag:F3}% of diagonal vs ridge {result.RmsePctDiag:F3}%)");

            //Determinism: a seeded re-run must reproduce the number, or an A/B is measuring RNG.
            var mlpAgain = UnitEye.Benchmark.GazeBenchmark.BenchmarkSession(session,
                new UnitEye.Benchmark.GazeBenchmark.Config
                {
                    Name = "t-mlp",
                    Model = UnitEye.Benchmark.GazeBenchmark.Head.Mlp,
                    Augmentation = false,
                });
            Check(Math.Abs(mlp.RmsePctDiag - mlpAgain.RmsePctDiag) < 1e-9,
                "A seeded MLP benchmark reproduces its number exactly across runs");
        }
        finally
        {
            try { Directory.Delete(folder, true); } catch { }
        }
    }

    private static void TestFaceLandmarkerBundleSupportsRequestedOutputs()
    {
        //FaceMeshSolution asks the FaceLandmarker for blendshapes and transformation matrixes. Both come
        //from extra models packed INSIDE the .task bundle, so pointing at a bundle that lacks one fails
        //task creation at runtime ("BLENDSHAPES Tag and blendshapes model must be both set") and takes the
        //whole native gaze path down — a mismatch no other test here can see. Build the landmarker with
        //exactly the options the runtime uses and assert it comes up.
        var modelName = Mediapipe.Unity.FaceMesh.FaceMeshSolution.ModelFileName;
        var packaged = System.IO.Path.GetFullPath(
            $"Packages/com.github.homuler.mediapipe/PackageResources/MediaPipe/{modelName}");
        Check(System.IO.File.Exists(packaged), $"The MediaPipe package ships {modelName}");

        var installed = System.IO.Path.Combine(Application.streamingAssetsPath, modelName);
        Check(System.IO.File.Exists(installed),
            $"{modelName} is installed in StreamingAssets (run UnitEye > Install MediaPipe StreamingAssets)");
        if (!System.IO.File.Exists(installed)) return;

        Mediapipe.Tasks.Vision.FaceLandmarker.FaceLandmarker landmarker = null;
        try
        {
            var options = new Mediapipe.Tasks.Vision.FaceLandmarker.FaceLandmarkerOptions(
                new Mediapipe.Tasks.Core.BaseOptions(
                    Mediapipe.Tasks.Core.BaseOptions.Delegate.CPU,
                    modelAssetBuffer: System.IO.File.ReadAllBytes(installed)),
                runningMode: Mediapipe.Tasks.Vision.Core.RunningMode.VIDEO,
                numFaces: 1,
                minFaceDetectionConfidence: 0.5f,
                minFacePresenceConfidence: 0.5f,
                minTrackingConfidence: 0.5f,
                outputFaceBlendshapes: true,
                outputFaceTransformationMatrixes: true);
            landmarker = Mediapipe.Tasks.Vision.FaceLandmarker.FaceLandmarker.CreateFromOptions(
                options, Mediapipe.Unity.GpuManager.GpuResources);
            Check(landmarker != null, $"{modelName} builds a FaceLandmarker with blendshapes + transformation matrixes");
        }
        catch (Exception e)
        {
            Check(false, $"{modelName} must support the outputs FaceMeshSolution requests: {e.Message}");
        }
        finally
        {
            landmarker?.Close();
        }
    }

    private static void TestIrisFeatures()
    {
        //The iris blocks must be paired with the eye each one actually sits in. MediaPipe names the two
        //5-point blocks (468.., 473..) by IMAGE side while the corner constants use SUBJECT side, so the
        //two namings are mirrored — pairing by name pairs each iris with the OTHER eye and yields
        //offsets of roughly +/-2.4 corner-distances instead of a fraction of one. Pin the real geometry:
        //index 468 lies BETWEEN corners 33 and 133, index 473 between corners 362 and 263 (measured on a
        //real detection). Laying the synthetic face out that way is what makes this test able to catch a
        //re-swap; a layout built from the naming alone passes either way.
        var landmarks = new List<Mediapipe.NormalizedLandmark>(478);
        for (int i = 0; i < 478; i++)
            landmarks.Add(new Mediapipe.NormalizedLandmark { X = 0f, Y = 0f });
        //Subject's LEFT eye: corners 362/263, on the image RIGHT (x 0.60..0.70); its iris is index 473.
        landmarks[362] = new Mediapipe.NormalizedLandmark { X = 0.60f, Y = 0.50f };
        landmarks[263] = new Mediapipe.NormalizedLandmark { X = 0.70f, Y = 0.50f };
        landmarks[473] = new Mediapipe.NormalizedLandmark { X = 0.675f, Y = 0.52f };
        //Subject's RIGHT eye: corners 33/133, on the image LEFT (x 0.30..0.40); its iris is index 468.
        landmarks[33] = new Mediapipe.NormalizedLandmark { X = 0.30f, Y = 0.50f };
        landmarks[133] = new Mediapipe.NormalizedLandmark { X = 0.40f, Y = 0.50f };
        landmarks[468] = new Mediapipe.NormalizedLandmark { X = 0.35f, Y = 0.50f };

        Check(HomulerFunctions.RightIrisCenter == 468 && HomulerFunctions.LeftIrisCenter == 473,
            "Iris centers map to the eye they lie in (468 -> corners 33/133, 473 -> corners 362/263)");

        var f = new float[4];
        HomulerFunctions.FillIrisFeatures(landmarks, f, 0);
        CheckClose(f[0], 0.25f, 1e-5f, "Left iris offset x = (iris - corner mid) / corner distance");
        CheckClose(f[1], 0.20f, 1e-5f, "Left iris offset y uses the same normalization");
        CheckClose(f[2], 0f, 1e-5f, "A centered right iris gives zero x offset");
        CheckClose(f[3], 0f, 1e-5f, "A centered right iris gives zero y offset");

        //Missing landmarks and a degenerate eye must yield zeros, never NaN/Infinity.
        var g = new float[4] { 9f, 9f, 9f, 9f };
        HomulerFunctions.FillIrisFeatures(null, g, 0);
        Check(g[0] == 0f && g[1] == 0f && g[2] == 0f && g[3] == 0f, "Missing landmarks give zero iris features");
        landmarks[HomulerFunctions.LeftEyeOuterCorner] = new Mediapipe.NormalizedLandmark { X = 0.60f, Y = 0.50f };
        HomulerFunctions.FillIrisFeatures(landmarks, g, 0);
        Check(!float.IsNaN(g[0]) && !float.IsInfinity(g[0]) && g[0] == 0f && g[1] == 0f,
            "A degenerate (zero-width) eye gives zeros, not NaN");
    }

    private static void TestEyeCropRect()
    {
        //Corners in MediaPipe convention (normalized, y-down): a 0.1-wide eye slightly above the
        //vertical image center, 1280x720 source
        var landmarks = new List<Mediapipe.NormalizedLandmark>
        {
            new Mediapipe.NormalizedLandmark { X = 0.45f, Y = 0.42f },
            new Mediapipe.NormalizedLandmark { X = 0.55f, Y = 0.42f },
        };

        var rect = HomulerFunctions.GetEyeCropRect(landmarks, 0, 1, 1280, 720);
        var rectAgain = HomulerFunctions.GetEyeCropRect(landmarks, 0, 1, 1280, 720);

        //GetEyeCropRect must not modify the landmarks (the old in-place Y flip corrupted the
        //EyeCorners model input and toggled the crop between eye and cheek on alternating frames)
        CheckClose(landmarks[0].Y, 0.42f, 1e-6f, "GetEyeCropRect must not mutate landmark Y (left)");
        CheckClose(landmarks[1].Y, 0.42f, 1e-6f, "GetEyeCropRect must not mutate landmark Y (right)");
        Check(rect.Equals(rectAgain), "GetEyeCropRect must be deterministic across repeated calls");

        //Pinned expected geometry: padded eyeLength 0.14 -> 179px square at (550, 316) bottom-left
        Check(rect.width == 179 && rect.height == 179, $"Eye crop should be a 179px square, got {rect.width}x{rect.height}");
        Check(rect.x == 550 && rect.y == 316, $"Eye crop origin should be (550, 316), got ({rect.x}, {rect.y})");

        //The eye center (640, 417.6 in bottom-left pixels) must fall inside the crop, in its upper half
        Check(rect.x <= 640 && 640 <= rect.x + rect.width, "Eye center X must be inside the crop");
        Check(rect.y <= 417 && 418 <= rect.y + rect.height, "Eye center Y must be inside the crop");
        Check(417.6f - rect.y > rect.height * 0.5f, "Eye center must sit above the crop's vertical midpoint");
    }

    private static int CountMissingScriptsRecursive(GameObject gameObject)
    {
        var missing = GameObjectUtility.GetMonoBehavioursWithMissingScriptCount(gameObject);
        foreach (Transform child in gameObject.transform)
            missing += CountMissingScriptsRecursive(child.gameObject);
        return missing;
    }

    #endregion

    #region EyeMU model (Inference Engine)

    private static void TestEyeMUModelLoadsAndRuns()
    {
        //Verifies the Barracuda->Inference Engine migration end-to-end at the model level:
        //the EyeMU .onnx imports as a ModelAsset, the reference is bound (rebinder ran),
        //the model loads, and it executes with the input/output names the runner relies on.
        //NOTE: this proves the graph loads and runs and (via Schedule's shape validation) that the
        //tensor layout is accepted — it does NOT prove the gaze output is numerically correct.
        var resource = Resources.Load<EyeMUResource>("EyeMU");
        Check(resource != null, "EyeMU resource should load from Resources");
        if (resource == null) return;

        Check(resource.modelAsset != null, "EyeMU modelAsset should be bound after the rebinder ran");
        if (resource.modelAsset == null) return;

        var model = ModelLoader.Load(resource.modelAsset);
        Check(model != null, "EyeMU model should load under Inference Engine");
        if (model == null) return;

        //Execute once with blank NHWC inputs (1x128x128x3) using the model's actual I/O names.
        //Pure-CPU tensors (no TextureConverter) so this runs under -nographics. This proves the model
        //loads and runs and that the names/shapes the runner uses match the model (Schedule validates
        //shapes). It does NOT prove the gaze output is numerically correct - that needs a live camera.
        Worker worker = null;
        Tensor<float> t1 = null, t2 = null, t4 = null, t5 = null;
        try
        {
            worker = new Worker(model, BackendType.CPU);

            t1 = new Tensor<float>(new TensorShape(1, 128, 128, 3), new float[128 * 128 * 3]);
            t2 = new Tensor<float>(new TensorShape(1, 128, 128, 3), new float[128 * 128 * 3]);
            t4 = new Tensor<float>(new TensorShape(1, 8), new float[8]);
            t5 = new Tensor<float>(new TensorShape(1, 4), new float[4]);

            worker.SetInput("input_1:0", t1);
            worker.SetInput("input_2:0", t2);
            worker.SetInput("input_4", t4);
            worker.SetInput("input_5", t5);
            worker.Schedule();

            var outT = worker.PeekOutput("dense_8") as Tensor<float>;
            Check(outT != null, "Output 'dense_8' should exist");
            float syncX = float.NaN, syncY = float.NaN;
            if (outT != null)
            {
                var data = outT.DownloadToArray();
                Check(data.Length >= 2, $"dense_8 should produce at least 2 values (got {data.Length})");
                if (data.Length >= 2)
                {
                    syncX = data[0];
                    syncY = data[1];
                    Check(!float.IsNaN(data[0]) && !float.IsInfinity(data[0]) && !float.IsNaN(data[1]) && !float.IsInfinity(data[1]),
                        $"dense_8 output should be finite (got {data[0]}, {data[1]})");
                }
            }

            //Async-readback API path (what the runners use with _asyncGpuReadback on): schedule again,
            //request a readback, poll for completion, then a non-blocking download. On the CPU backend the
            //readback completes promptly; this verifies the API sequence compiles/works and that the
            //pipelined result matches the synchronous one for identical inputs.
            worker.Schedule();
            var asyncOut = worker.PeekOutput("dense_8") as Tensor<float>;
            Check(asyncOut != null, "Output 'dense_8' should exist for the async-readback pass");
            if (asyncOut != null)
            {
                asyncOut.ReadbackRequest();
                var done = false;
                for (var spin = 0; spin < 100000 && !done; spin++)
                    done = asyncOut.IsReadbackRequestDone();
                Check(done, "ReadbackRequest should complete (CPU backend)");
                if (done)
                {
                    var data = asyncOut.DownloadToArray();
                    Check(data.Length >= 2 && data[0] == syncX && data[1] == syncY,
                        "Async readback returns the same result as the synchronous download for identical inputs");
                }
            }
        }
        finally
        {
            t1?.Dispose();
            t2?.Dispose();
            t4?.Dispose();
            t5?.Dispose();
            worker?.Dispose();
        }
    }

    //A confident head: peaked in LOGIT space, so the softmax floor is negligible and the expectation
    //lands on the peak.
    private static float[] GazeBinGaussian(int centre, float sigma)
    {
        var bins = new float[90];
        for (int i = 0; i < bins.Length; i++)
            bins[i] = -((i - centre) * (i - centre)) / (2f * sigma * sigma);
        return bins;
    }

    //A broad unimodal lobe, the shape the MobileGaze heads actually emit — unlike a one-hot spike.

    private static float[] GazeBinLobe(int centre, float falloff)

    {

        var bins = new float[90];

        for (int i = 0; i < bins.Length; i++)

            bins[i] = 4f * Mathf.Exp(-Mathf.Abs(i - centre) * falloff);

        return bins;

    }


    //Two lobes, the second scaled by secondHeight. The shape that makes a windowed decode bistable.

    private static float[] GazeBinTwoLobes(int a, int b, float secondHeight)

    {

        var bins = new float[90];

        for (int i = 0; i < bins.Length; i++)

            bins[i] = 2f * Mathf.Exp(-Mathf.Abs(i - a) * 0.35f)

                    + secondHeight * Mathf.Exp(-Mathf.Abs(i - b) * 0.35f);

        return bins;

    }


    private static float[] GazeBinSpike(int index)
    {
        var b = new float[90];
        b[index] = 100f;   // softmax -> ~one-hot at index
        return b;
    }

    private static void TestCalibrationFileNames()
    {
        //Per-backbone calibration files: each backbone gets a distinct name so a calibration for one
        //model never overwrites another's (their feature vectors differ). Save and Load share this helper.
        Check(CalibrationModelStore.FileName("Reg_X.json", GazeBackbone.EyeMU) == "Reg_X_EyeMU.json",
            "Ridge X calibration file name for EyeMU");
        Check(CalibrationModelStore.FileName("MLP.json", GazeBackbone.GazeMobileOne) == "MLP_GazeMobileOne.json",
            "MLP calibration file name for MobileOne");
        var a = CalibrationModelStore.FileName("Reg_Y.json", GazeBackbone.EyeMU);
        var b = CalibrationModelStore.FileName("Reg_Y.json", GazeBackbone.GazeMobileOne);
        var c = CalibrationModelStore.FileName("Reg_Y.json", GazeBackbone.GazeMobileNetV2);
        Check(a != b && b != c && a != c, "Calibration file names are distinct per backbone");
    }

    private static void TestGazeEstimationDecode()
    {
        //Decode: softmax + expectation over all 90 bins, index*4deg - 180deg -> radians. Pure math.
        //A one-hot spike is NOT what a trained head emits, and against a flat floor the full expectation is
        //legitimately pulled toward the mean of that floor, so the spike cases are checked on the windowed
        //decode and the default is checked on a smooth lobe — which is the shape the network actually
        //produces.
        CheckClose(GazeEstimationRunner.DecodeAngleRadiansWindowed(GazeBinSpike(45)), 0f, 0.02f, "Gaze decode: center bin ~ 0 rad");
        CheckClose(GazeEstimationRunner.DecodeAngleRadiansWindowed(GazeBinSpike(0)), -Mathf.PI, 0.02f, "Gaze decode: bin 0 ~ -pi rad");
        CheckClose(GazeEstimationRunner.DecodeAngleRadiansWindowed(GazeBinSpike(89)), (89f * 4f - 180f) * Mathf.Deg2Rad, 0.02f, "Gaze decode: bin 89");

        //Default decode on a CONFIDENT head (peaked in logit space): lands exactly on the peak.
        CheckClose(GazeEstimationRunner.DecodeAngleRadians(GazeBinGaussian(60, 3f)), (60f * 4f - 180f) * Mathf.Deg2Rad, 0.02f,
            "Gaze decode: full expectation lands on a confident peak");

        //Default decode on a BROAD head — which is what these exports actually emit. The contract is
        //MONOTONE, not exact: the full expectation compresses (bin 65 reads +43.7° rather than +80°)
        //because a broad floor drags it toward the mean. That is a scale error, and absorbing scale errors
        //is what the calibration is for. Asserting exactness here would be asserting something false, and
        //asserting nothing would let a decode that has stopped tracking pass.
        float prev = float.NegativeInfinity;
        foreach (int centre in new[] { 35, 45, 55, 65 })
        {
            float deg = GazeEstimationRunner.DecodeAngleRadians(GazeBinLobe(centre, 0.35f)) * Mathf.Rad2Deg;
            Check(deg > prev + 1f, $"Gaze decode: broad head still tracks monotonically (bin {centre} -> {deg:F1} deg)");
            prev = deg;
        }

        //Windowed decode, on a spike over a flat floor, resists the far-bin drag. Real, and the reason the
        //window was introduced — it just does not describe these heads.
        var noisy = new float[90];
        noisy[60] = 5f;
        CheckClose(GazeEstimationRunner.DecodeAngleRadiansWindowed(noisy), (60f * 4f - 180f) * Mathf.Deg2Rad, 0.05f,
            "Gaze decode: windowed expectation resists far-bin noise (no centre drag)");

        //WHY THE WINDOW IS NOT THE DEFAULT. Two lobes of near-equal height, far apart — the shape the
        //MobileGaze heads actually produce, where the softmax peak is only ~6x uniform. Nudging the second
        //lobe past the first by an amount far too small to be an eye movement swings the windowed decode
        //the whole distance between the lobes; the full expectation barely moves.
        float aWin = GazeEstimationRunner.DecodeAngleRadiansWindowed(GazeBinTwoLobes(40, 85, 1.99f));
        float aFull = GazeEstimationRunner.DecodeAngleRadians(GazeBinTwoLobes(40, 85, 1.99f));
        float bWin = GazeEstimationRunner.DecodeAngleRadiansWindowed(GazeBinTwoLobes(40, 85, 2.01f));
        float bFull = GazeEstimationRunner.DecodeAngleRadians(GazeBinTwoLobes(40, 85, 2.01f));
        Check(Mathf.Abs(aWin - bWin) * Mathf.Rad2Deg > 100f,
            "Gaze decode: windowed decode jumps between lobes on a nudge (why it is not the default)");
        Check(Mathf.Abs(aFull - bFull) * Mathf.Rad2Deg < 10f,
            "Gaze decode: full expectation is stable across the same nudge");
    }

    private static void TestGazeFeaturePolynomial()
    {
        //The direction backbone must emit the polynomial-expanded calibration feature vector so a per-axis
        //LINEAR ridge can bend to the corners (raw [yaw,pitch] can't: the angle->screen map is nonlinear
        //with a yaw*pitch coupling). Pin the length + exact term layout so the basis isn't silently
        //changed and train/predict stay in lockstep (both read this same vector).
        Check(GazeEstimationRunner.FeatureCount == 32, "Gaze calibration feature vector is 32 terms (polynomial + head pose + iris + context)");
        Check(GazeEstimationRunner.IrisFeatureStart == 11, "Direction-model iris block starts after the head pose (7/8/9 stay stable)");
        Check(GazeEstimationRunner.ContextFeatureStart == 15, "Direction-model context block starts after the iris block");
        var f = new float[GazeEstimationRunner.FeatureCount];
        float yaw = 0.3f, pitch = -0.2f;
        GazeEstimationRunner.FillGazeFeatures(f, yaw, pitch, 0.11f, 0.12f, 0.13f, 0.14f);
        CheckClose(f[0], yaw, 1e-6f, "feature[0] = yaw");
        CheckClose(f[1], pitch, 1e-6f, "feature[1] = pitch");
        CheckClose(f[2], yaw * yaw, 1e-6f, "feature[2] = yaw^2");
        CheckClose(f[3], pitch * pitch, 1e-6f, "feature[3] = pitch^2");
        CheckClose(f[4], yaw * pitch, 1e-6f, "feature[4] = yaw*pitch (the cross term the corners need)");
        CheckClose(f[5], yaw * yaw * yaw, 1e-6f, "feature[5] = yaw^3 (tan-reach term)");
        CheckClose(f[6], pitch * pitch * pitch, 1e-6f, "feature[6] = pitch^3");
        CheckClose(f[7], 0.11f, 1e-6f, "feature[7] = headYaw (linear)");
        CheckClose(f[10], 0.14f, 1e-6f, "feature[10] = headArea (linear)");
    }

    private static void TestEyeMUFeaturePolynomial()
    {
        //EyeMU regresses a screen POINT trained on portrait phones; the map onto a desktop screen is
        //nonlinear, so its calibration features now carry a polynomial of the normalized gaze point (like
        //the direction backbones carry one of the gaze angles) — otherwise a linear ridge compresses the
        //corners. Pin the length + exact layout so train/predict stay in lockstep, and so HeadPoseFeature-
        //Indices (11/12/13) keeps matching.
        Check(HomulerEyeMURunner.FeatureCount == 36, "EyeMU calibration feature vector is 36 terms (embedding + gaze polynomial + head pose + iris + context)");
        Check(HomulerEyeMURunner.IrisFeatureStart == 15, "EyeMU iris block starts after the head pose (11/12/13 stay stable)");
        Check(HomulerEyeMURunner.ContextFeatureStart == 19, "EyeMU context block starts after the iris block");
        var f = new float[HomulerEyeMURunner.FeatureCount];
        var emb = new[] { 0.1f, 0.2f, 0.3f, 0.4f };
        float gx = 0.25f, gy = 0.75f;
        HomulerEyeMURunner.FillEyeMUFeatures(f, emb, gx, gy, 0.11f, 0.12f, 0.13f, 0.14f);
        CheckClose(f[0], 0.1f, 1e-6f, "feature[0] = embedding[0]");
        CheckClose(f[3], 0.4f, 1e-6f, "feature[3] = embedding[3]");
        CheckClose(f[4], gx, 1e-6f, "feature[4] = gx (normalized gaze x)");
        CheckClose(f[5], gy, 1e-6f, "feature[5] = gy (normalized gaze y)");
        CheckClose(f[6], gx * gx, 1e-6f, "feature[6] = gx^2");
        CheckClose(f[7], gy * gy, 1e-6f, "feature[7] = gy^2");
        CheckClose(f[8], gx * gy, 1e-6f, "feature[8] = gx*gy (the cross term the corners need)");
        CheckClose(f[9], gx * gx * gx, 1e-6f, "feature[9] = gx^3 (reach term)");
        CheckClose(f[10], gy * gy * gy, 1e-6f, "feature[10] = gy^3");
        CheckClose(f[11], 0.11f, 1e-6f, "feature[11] = headYaw (matches HeadPoseFeatureIndices)");
        CheckClose(f[12], 0.12f, 1e-6f, "feature[12] = headPitch");
        CheckClose(f[13], 0.13f, 1e-6f, "feature[13] = headRoll");
        CheckClose(f[14], 0.14f, 1e-6f, "feature[14] = headArea (linear)");

        //Shared context block (both backbones use the same filler): tx/ty/dist, 8 eyeLook blendshapes,
        //then the gaze x pose/translation/distance interaction terms — the multiplicative structure of
        //the physical map (x ~ eyePos + D*tan(yaw+headYaw)) a per-axis linear ridge cannot represent.
        var tail = new float[HomulerFunctions.ContextTailCount];
        for (var i = 0; i < tail.Length; i++) tail[i] = 0.5f + i * 0.1f;   // tx=0.5, ty=0.6, dist=0.7, looks...
        var ctx = new float[HomulerFunctions.ContextFeatureCount];
        float gA = 0.3f, gB = -0.2f, hYaw = 0.11f, hPitch = 0.12f;
        HomulerFunctions.FillContextFeatures(ctx, 0, gA, gB, hYaw, hPitch, tail, 0);
        CheckClose(ctx[0], 0.5f, 1e-6f, "context[0] = tx");
        CheckClose(ctx[1], 0.6f, 1e-6f, "context[1] = ty");
        CheckClose(ctx[2], 0.7f, 1e-5f, "context[2] = dist");
        CheckClose(ctx[3], 0.8f, 1e-5f, "context[3] = first eyeLook blendshape");
        CheckClose(ctx[10], 1.5f, 1e-5f, "context[10] = last eyeLook blendshape");
        CheckClose(ctx[11], gA * hYaw, 1e-6f, "context[11] = gazeA*headYaw");
        CheckClose(ctx[12], gB * hPitch, 1e-6f, "context[12] = gazeB*headPitch");
        CheckClose(ctx[13], gA * 0.5f, 1e-6f, "context[13] = gazeA*tx");
        CheckClose(ctx[14], gB * 0.6f, 1e-6f, "context[14] = gazeB*ty");
        CheckClose(ctx[15], gA * 0.7f, 1e-5f, "context[15] = gazeA*dist");
        CheckClose(ctx[16], gB * 0.7f, 1e-5f, "context[16] = gazeB*dist");

        //Ensemble backbone: EyeMU's full vector leads (head-pose slots keep their indices), followed by the
        //direction model's leading gaze-angle polynomial block.
        Check(CompositeGazeBackbone.FeatureCount ==
              HomulerEyeMURunner.FeatureCount + GazeEstimationRunner.GazeAngleTermCount,
            "Ensemble feature vector = EyeMU block + direction gaze-angle polynomial");
        var eyeMu = new float[HomulerEyeMURunner.FeatureCount];
        var direction = new float[GazeEstimationRunner.FeatureCount];
        for (var i = 0; i < eyeMu.Length; i++) eyeMu[i] = i;
        for (var i = 0; i < direction.Length; i++) direction[i] = 100 + i;
        var combined = new float[CompositeGazeBackbone.FeatureCount];
        CompositeGazeBackbone.ConcatFeatures(eyeMu, direction, combined);
        Check(combined[0] == 0f && combined[HomulerEyeMURunner.FeatureCount - 1] == HomulerEyeMURunner.FeatureCount - 1,
            "Ensemble vector starts with the EyeMU block in order");
        Check(combined[HomulerEyeMURunner.FeatureCount] == 100f &&
              combined[CompositeGazeBackbone.FeatureCount - 1] == 100 + GazeEstimationRunner.GazeAngleTermCount - 1,
            "Ensemble vector ends with the direction model's first gaze-angle terms only");
    }

    private static void TestThinPlateSplineWarp()
    {
        //3x3 anchor grid in normalized screen coords.
        var grid = new List<Vector2>();
        foreach (var y in new[] { 0.1f, 0.5f, 0.9f })
            foreach (var x in new[] { 0.1f, 0.5f, 0.9f })
                grid.Add(new Vector2(x, y));
        var source = grid.ToArray();

        //Identity: mapping the grid onto itself must reproduce any probe (affine part carries it exactly).
        var identity = ThinPlateSplineWarp.Fit(source, (Vector2[])source.Clone());
        Check(identity != null, "TPS fits an identity mapping");
        var probe = new Vector2(0.37f, 0.62f);
        Check(Vector2.Distance(identity.Apply(probe), probe) < 1e-3f, "Identity TPS leaves points unchanged");

        //Pure translation: affine part must carry it everywhere, not just at anchors.
        var shifted = new Vector2[source.Length];
        var offset = new Vector2(0.1f, -0.05f);
        for (var i = 0; i < source.Length; i++) shifted[i] = source[i] + offset;
        var translation = ThinPlateSplineWarp.Fit(source, shifted);
        Check(Vector2.Distance(translation.Apply(probe), probe + offset) < 1e-3f,
            "A pure translation is reproduced everywhere");

        //LOCAL correction: move only the top-left anchor. The warp must correct strongly there while
        //leaving the far side of the screen essentially untouched — the local behavior the global
        //polynomial cannot express.
        var local = (Vector2[])source.Clone();
        local[0] = source[0] + new Vector2(0.08f, 0.06f);
        var warp = ThinPlateSplineWarp.Fit(source, local);
        Check(Vector2.Distance(warp.Apply(source[0]), local[0]) < 0.02f,
            "TPS corrects the displaced anchor toward its target");
        Check(Vector2.Distance(warp.Apply(source[8]), source[8]) < 0.02f,
            "TPS leaves the opposite corner essentially unchanged (local, not global)");

        //Serialization round trip (same JSON path Save/Load use).
        var json = JsonConvert.SerializeObject(warp);
        var loaded = JsonConvert.DeserializeObject<ThinPlateSplineWarp>(json);
        Check(Vector2.Distance(loaded.Apply(probe), warp.Apply(probe)) < 1e-5f,
            "TPS warp survives the serialization round trip");

        //Guards: too few anchors -> null; a malformed warp applies as identity.
        Check(ThinPlateSplineWarp.Fit(new[] { Vector2.zero, Vector2.one }, new[] { Vector2.zero, Vector2.one }) == null,
            "TPS refuses to fit on too few anchors");
        var malformed = new ThinPlateSplineWarp();
        Check(malformed.Apply(probe) == probe, "A malformed warp falls back to identity");

        //The property that makes HomulerGazeCalibration.TryBuildValidatedWarp's gate honest: a warp fitted
        //WITHOUT an anchor must not be able to reproduce that anchor's displacement. The old gate scored the
        //full warp on holdout samples whose target WAS an anchor, which it always "improves" by construction
        //— so it kept warps that did nothing between anchors. Leave-one-anchor-out is what makes the check
        //answer the deployment question (gaze lands everywhere, not only on calibration targets).
        var displaced = (Vector2[])source.Clone();
        var bump = new Vector2(0.08f, 0.06f);
        displaced[0] = source[0] + bump;                  //only the held-out anchor moves

        var withAnchor = ThinPlateSplineWarp.Fit(source, displaced);
        Check(Vector2.Distance(withAnchor.Apply(source[0]), displaced[0]) < 0.02f,
            "A warp WITH the anchor reproduces that anchor's displacement");

        var looSource = new List<Vector2>(source); looSource.RemoveAt(0);
        var looDest = new List<Vector2>(displaced); looDest.RemoveAt(0);
        var withoutAnchor = ThinPlateSplineWarp.Fit(looSource.ToArray(), looDest.ToArray());
        Check(withoutAnchor != null, "A warp still fits with one anchor left out");
        //Every remaining anchor is an identity pair, so an honest LOO warp leaves the held-out point alone.
        var looError = Vector2.Distance(withoutAnchor.Apply(source[0]), displaced[0]);
        Check(looError > 0.5f * bump.magnitude,
            $"A leave-one-anchor-out warp must NOT reproduce the held-out displacement (error {looError:F4}, " +
            $"displacement {bump.magnitude:F4}) - otherwise the validation gate is scoring its own fit");
    }

    private static void TestCalibrationProfiles()
    {
        //Path safety: a crafted profile may only write "<knownSubfolder>/<file>.json" — never traverse out.
        Check(CalibrationProfileStore.IsSafeRelativePath("RidgeRegression/Reg_X_EyeMU.json"), "A normal ridge profile path is safe");
        Check(CalibrationProfileStore.IsSafeRelativePath("MLP/MLP_EyeMU.json"), "A normal MLP profile path is safe");
        Check(!CalibrationProfileStore.IsSafeRelativePath("../evil.json"), "Path traversal is rejected");
        Check(!CalibrationProfileStore.IsSafeRelativePath("RidgeRegression/../../evil.json"), "Nested traversal is rejected");
        Check(!CalibrationProfileStore.IsSafeRelativePath("Unknown/x.json"), "An unknown subfolder is rejected");
        Check(!CalibrationProfileStore.IsSafeRelativePath("RidgeRegression/x.txt"), "A non-json profile entry is rejected");
        Check(!CalibrationProfileStore.IsSafeRelativePath("RidgeRegression/Reg_X:evil.json"),
            "Reserved name characters (NTFS alternate-data-stream ':') are rejected");

        //Sanitize turns a name into a safe file stem.
        Check(CalibrationProfileStore.Sanitize("MC-14-07-2026") == "MC-14-07-2026", "A clean profile name is unchanged");
        Check(!CalibrationProfileStore.Sanitize("a/b:c*d").Contains("/"), "Sanitize strips invalid file-name characters");

        //Serialize round-trip preserves the embedded calibration files verbatim.
        var profile = new CalibrationProfile { name = "t", backbone = "EyeMU" };
        profile.files["RidgeRegression/Reg_X_EyeMU.json"] = "{\"W\":[1,2,3]}";
        var round = JsonConvert.DeserializeObject<CalibrationProfile>(JsonConvert.SerializeObject(profile));
        Check(round.files["RidgeRegression/Reg_X_EyeMU.json"] == "{\"W\":[1,2,3]}",
            "Profile serialization preserves the embedded calibration file content");

        //The shipped MC-14-07-2026 profile must load and parse from Resources. It was saved with the
        //15-feature EyeMU vector (16 ridge weights incl. affine bias); the vector is now 19 (iris features),
        //so the profile is a LEGACY one — the contract is that it must load without error and its model must
        //return NaN on current-length features (RefineGazeLocation's raw-gaze fallback), never garbage.
        //Re-save the profile after recalibrating to make it current again.
        var shipped = Resources.Load<TextAsset>($"{CalibrationProfileStore.ResourcesFolder}/MC-14-07-2026");
        Check(shipped != null, "The shipped MC-14-07-2026 calibration profile should be in Resources");
        if (shipped != null)
        {
            var mc = JsonConvert.DeserializeObject<CalibrationProfile>(shipped.text);
            Check(mc.files.ContainsKey("RidgeRegression/Reg_X_EyeMU.json") &&
                  mc.files.ContainsKey("RidgeRegression/Reg_Y_EyeMU.json"),
                "MC-14-07-2026 profile contains the EyeMU ridge X/Y calibration files");
            var rx = JsonConvert.DeserializeObject<RidgeRegression>(mc.files["RidgeRegression/Reg_X_EyeMU.json"]);
            Check(rx.W != null, "MC-14-07-2026 ridge X parses with weights");
            if (rx.W != null && rx.W.Count != HomulerEyeMURunner.FeatureCount + 1)
                Check(float.IsNaN(rx.Predict(new float[HomulerEyeMURunner.FeatureCount])),
                    "A legacy-length profile model must NaN on current features (raw-gaze fallback), not mispredict");
        }
    }

    private static void TestGazeModelsLoadAndRun()
    {
        //Verifies the yakhyo/gaze-estimation ONNX models import and expose the I/O GazeEstimationRunner
        //codes against: one input (1,3,448,448) named "input", outputs "yaw"+"pitch" (90 bins each) PLUS
        //the "embedding" output added by graph edit (the pre-logit GAP vector the embedding-head
        //personalization regresses on: MobileOne 1024, MobileNetV2 1280, ResNet34 512).
        //Runs once with a blank CPU input so it works under -nographics. Does NOT prove gaze accuracy.
        var expectedEmbedding = new Dictionary<string, int>
        {
            ["ONNX/GazeEstimation/mobileone_s0_gaze"] = 1024,
            ["ONNX/GazeEstimation/mobilenetv2_gaze"] = 1280,
            ["ONNX/GazeEstimation/resnet34_gaze"] = 512,
        };
        foreach (var path in expectedEmbedding.Keys)
        {
            var asset = Resources.Load<ModelAsset>(path);
            Check(asset != null, $"Gaze model should load from Resources: {path}");
            if (asset == null) continue;

            var model = ModelLoader.Load(asset);
            Check(model.inputs.Count == 1, $"{path}: should have 1 input");
            Check(model.outputs.Count == 3, $"{path}: should have 3 outputs (yaw, pitch, embedding)");

            bool hasYaw = false, hasPitch = false, hasEmbedding = false;
            foreach (var o in model.outputs)
            {
                if (o.name == "yaw") hasYaw = true;
                if (o.name == "pitch") hasPitch = true;
                if (o.name == "embedding") hasEmbedding = true;
            }
            Check(hasYaw && hasPitch, $"{path}: outputs should be named yaw + pitch");
            Check(hasEmbedding, $"{path}: the embedding tap output should exist");

            Worker worker = null;
            Tensor<float> input = null;
            try
            {
                worker = new Worker(model, BackendType.CPU);
                input = new Tensor<float>(new TensorShape(1, 3, 448, 448), new float[3 * 448 * 448]);
                worker.SetInput("input", input);
                worker.Schedule();

                var yaw = worker.PeekOutput("yaw") as Tensor<float>;
                Check(yaw != null, $"{path}: 'yaw' output should exist");
                if (yaw != null)
                {
                    var bins = yaw.DownloadToArray();
                    Check(bins.Length == 90, $"{path}: yaw should have 90 bins (got {bins.Length})");
                }
                var embedding = worker.PeekOutput("embedding") as Tensor<float>;
                Check(embedding != null, $"{path}: 'embedding' output should run");
                if (embedding != null)
                {
                    var values = embedding.DownloadToArray();
                    Check(values.Length == expectedEmbedding[path],
                        $"{path}: embedding should be {expectedEmbedding[path]}-d (got {values.Length})");
                    var finite = true;
                    foreach (var v in values)
                        if (float.IsNaN(v) || float.IsInfinity(v)) finite = false;
                    Check(finite, $"{path}: embedding values should be finite");
                }
            }
            finally
            {
                input?.Dispose();
                worker?.Dispose();
            }
        }
    }

    #endregion

    #region OneEuroFilter

    private static void TestOneEuroFilter()
    {
        //Constant input converges to the input value
        var filter = new OneEuroFilter<Vector2>(60f, 1.0f, 0f, 1.0f);
        var result = Vector2.zero;
        for (int i = 0; i < 100; i++)
            result = filter.Filter(new Vector2(5f, 5f), i / 60f);
        CheckClose(result.x, 5f, 0.01f, "One Euro filter should converge to a constant input");

        //Alternating jitter around a fixed point is strongly attenuated
        var jitterFilter = new OneEuroFilter<Vector2>(60f, 1.0f, 0f, 1.0f);
        //Warm up to the center first
        for (int i = 0; i < 100; i++)
            jitterFilter.Filter(new Vector2(5f, 5f), i / 60f);
        var maxDeviation = 0f;
        for (int i = 100; i < 200; i++)
        {
            var raw = 5f + ((i % 2 == 0) ? 1f : -1f);
            var filtered = jitterFilter.Filter(new Vector2(raw, raw), i / 60f);
            maxDeviation = Mathf.Max(maxDeviation, Mathf.Abs(filtered.x - 5f));
        }
        Check(maxDeviation < 0.5f, $"One Euro filter should attenuate alternating jitter, max deviation was {maxDeviation}");
    }

    private static void TestOneEuroFilterVector2FastPath()
    {
        //FilterVector2 (the fast path the gaze pipeline uses every frame) must stay numerically
        //identical to the generic Filter<Vector2> AND keep the documented public currValue/prevValue
        //state in sync — the original fast path skipped the state update, so currValue silently read
        //(0,0) forever once callers switched to it.
        var generic = new OneEuroFilter<Vector2>(60f, 1.0f, 0.01f, 1.0f);
        var fast = new OneEuroFilter<Vector2>(60f, 1.0f, 0.01f, 1.0f);
        Vector2 g = Vector2.zero, f = Vector2.zero, fPrev = Vector2.zero;
        for (int i = 0; i < 50; i++)
        {
            var input = new Vector2(Mathf.Sin(i * 0.3f) * 100f, Mathf.Cos(i * 0.2f) * 50f);
            g = generic.Filter(input, i / 60f);
            fPrev = f;
            f = fast.FilterVector2(input, i / 60f);
        }
        CheckClose(f.x, g.x, 1e-4f, "FilterVector2 must match the generic Vector2 path (x)");
        CheckClose(f.y, g.y, 1e-4f, "FilterVector2 must match the generic Vector2 path (y)");
        CheckClose(fast.currValue.x, f.x, 1e-6f, "FilterVector2 must update currValue (was stuck at zero)");
        CheckClose(fast.currValue.y, f.y, 1e-6f, "FilterVector2 must update currValue (y)");
        CheckClose(fast.prevValue.x, fPrev.x, 1e-6f, "FilterVector2 must update prevValue");
    }

    private static void TestGazeStatistics()
    {
        //Known cluster: samples on a cross around (110, 205) against target (100, 200) -> bias (10, 5),
        //per-axis SD sqrt(mean of squared offsets).
        var samples = new List<Vector2>
        {
            new Vector2(108, 205), new Vector2(112, 205), new Vector2(110, 203), new Vector2(110, 207),
        };
        var s = GazeStatistics.Compute(samples, new Vector2(100, 200));
        CheckClose(s.bias.x, 10f, 1e-4f, "GazeStatistics bias x");
        CheckClose(s.bias.y, 5f, 1e-4f, "GazeStatistics bias y");
        CheckClose(s.sd.x, Mathf.Sqrt(2f), 1e-4f, "GazeStatistics per-axis SD x");
        CheckClose(s.sd.y, Mathf.Sqrt(2f), 1e-4f, "GazeStatistics per-axis SD y");
        Check(s.rmsS2S > 0f, "GazeStatistics computes sample-to-sample RMS");
        Check(s.bcea > 0f, "GazeStatistics computes a positive BCEA for a 2D cluster");

        //Aggregate: pure-bias targets (zero scatter) must yield precision 0 and accuracy = mean |bias|.
        var perTarget = new List<GazeStatistics.FixationStats>
        {
            new GazeStatistics.FixationStats { bias = new Vector2(3, 4), sd = Vector2.zero, count = 5 },
            new GazeStatistics.FixationStats { bias = new Vector2(-3, -4), sd = Vector2.zero, count = 5 },
        };
        GazeStatistics.Aggregate(perTarget, out var acc, out var meanBias, out var prec, out _);
        CheckClose(acc, 5f, 1e-4f, "GazeStatistics aggregate accuracy = mean |bias|");
        CheckClose(prec, 0f, 1e-4f, "GazeStatistics aggregate precision 0 for zero scatter");
        CheckClose(meanBias.x, 0f, 1e-4f, "GazeStatistics aggregate mean bias cancels opposing biases");
    }

    private static void TestDriftCorrector()
    {
        //Identity before any anchors.
        var corrector = new DriftCorrector();
        var p = new Vector2(0.3f, 0.7f);
        CheckClose((corrector.Apply(p) - p).magnitude, 0f, 1e-6f, "DriftCorrector starts at identity");

        //Feed anchors with a constant true translation of (+0.05, -0.03): the corrector must learn it.
        var rng = new System.Random(42);
        for (int i = 0; i < 40; i++)
        {
            var target = new Vector2(0.1f + 0.8f * (float)rng.NextDouble(), 0.1f + 0.8f * (float)rng.NextDouble());
            var predicted = target - new Vector2(0.05f, -0.03f);   // systematic drift
            corrector.AddAnchor(predicted, target);
        }
        var corrected = corrector.Apply(new Vector2(0.5f, 0.5f) - new Vector2(0.05f, -0.03f));
        CheckClose(corrected.x, 0.5f, 0.01f, "DriftCorrector learns a translation drift (x)");
        CheckClose(corrected.y, 0.5f, 0.01f, "DriftCorrector learns a translation drift (y)");
        Check(corrector.AcceptedAnchors > 30, "DriftCorrector accepts consistent anchors");

        //Outlier gate: a wild anchor (user clicked without looking) must be rejected.
        int before = corrector.AcceptedAnchors;
        bool accepted = corrector.AddAnchor(new Vector2(0.1f, 0.1f), new Vector2(0.9f, 0.9f));
        Check(!accepted && corrector.AcceptedAnchors == before, "DriftCorrector rejects an out-of-band anchor");

        //Gain drift: predictions compressed toward centre by 0.85 need >= the affine unlock to fix; feed
        //spread anchors and verify the gain is (partially, caps allowed) recovered.
        corrector.Reset();
        for (int i = 0; i < 60; i++)
        {
            var target = new Vector2(0.1f + 0.8f * (float)rng.NextDouble(), 0.1f + 0.8f * (float)rng.NextDouble());
            var predicted = new Vector2(0.5f + (target.x - 0.5f) * 0.85f, 0.5f + (target.y - 0.5f) * 0.85f);
            corrector.AddAnchor(predicted, target);
        }
        Check(corrector.AffineUnlocked, "DriftCorrector unlocks the affine DOFs with spread anchors");
        var edge = corrector.Apply(new Vector2(0.5f + 0.4f * 0.85f, 0.5f));
        CheckClose(edge.x, 0.9f, 0.02f, "DriftCorrector recovers a gain (scale) drift the translation-only re-center cannot");

        //Persistence round-trip.
        var json = corrector.SaveToJson();
        var restored = new DriftCorrector();
        restored.LoadFromJson(json);
        CheckClose((restored.Apply(p) - corrector.Apply(p)).magnitude, 0f, 1e-5f, "DriftCorrector state round-trips through JSON");
    }

    private static void TestFixationAggregator()
    {
        var aggregator = new FixationAggregator();
        //A tight cluster: after enough samples the aggregator enters fixation and returns the centroid.
        var rng = new System.Random(7);
        Vector2 lastOut = default;
        for (int i = 0; i < 12; i++)
        {
            var sample = new Vector2(500f + (float)rng.NextDouble() * 4f, 300f + (float)rng.NextDouble() * 4f);
            lastOut = aggregator.Add(sample, i * (1.0 / 30.0));
        }
        Check(aggregator.InFixation, "FixationAggregator detects a tight cluster as a fixation");
        Check(Mathf.Abs(lastOut.x - 502f) < 3f && Mathf.Abs(lastOut.y - 302f) < 3f,
            "FixationAggregator returns the cluster centroid");
        Check(aggregator.FixationDuration > 0.2, "FixationAggregator tracks fixation duration");

        //A saccade (large jump) must break the fixation and return the raw sample.
        var jumped = aggregator.Add(new Vector2(1500f, 800f), 0.5);
        Check(!aggregator.InFixation, "FixationAggregator ends the fixation on a saccade");
        CheckClose(jumped.x, 1500f, 1e-3f, "FixationAggregator passes raw samples through during saccades");
    }

    private static void TestPursuitCorrelator()
    {
        //Simulates the REAL clock relationship: the object position is sampled on the render clock
        //(t + latency, position at that same moment), while the gaze sample carries its camera CAPTURE
        //time t. The correlator must pair gaze at t with the object at t - pursuitLag, unaffected by the
        //pipeline latency — stamping the object with the gaze clock (the original bug) shifted every
        //anchor by latency x object-speed along the motion path.
        const double latency = 0.15;
        const float speed = 0.3f;
        var correlator = new PursuitCorrelator();
        bool certified = false;
        Vector2 pairedGaze = default, pairedTarget = default;
        for (int i = 0; i < 40; i++)
        {
            double t = i / 30.0;                                 // gaze capture time
            double renderT = t + latency;                        // object sampled on the render clock
            var target = new Vector2(0.2f + (float)renderT * speed, 0.5f);
            //The eye pursues with ~100ms lag: at capture time t it sits where the object was at t-0.1.
            var gaze = new Vector2(0.2f + Mathf.Max(0f, (float)t - 0.1f) * speed, 0.5f + 0.002f * (i % 3));
            certified |= correlator.Feed(gaze, t, target, renderT, out pairedGaze, out pairedTarget);
        }
        Check(certified, "PursuitCorrelator certifies gaze tracking a moving target");
        //With correct two-clock pairing the anchor pair must nearly coincide despite the 150ms latency
        //(the buggy single-clock pairing left a speed*(latency+2*lag) ≈ 0.10 gap here).
        Check(Mathf.Abs(pairedGaze.x - pairedTarget.x) < 0.02f,
            $"PursuitCorrelator pairs gaze with the lag-corrected target despite pipeline latency (gap {Mathf.Abs(pairedGaze.x - pairedTarget.x):F3})");

        //Duplicate gaze timestamps (render frames without a fresh camera sample) must not certify/emit.
        int emitted = 0;
        for (int i = 0; i < 5; i++)
            if (correlator.Feed(pairedGaze, 39 / 30.0, new Vector2(0.9f, 0.5f), 39 / 30.0 + latency, out _, out _))
                emitted++;
        Check(emitted == 0, "PursuitCorrelator ignores repeated gaze samples (one anchor per camera frame)");

        //Uncorrelated gaze (fixating while the target moves) must NOT certify.
        correlator.Reset();
        bool wrongCertified = false;
        for (int i = 0; i < 40; i++)
        {
            double t = i / 30.0;
            var target = new Vector2(0.2f + (float)t * speed, 0.5f);
            var gaze = new Vector2(0.55f + 0.003f * (i % 5), 0.48f);
            wrongCertified |= correlator.Feed(gaze, t, target, t + latency, out _, out _);
        }
        Check(!wrongCertified, "PursuitCorrelator rejects gaze that does not follow the target");
    }

    private static void TestAOIProbability()
    {
        //A fixation dead-centre in a large box ~ probability 1; far outside ~ 0; on the edge ~ 0.5.
        var box = new AOIBox("probBox", new Vector2(0.4f, 0.4f), new Vector2(0.6f, 0.6f));
        float sigma = 0.01f;   // tight ellipse vs a 0.2-wide box
        float inside = box.HitProbability(new Vector2(0.5f, 0.5f), sigma * sigma, 0f, sigma * sigma);
        float outside = box.HitProbability(new Vector2(0.9f, 0.9f), sigma * sigma, 0f, sigma * sigma);
        float edge = box.HitProbability(new Vector2(0.4f, 0.5f), sigma * sigma, 0f, sigma * sigma);
        Check(inside > 0.95f, $"AOI probability ~1 well inside (got {inside})");
        Check(outside < 0.05f, $"AOI probability ~0 far outside (got {outside})");
        Check(edge > 0.2f && edge < 0.8f, $"AOI probability ~0.5 on the border (got {edge})");
        //Determinism (fixed offsets): identical inputs -> identical probability.
        CheckClose(box.HitProbability(new Vector2(0.4f, 0.5f), sigma * sigma, 0f, sigma * sigma), edge, 1e-6f,
            "AOI probability is deterministic");

        //Margin: a point just outside the exact shape counts as inside with a margin.
        var strict = new AOIBox("strict", new Vector2(0.4f, 0.4f), new Vector2(0.6f, 0.6f));
        Check(!strict.CheckAOIWithMargin(new Vector2(0.39f, 0.5f)), "No margin: just-outside point misses");
        strict.margin = 0.02f;
        Check(strict.CheckAOIWithMargin(new Vector2(0.39f, 0.5f)), "Margin: just-outside point hits");
    }

    private static void TestGazeErrorModel()
    {
        var model = new GazeErrorModel();
        model.AddAnchor(new Vector2(0.25f, 0.5f), new Vector2(0.02f, 0f), 0.001f, 0f, 0.001f);
        model.AddAnchor(new Vector2(0.75f, 0.5f), new Vector2(-0.02f, 0f), 0.004f, 0f, 0.004f);
        //Query AT an anchor: (near-)exact bias thanks to the inverse-distance weighting epsilon.
        model.Query(new Vector2(0.25f, 0.5f), out var bias, out var cxx, out _, out _);
        CheckClose(bias.x, 0.02f, 0.005f, "GazeErrorModel returns the anchor's bias at the anchor");
        //Query midway: interpolated bias ~0, covariance between the anchors'.
        model.Query(new Vector2(0.5f, 0.5f), out var midBias, out var midCxx, out _, out _);
        Check(Mathf.Abs(midBias.x) < 0.01f, "GazeErrorModel interpolates bias between anchors");
        Check(midCxx > 0.001f && midCxx < 0.004f, "GazeErrorModel interpolates covariance between anchors");
        CheckClose(model.MeanErrorNormalized(), 0.02f, 1e-4f, "GazeErrorModel mean error magnitude");
    }

    private static void TestEmbeddingProjection()
    {
        //The sparse-JL projection that compresses the 512-1280-d model embedding to 64 calibration
        //features must be DETERMINISTIC across runs/platforms (saved calibrations depend on the exact
        //signs) and actually mix all inputs.
        var signs1 = GazeEstimationRunner.BuildEmbeddingSigns(512, GazeEstimationRunner.EmbeddingProjectionDim);
        var signs2 = GazeEstimationRunner.BuildEmbeddingSigns(512, GazeEstimationRunner.EmbeddingProjectionDim);
        bool identical = signs1.Length == signs2.Length;
        for (int i = 0; identical && i < signs1.Length; i++)
            identical = signs1[i] == signs2[i];
        Check(identical, "Embedding projection signs are deterministic");
        //Entries are ±1/sqrt(rawDim); the sign split should be roughly balanced.
        float expectedMagnitude = 1f / Mathf.Sqrt(512f);
        int positive = 0;
        bool magnitudeOk = true;
        foreach (var s in signs1)
        {
            if (Mathf.Abs(Mathf.Abs(s) - expectedMagnitude) > 1e-6f) magnitudeOk = false;
            if (s > 0) positive++;
        }
        Check(magnitudeOk, "Embedding projection entries are +/- 1/sqrt(rawDim)");
        float positiveFraction = (float)positive / signs1.Length;
        Check(positiveFraction > 0.45f && positiveFraction < 0.55f,
            $"Embedding projection signs are balanced (got {positiveFraction:F3} positive)");

        //Projection of a one-hot input reproduces that input's column of signs.
        var raw = new float[512];
        raw[37] = 2f;
        var dest = new float[10 + GazeEstimationRunner.EmbeddingProjectionDim];
        GazeEstimationRunner.ProjectEmbedding(raw, signs1, dest, 10, GazeEstimationRunner.EmbeddingProjectionDim);
        CheckClose(dest[10], 2f * signs1[0 * 512 + 37], 1e-6f, "Embedding projection computes the sign-weighted sum (k=0)");
        CheckClose(dest[10 + 63], 2f * signs1[63 * 512 + 37], 1e-6f, "Embedding projection computes the sign-weighted sum (k=63)");
    }

    private static void TestInteriorPreset()
    {
        var preset = new InteriorPreset(20f, 1.5f);
        var points = preset.GetPoints();
        Check(preset.StopAtWaypoints, "InteriorPreset dwells at its waypoints (TPS anchors)");
        Check(!preset.IsHeadMovement, "InteriorPreset is a sit-still preset (its dwells feed the warp)");
        Check(points.Count == 7, "InteriorPreset emits centre + 4 quadrant points (+ lead/close)");
        //All points strictly interior — the whole reason this preset exists.
        bool interior = true;
        foreach (var p in points)
            if (p.x < Screen.width * 0.2f || p.x > Screen.width * 0.8f ||
                p.y < Screen.height * 0.2f || p.y > Screen.height * 0.8f)
                interior = false;
        Check(interior, "InteriorPreset points are strictly interior (not on the boundary)");
    }

    #endregion
}
