using System;
using System.Collections.Generic;
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
            TestNoShippedDefaultCalibration();
            TestTrainerOnSyntheticData();
            TestSpatiallyBalancedCalibrationSamples();
            TestSimpleMLP();
            TestGazeGridQuantizer();
            TestOneEuroFilter();
            TestOneEuroFilterVector2FastPath();
            TestEyeCropRect();
            TestScenesAndPrefabsHaveNoMissingScripts();
            TestEyeMUModelLoadsAndRuns();
            TestGazeEstimationDecode();
            TestGazeFeaturePolynomial();
            TestGazeModelsLoadAndRun();
            TestCalibrationFileNames();
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
        var message = mlp.Train(x, y);
        Check(message.Contains("MLP Training done"), "SimpleMLP.Train should return the accuracy message");

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
            if (outT != null)
            {
                var data = outT.DownloadToArray();
                Check(data.Length >= 2, $"dense_8 should produce at least 2 values (got {data.Length})");
                if (data.Length >= 2)
                    Check(!float.IsNaN(data[0]) && !float.IsInfinity(data[0]) && !float.IsNaN(data[1]) && !float.IsInfinity(data[1]),
                        $"dense_8 output should be finite (got {data[0]}, {data[1]})");
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
        //L2CS decode: softmax + expectation over 90 bins, index*4deg - 180deg -> radians. Pure math.
        //Center bin (45) -> 45*4-180 = 0 deg; bin 0 -> -180 deg = -pi; bin 89 -> 176 deg.
        CheckClose(GazeEstimationRunner.DecodeAngleRadians(GazeBinSpike(45)), 0f, 0.02f, "Gaze decode: center bin ~ 0 rad");
        CheckClose(GazeEstimationRunner.DecodeAngleRadians(GazeBinSpike(0)), -Mathf.PI, 0.02f, "Gaze decode: bin 0 ~ -pi rad");
        CheckClose(GazeEstimationRunner.DecodeAngleRadians(GazeBinSpike(89)), (89f * 4f - 180f) * Mathf.Deg2Rad, 0.02f, "Gaze decode: bin 89");
    }

    private static void TestGazeFeaturePolynomial()
    {
        //The direction backbone must emit the polynomial-expanded calibration feature vector so a per-axis
        //LINEAR ridge can bend to the corners (raw [yaw,pitch] can't: the angle->screen map is nonlinear
        //with a yaw*pitch coupling). Pin the length + exact term layout so the basis isn't silently
        //changed and train/predict stay in lockstep (both read this same vector).
        Check(GazeEstimationRunner.FeatureCount == 11, "Gaze calibration feature vector is the 11-term polynomial");
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

    private static void TestGazeModelsLoadAndRun()
    {
        //Verifies both yakhyo/gaze-estimation ONNX models import and expose the I/O GazeEstimationRunner
        //codes against: one input (1,3,448,448) named "input", two outputs "yaw"+"pitch" of 90 bins each.
        //Runs once with a blank CPU input so it works under -nographics. Does NOT prove gaze accuracy.
        foreach (var path in new[] { "ONNX/GazeEstimation/mobileone_s0_gaze", "ONNX/GazeEstimation/mobilenetv2_gaze", "ONNX/GazeEstimation/resnet34_gaze" })
        {
            var asset = Resources.Load<ModelAsset>(path);
            Check(asset != null, $"Gaze model should load from Resources: {path}");
            if (asset == null) continue;

            var model = ModelLoader.Load(asset);
            Check(model.inputs.Count == 1, $"{path}: should have 1 input");
            Check(model.outputs.Count == 2, $"{path}: should have 2 outputs");

            bool hasYaw = false, hasPitch = false;
            foreach (var o in model.outputs)
            {
                if (o.name == "yaw") hasYaw = true;
                if (o.name == "pitch") hasPitch = true;
            }
            Check(hasYaw && hasPitch, $"{path}: outputs should be named yaw + pitch");

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

    #endregion
}
