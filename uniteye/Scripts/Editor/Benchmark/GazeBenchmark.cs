using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace UnitEye.Benchmark
{
    /// <summary>
    /// Runs an honest accuracy benchmark over every donated calibration session, so "did this change help?"
    /// gets answered against all the data at once instead of by re-running one calibration by hand.
    ///
    /// Menu: UnitEye > Run Gaze Benchmark. Headless:
    ///   Unity.exe -batchmode -projectPath [host project] -executeMethod UnitEye.Benchmark.GazeBenchmark.Run
    ///
    /// TWO THINGS MAKE THIS TRUSTWORTHY, and both are easy to get wrong:
    ///
    /// 1. THE SPLIT IS BY SCREEN LOCATION, NOT BY SAMPLE. Consecutive dwell rows are the same target, same
    ///    head pose, milliseconds apart — near-duplicates. A per-sample split (which is what the shipped
    ///    trainer does internally) puts near-copies on both sides and reports a flatteringly low error. Here
    ///    one whole dwell location is held out per fold, and a spatial BUFFER additionally removes sweep rows
    ///    that pass through it — without that the "held out" location is not held out at all.
    /// 2. IT TRAINS THROUGH THE SHIPPED CODE. CalibrationSampleBalancer and RidgeCalibrationTrainer are the
    ///    same types the calibration uses, with the same augmentation settings and head-pose feature indices.
    ///    A reimplementation would benchmark a pipeline nobody runs.
    ///
    /// The numbers this reports are EXPECTED TO BE WORSE than summary.json's holdoutRmseCm, because that one
    /// is measured on the leaky per-sample split. Both are printed side by side; a config that beats the
    /// app's self-report means the split has broken, not that the config is brilliant.
    /// </summary>
    public static class GazeBenchmark
    {
        /// <summary>Fraction of the screen diagonal around a held-out target that is also removed from train.</summary>
        public const float BufferFraction = 0.06f;
        /// <summary>A session needs this many distinct sit-still dwell locations to be worth folding.</summary>
        public const int MinimumLocations = 8;
        public const int MinimumRows = 300;

        /// <summary>Which calibration head to fit. Both are scored in PIXELS so the numbers are comparable.</summary>
        public enum Head { Ridge, Mlp }

        public class Config
        {
            public string Name = "ridge";
            public Head Model = Head.Ridge;
            public bool Augmentation = true;
            public float HeadPoseJitterDegrees = 3f;
            /// <summary>Fixed so a re-run on unchanged data reproduces the number exactly.</summary>
            public int Seed = 42;
        }

        public class SessionResult
        {
            public string Token, Backbone, GroupKey, Status = "ok";
            public int FeatureLength, Locations, TestRows;
            public double RmsePctDiag = double.NaN;
            public double RmseDegrees = double.NaN;
            public float AppHoldoutRmseCm = -1f;
            /// <summary>
            /// Error of the trivial predictor — always answer "screen centre" — over the SAME folds and the
            /// SAME test rows. A session's difficulty (how spread its targets are, how big the screen is,
            /// how much the participant moved) sets this, so dividing by it removes exactly the part of the
            /// score that is about the session rather than the system.
            /// </summary>
            public double CentreBaselinePctDiag = double.NaN;
            /// <summary>
            /// RmsePctDiag / CentreBaselinePctDiag. Below 1 means the calibration beat "always guess
            /// centre"; 1 means it learned nothing. THIS is the number that may be compared across
            /// backbones and feature layouts — the absolute error may not, which is why the per-group
            /// aggregation refuses to pool it.
            /// </summary>
            public double SkillScore = double.NaN;
        }

        [MenuItem("UnitEye/Run Gaze Benchmark")]
        public static void RunFromMenu() => Execute(DefaultConfigs(), interactive: true);

        /// <summary>Headless entry point. Exits 0 on success, 1 if nothing could be benchmarked.</summary>
        public static void Run()
        {
            var ok = Execute(DefaultConfigs(), interactive: false);
            if (Application.isBatchMode) EditorApplication.Exit(ok ? 0 : 1);
        }

        //The shipped levers, so a run answers "which head, and does augmentation earn its keep?" out of the
        //box. MLP folds are ~200 epochs each and dominate the runtime — drop them from this list if you only
        //want a quick ridge A/B.
        private static List<Config> DefaultConfigs() => new List<Config>
        {
            new Config { Name = "ridge-aug",   Model = Head.Ridge, Augmentation = true },
            new Config { Name = "ridge-noaug", Model = Head.Ridge, Augmentation = false },
            new Config { Name = "mlp-aug",     Model = Head.Mlp,   Augmentation = true },
            new Config { Name = "mlp-noaug",   Model = Head.Mlp,   Augmentation = false },
        };

        private static bool Execute(List<Config> configs, bool interactive)
        {
            var sessions = GazeDatasetReader.LoadAll();
            var usable = new List<GazeDatasetReader.Session>();
            foreach (var s in sessions)
            {
                if (!s.Usable) continue;
                if (s.Samples.Count < MinimumRows) { s.Status = "excluded:too-few-rows"; continue; }
                usable.Add(s);
            }

            if (usable.Count == 0)
            {
                var msg = sessions.Count == 0
                    ? $"No recordings found under {Path.Combine(Application.persistentDataPath, "UnitEyeRecordings")}."
                    : $"{sessions.Count} session(s) found, none usable: " +
                      string.Join(", ", sessions.Select(s => $"{s.Token}={s.Status}"));
                Debug.LogWarning("UNITEYE_BENCHMARK: " + msg);
                if (interactive) EditorUtility.DisplayDialog("Gaze benchmark", msg, "OK");
                return false;
            }

            var report = new StringBuilder();
            report.AppendLine("# unitEyeBenchmark v1");
            report.AppendLine($"# sessions\t{usable.Count} usable of {sessions.Count}");
            report.AppendLine($"# split\tleave-one-target-out buffer={BufferFraction.ToString("F3", CultureInfo.InvariantCulture)}");
            report.AppendLine("# note\tthese numbers are expected to be WORSE than appHoldout (which uses a leaky per-sample split)");
            report.AppendLine("# skill\trmse / centre-baseline rmse on the same rows; <1 beats \"always guess centre\". " +
                              "Compare BACKBONES on this, never on rmse_pctdiag");
            report.AppendLine("config\tsession\tbackbone\tfeatures\tlocations\ttestRows\trmse_pctdiag\trmse_deg\t" +
                              "centre_pctdiag\tskill\tappHoldout_cm\tstatus");

            foreach (var config in configs)
            {
                var results = new List<SessionResult>();
                for (var i = 0; i < usable.Count; i++)
                {
                    var s = usable[i];
                    if (interactive && EditorUtility.DisplayCancelableProgressBar("Gaze benchmark",
                            $"{config.Name}: {s.Token} ({i + 1}/{usable.Count})", (float)i / usable.Count))
                    {
                        EditorUtility.ClearProgressBar();
                        return false;
                    }
                    results.Add(BenchmarkSession(s, config));
                }
                if (interactive) EditorUtility.ClearProgressBar();

                foreach (var r in results.OrderBy(r => r.Token, StringComparer.Ordinal))
                    report.AppendLine(string.Join("\t", config.Name, r.Token, r.Backbone,
                        r.FeatureLength.ToString(CultureInfo.InvariantCulture),
                        r.Locations.ToString(CultureInfo.InvariantCulture),
                        r.TestRows.ToString(CultureInfo.InvariantCulture),
                        Fmt(r.RmsePctDiag), Fmt(r.RmseDegrees),
                        Fmt(r.CentreBaselinePctDiag), Fmt(r.SkillScore),
                        r.AppHoldoutRmseCm >= 0 ? Fmt(r.AppHoldoutRmseCm) : "-", r.Status));

                //Aggregate per feature-layout group. Absolute accuracy never pools across groups: a 36-value
                //EyeMU vector and a 32-value direction vector describe different systems.
                foreach (var group in results.Where(r => r.Status == "ok").GroupBy(r => r.GroupKey))
                {
                    var vals = group.Select(r => r.RmsePctDiag).Where(v => !double.IsNaN(v)).OrderBy(v => v).ToList();
                    if (vals.Count == 0) continue;
                    var median = vals[vals.Count / 2];
                    //Median and IQR, not mean and SD: with a handful of sessions one bad capture would drag
                    //a mean around and hide the change under test.
                    var q1 = vals[vals.Count / 4];
                    var q3 = vals[Mathf.Min(vals.Count - 1, 3 * vals.Count / 4)];
                    report.AppendLine($"# summary\t{config.Name}\t{group.Key}\tn={vals.Count}" +
                                      $"\tmedian={Fmt(median)}\tIQR={Fmt(q1)}..{Fmt(q3)}");

                    //The leak sanity check. If the honest split beats the leaky one, the split is broken.
                    var paired = group.Where(r => r.AppHoldoutRmseCm > 0 && !double.IsNaN(r.RmseDegrees)).ToList();
                    var suspicious = paired.Count(r => ToCm(r, usable) < r.AppHoldoutRmseCm);
                    if (suspicious > paired.Count / 2 && paired.Count >= 3)
                        report.AppendLine($"# WARNING\t{config.Name}\t{group.Key}\t{suspicious}/{paired.Count} sessions " +
                                          "scored BETTER than the app's leaky holdout - the split is probably broken, " +
                                          "not the config improved");
                }

                AppendBackboneRanking(report, config.Name, results);
            }

            var outPath = Path.Combine(Application.persistentDataPath, "UnitEyeRecordings", "benchmark.tsv");
            Directory.CreateDirectory(Path.GetDirectoryName(outPath));
            File.WriteAllText(outPath, report.ToString(), new UTF8Encoding(false));
            Debug.Log("UNITEYE_BENCHMARK_DONE\n" + report);
            if (interactive)
            {
                EditorUtility.RevealInFinder(outPath);
                EditorUtility.DisplayDialog("Gaze benchmark",
                    $"Benchmarked {usable.Count} session(s) across {configs.Count} config(s).\n\n{outPath}", "OK");
            }
            return true;
        }

        /// <summary>
        /// Cross-backbone ranking on the SKILL score.
        ///
        /// This exists because the per-group summary above deliberately refuses to pool absolute error
        /// across feature layouts — correctly, since a 36-value EyeMU vector and a 36-value direction vector
        /// describe different systems — and that left the question "which backbone is actually better?"
        /// unanswerable from this report. It is not an academic question: resnet34_gaze was written off as
        /// having a dead yaw head, and that verdict was reached through the windowed decode that 8c66d89
        /// removed. Re-scored against the centre baseline it came out at 0.209 versus 0.4721, i.e. clearly
        /// informative. Every backbone comparison made before that decode change is suspect for the same
        /// reason, so the report needs to make re-deriving one cheap.
        ///
        /// Skill is comparable where absolute error is not, because it divides out the per-session
        /// difficulty that the different backbones were never measured against a common version of.
        ///
        /// TWO CAVEATS, both fatal to a naive reading, so they are printed with the table:
        ///  - Each session used ONE backbone, so this is a between-subjects comparison. With a handful of
        ///    sessions per backbone the participant and their setup are confounded with the model.
        ///  - Sessions of unknown feature-pipeline generation (v0, recorded before the field existed) are
        ///    listed separately: they may have been captured under the old decode, and pooling them would
        ///    reintroduce exactly the contamination this is trying to undo.
        /// </summary>
        private static void AppendBackboneRanking(StringBuilder report, string configName, List<SessionResult> results)
        {
            var scored = results.Where(r => r.Status == "ok" && !double.IsNaN(r.SkillScore)).ToList();
            if (scored.Count == 0) return;

            report.AppendLine($"# ranking\t{configName}\tskill = rmse / centre-baseline; lower is better, " +
                              "1.0 = learned nothing. BETWEEN-SUBJECTS: each session used one backbone");
            foreach (var group in scored.GroupBy(r => $"{r.Backbone}/v{PipelineVersionOf(r)}")
                                        .OrderBy(g => g.Select(r => r.SkillScore).Average()))
            {
                var vals = group.Select(r => r.SkillScore).OrderBy(v => v).ToList();
                var median = vals[vals.Count / 2];
                var best = vals[0];
                var worst = vals[vals.Count - 1];
                report.AppendLine($"# ranking\t{configName}\t{group.Key}\tn={vals.Count}\tmedian_skill={Fmt(median)}" +
                                  $"\tbest={Fmt(best)}\tworst={Fmt(worst)}");
            }
            if (scored.Any(r => PipelineVersionOf(r) == 0))
                report.AppendLine($"# ranking\t{configName}\tNOTE some sessions are pipeline v0 (unknown generation, " +
                                  "possibly the pre-8c66d89 decode) and are ranked in their own rows - do not merge them");
        }

        //The generation is carried in GroupKey's trailing "/v<n>" segment (see GazeDatasetReader.GroupKey).
        private static int PipelineVersionOf(SessionResult r)
        {
            var i = r.GroupKey?.LastIndexOf("/v", StringComparison.Ordinal) ?? -1;
            return i >= 0 && int.TryParse(r.GroupKey.Substring(i + 2), NumberStyles.Integer,
                CultureInfo.InvariantCulture, out var v) ? v : 0;
        }

        private static double ToCm(SessionResult r, List<GazeDatasetReader.Session> sessions)
        {
            var s = sessions.FirstOrDefault(x => x.Token == r.Token);
            if (s == null || s.ScreenWidthCm <= 0) return double.MaxValue;
            var diagCm = Math.Sqrt(s.ScreenWidthCm * s.ScreenWidthCm + s.ScreenHeightCm * s.ScreenHeightCm);
            return r.RmsePctDiag / 100.0 * diagCm;
        }

        /// <summary>
        /// One session, one config: leave-one-target-out over its sit-still dwell locations.
        /// </summary>
        public static SessionResult BenchmarkSession(GazeDatasetReader.Session s, Config config)
        {
            var r = new SessionResult
            {
                Token = s.Token, Backbone = s.Backbone, GroupKey = s.GroupKey,
                FeatureLength = s.Samples[0].Features.Length, AppHoldoutRmseCm = s.AppHoldoutRmseCm,
            };

            float w = s.ScreenWidthPx, h = s.ScreenHeightPx;
            var diagPx = Mathf.Sqrt(w * w + h * h);
            var buffer = BufferFraction * diagPx;

            //Fold unit: distinct sit-still dwell locations. For dwell rows label == target exactly (the
            //pursuit-lag correction only applies while sweeping), so the key is exact and the result does not
            //depend on that heuristic.
            var locations = s.Samples
                .Where(x => x.Dwell && !x.HeadRotation)
                .Select(x => new Vector2(Mathf.Round(x.LabelPx.x), Mathf.Round(x.LabelPx.y)))
                .Distinct().ToList();
            r.Locations = locations.Count;
            if (locations.Count < MinimumLocations) { r.Status = "excluded:too-few-locations"; return r; }

            var headPoseIndices = CalibrationSampleBalancer.HeadPoseFeatureIndices(ParseBackbone(s.Backbone));
            double sumSq = 0, sumSqDeg = 0;
            int n = 0, nDeg = 0;
            var perLocation = new List<double>();
            var perLocationCentre = new List<double>();
            var screenCentre = new Vector2(w * 0.5f, h * 0.5f);

            foreach (var held in locations)
            {
                var train = new List<GazeDatasetReader.Sample>();
                var test = new List<GazeDatasetReader.Sample>();
                foreach (var sample in s.Samples)
                {
                    //The buffer is what makes the hold-out real: sweep rows pass straight through the held
                    //location carrying the correct label, so distance-gating on the LABEL (not the flag) is
                    //what removes them.
                    bool near = Vector2.Distance(sample.LabelPx, held) <= buffer;
                    if (!near) { train.Add(sample); continue; }
                    //Scored on still fixations only - the shipped evaluation measures a seated participant,
                    //so scoring deliberate head swings would make head-pose robustness look like a regression.
                    if (sample.Dwell && !sample.HeadRotation) test.Add(sample);
                }
                if (test.Count == 0 || train.Count < 50) continue;

                float[][] features; float[] tX, tY; Vector2[] targets;
                try
                {
                    CalibrationSampleBalancer.Build(
                        train.Select(x => x.Features).ToList(),
                        train.Select(x => x.LabelPx.x / w).ToList(),
                        train.Select(x => x.LabelPx.y / h).ToList(),
                        train.Select(x => x.LabelPx).ToList(),
                        train.Select(x => x.TargetPx).ToList(),
                        train.Select(x => x.Dwell).ToList(),
                        train.Select(x => x.HeadRotation).ToList(),
                        w, h, 15, 3f, out features, out tX, out tY, out targets, out _);
                }
                catch (Exception) { continue; }   // degenerate fold; other folds still count

                var augmentation = config.Augmentation
                    ? new CalibrationFeatureAugmentationSettings
                      { headPoseJitterDegrees = config.HeadPoseJitterDegrees, headPoseFeatureIndices = headPoseIndices }
                    : null;

                //Both heads are reduced to one PIXEL-space predictor here, because they natively disagree:
                //the ridge pair predicts in normalized units (CalibrationModelStore scales them afterwards)
                //while SimpleMLP is trained on pixel targets and predicts pixels directly. Scoring each in
                //its own space and converting later is how a benchmark ends up silently comparing a
                //normalized error against a pixel one.
                Func<float[], Vector2> predictPx;
                try
                {
                    if (config.Model == Head.Mlp)
                    {
                        //Seeded so a re-run reproduces the number; the app leaves it unseeded.
                        var mlp = new SimpleMLP(config.Seed);
                        mlp.Train(features, targets, augmentation);
                        predictPx = f => mlp.Predict(f);
                    }
                    else
                    {
                        var fit = RidgeCalibrationTrainer.Train(features, tX, tY,
                            rmseScaleX: 1f, rmseScaleY: 1f, augmentation: augmentation);
                        predictPx = f =>
                        {
                            var nx = fit.XModel.Predict(f);
                            var ny = fit.YModel.Predict(f);
                            return new Vector2(nx * w, ny * h);
                        };
                    }
                }
                catch (Exception) { continue; }

                double locSumSq = 0, locSumSqCentre = 0; int locN = 0;
                foreach (var sample in test)
                {
                    var p = predictPx(sample.Features);
                    if (float.IsNaN(p.x) || float.IsNaN(p.y)) continue;
                    double ex = p.x - sample.LabelPx.x;
                    double ey = p.y - sample.LabelPx.y;
                    double d = Math.Sqrt(ex * ex + ey * ey);
                    locSumSq += d * d; locN++;
                    sumSq += d * d; n++;
                    //Same row, trivial predictor. Accumulated inside the identical guard so the baseline is
                    //measured on exactly the rows the model was scored on — a baseline over a different row
                    //set would make the ratio meaningless.
                    double dc = (screenCentre - sample.LabelPx).magnitude;
                    locSumSqCentre += dc * dc;

                    //Degrees where the row recorded a viewing distance AND the session's centimetres are
                    //real. A windowed Editor Game view makes screenWidthCm wrong by the viewport ratio, and
                    //a wrong degree figure pooled into a median is worse than a missing one — % of diagonal
                    //remains valid there because it is a ratio of the same units.
                    if (s.PhysicalScaleTrustworthy &&
                        !float.IsNaN(sample.DistanceMm) && sample.DistanceMm > 1f && s.ScreenWidthCm > 0)
                    {
                        double mmPerPx = 10.0 * s.ScreenWidthCm / w;
                        double deg = Math.Atan(d * mmPerPx / sample.DistanceMm) * 180.0 / Math.PI;
                        sumSqDeg += deg * deg; nDeg++;
                    }
                }
                if (locN > 0)
                {
                    perLocation.Add(Math.Sqrt(locSumSq / locN));
                    perLocationCentre.Add(Math.Sqrt(locSumSqCentre / locN));
                }
            }

            if (n == 0 || perLocation.Count == 0) { r.Status = "excluded:no-scorable-folds"; return r; }
            r.TestRows = n;
            //Macro-average over locations so a target that happened to collect more samples does not dominate.
            r.RmsePctDiag = 100.0 * Math.Sqrt(perLocation.Sum(v => v * v) / perLocation.Count) / diagPx;
            r.CentreBaselinePctDiag = 100.0 * Math.Sqrt(perLocationCentre.Sum(v => v * v) / perLocationCentre.Count) / diagPx;
            if (r.CentreBaselinePctDiag > 1e-9) r.SkillScore = r.RmsePctDiag / r.CentreBaselinePctDiag;
            if (nDeg > 0) r.RmseDegrees = Math.Sqrt(sumSqDeg / nDeg);
            return r;
        }

        private static GazeBackbone ParseBackbone(string name)
            => Enum.TryParse<GazeBackbone>(name, out var b) ? b : GazeBackbone.EyeMU;

        private static string Fmt(double v)
            => double.IsNaN(v) ? "-" : v.ToString("F4", CultureInfo.InvariantCulture);
    }
}
