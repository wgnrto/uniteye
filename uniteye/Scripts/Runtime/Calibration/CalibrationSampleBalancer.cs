using System;
using System.Collections.Generic;
using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// The pure sample-selection half of the calibration: boundary-target classification, per-target
    /// stability rejection, and spatial balancing.
    ///
    /// Extracted verbatim from HomulerGazeCalibration so the OFFLINE BENCHMARK trains on exactly the samples
    /// the shipped calibration would train on. A benchmark that reimplemented this would be measuring a
    /// pipeline nobody runs — the whole point is to detect whether a change helps the real thing. The
    /// MonoBehaviour now calls straight through to here, so the two cannot drift.
    /// </summary>
    public static class CalibrationSampleBalancer
    {
        //Guards a divide-by-variance when a feature is constant within a target group.
        public const double MinimumVarianceFloor = 1e-8;
        //Fixed so a repeat calibration on the same samples selects the same subset.
        public const int SpatialBalancingSeed = 12345;

        /// <summary>
        /// Whether a target sits in the outer ring of the 3x3 screen grid. Boundary targets get the stability
        /// rejection below; centre targets do not, because the extremes are where calibration quality is won
        /// or lost and where a bad sample does the most damage.
        /// </summary>
        public static bool IsBoundaryTarget(Vector2 targetPx, float screenWidth, float screenHeight)
        {
            var x = targetPx.x / screenWidth;
            var y = targetPx.y / screenHeight;
            return x <= 1f / 3f || x >= 2f / 3f || y <= 1f / 3f || y >= 2f / 3f;
        }

        /// <summary>
        /// Indices of the head yaw/pitch/roll features for a backbone, which the augmentation jitters
        /// specifically. Leaving this null silently disables the entire head-pose jitter block, so any caller
        /// driving the trainers directly must set it or it will be evaluating a configuration that never ships.
        /// </summary>
        public static int[] HeadPoseFeatureIndices(GazeBackbone backbone)
        {
            switch (backbone)
            {
                case GazeBackbone.GazeMobileOne:
                case GazeBackbone.GazeMobileNetV2:
                case GazeBackbone.GazeResNet34:
                    return new[] { 7, 8, 9 };
                default: // EyeMU + the ensemble (whose vector STARTS with the full EyeMU block)
                    return new[] { 11, 12, 13 };
            }
        }

        /// <summary>
        /// Rejects unstable/undersampled boundary-dwell samples, then spatially balances what remains so a
        /// dense sweep cannot outweigh the screen extremes. All list inputs are parallel and index-aligned.
        /// Throws InvalidOperationException when nothing survives — a degenerate capture must not train.
        /// </summary>
        /// <param name="report">Human-readable kept/rejected summary; the caller decides whether to log it.</param>
        public static void Build(
            IReadOnlyList<float[]> xData,
            IReadOnlyList<float> yXData, IReadOnlyList<float> yYData, IReadOnlyList<Vector2> yData,
            IReadOnlyList<Vector2> sampleTargets,
            IReadOnlyList<bool> capturedAtDwell, IReadOnlyList<bool> fromHeadRotation,
            float screenWidth, float screenHeight,
            int minimumCornerSamples, float cornerOutlierZScore,
            out float[][] features, out float[] targetsX, out float[] targetsY, out Vector2[] targets,
            out string report)
        {
            var keep = new bool[xData.Count];
            for (var i = 0; i < keep.Length; i++)
                keep[i] = true;
            var dwellGroups = new Dictionary<Vector2, List<int>>();
            for (var i = 0; i < xData.Count; i++)
            {
                //Head-movement samples are intentionally high-variance (the head yaw/pitch/roll features
                //swing while the eye fixates), so they are exempt from the per-target stability rejection
                //below — grouping them would flag that wanted variance as "unstable" and discard it.
                if (fromHeadRotation[i])
                    continue;
                if (!capturedAtDwell[i] || !IsBoundaryTarget(sampleTargets[i], screenWidth, screenHeight))
                    continue;
                if (!dwellGroups.TryGetValue(sampleTargets[i], out var group))
                {
                    group = new List<int>();
                    dwellGroups.Add(sampleTargets[i], group);
                }
                group.Add(i);
            }

            var rejected = 0;
            foreach (var group in dwellGroups.Values)
            {
                if (group.Count < minimumCornerSamples)
                {
                    foreach (var index in group) keep[index] = false;
                    rejected += group.Count;
                    continue;
                }

                var featureCount = xData[group[0]].Length;
                var mean = new double[featureCount];
                var variance = new double[featureCount];
                foreach (var index in group)
                    for (var feature = 0; feature < featureCount; feature++)
                        mean[feature] += xData[index][feature];
                for (var feature = 0; feature < featureCount; feature++)
                    mean[feature] /= group.Count;
                foreach (var index in group)
                    for (var feature = 0; feature < featureCount; feature++)
                    {
                        var delta = xData[index][feature] - mean[feature];
                        variance[feature] += delta * delta;
                    }
                for (var feature = 0; feature < featureCount; feature++)
                    variance[feature] = Math.Max(MinimumVarianceFloor, variance[feature] / group.Count);

                foreach (var index in group)
                {
                    double sumZSquared = 0;
                    for (var feature = 0; feature < featureCount; feature++)
                    {
                        var delta = xData[index][feature] - mean[feature];
                        sumZSquared += delta * delta / variance[feature];
                    }
                    if (Math.Sqrt(sumZSquared / featureCount) > cornerOutlierZScore)
                    {
                        keep[index] = false;
                        rejected++;
                    }
                }
            }

            var acceptedFeatures = new List<float[]>();
            var acceptedX = new List<float>();
            var acceptedY = new List<float>();
            var acceptedTargets = new List<Vector2>();
            for (var i = 0; i < xData.Count; i++)
            {
                if (!keep[i]) continue;
                acceptedFeatures.Add(xData[i]);
                acceptedX.Add(yXData[i]);
                acceptedY.Add(yYData[i]);
                acceptedTargets.Add(yData[i]);
            }
            if (acceptedFeatures.Count == 0)
                throw new InvalidOperationException("No valid calibration samples remain after corner quality checks. " +
                    "Increase Corner Dwell Seconds, lower Minimum Corner Samples, or relax Corner Outlier Z Score.");

            var selected = RidgeCalibrationTrainer.SpatiallyBalancedIndices(
                acceptedX, acceptedY, new System.Random(SpatialBalancingSeed));
            features = new float[selected.Length][];
            targetsX = new float[selected.Length];
            targetsY = new float[selected.Length];
            targets = new Vector2[selected.Length];
            for (var i = 0; i < selected.Length; i++)
            {
                var index = selected[i];
                features[i] = acceptedFeatures[index];
                targetsX[i] = acceptedX[index];
                targetsY[i] = acceptedY[index];
                targets[i] = acceptedTargets[index];
            }
            report = $"Calibration kept {acceptedFeatures.Count}/{xData.Count} raw samples " +
                     $"({rejected} unstable/undersampled boundary samples rejected), spatially balanced to {features.Length}.";
        }
    }
}
