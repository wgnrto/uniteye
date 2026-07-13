using System;
using UnityEngine;
using Random = System.Random;

namespace UnitEye
{
    /// <summary>
    /// Settings for optional calibration feature jitter. This augments only numerical gaze-model features;
    /// it must not be confused with image augmentation because calibration labels are screen targets.
    /// </summary>
    [Serializable]
    public class CalibrationFeatureAugmentationSettings
    {
        [Tooltip("Add bounded zero-mean jitter to training features. Holdout and evaluation samples are never augmented.")]
        public bool enabled;
        [Range(1, 5)]
        public int copiesPerSample = 1;
        [Range(0f, 0.5f)]
        public float standardDeviationScale = 0.05f;
        [Range(0.1f, 5f)]
        public float maximumStandardDeviations = 3f;
        public int seed = 12345;
    }

    /// <summary>Shared, deterministic feature-space augmentation used by both calibration trainers.</summary>
    public static class CalibrationFeatureAugmentation
    {
        public static float[][] Augment(float[][] features, CalibrationFeatureAugmentationSettings settings,
            Random random = null)
        {
            if (features == null)
                throw new ArgumentNullException(nameof(features));
            if (!IsEnabled(settings) || features.Length == 0)
                return features;

            var featureCount = features[0].Length;
            var mean = new double[featureCount];
            foreach (var sample in features)
            {
                if (sample == null || sample.Length != featureCount)
                    throw new ArgumentException("Calibration features must have a consistent dimensionality.");
                for (var feature = 0; feature < featureCount; feature++)
                    mean[feature] += sample[feature];
            }
            for (var feature = 0; feature < featureCount; feature++)
                mean[feature] /= features.Length;

            var standardDeviation = new float[featureCount];
            foreach (var sample in features)
                for (var feature = 0; feature < featureCount; feature++)
                {
                    var delta = sample[feature] - mean[feature];
                    standardDeviation[feature] += (float)(delta * delta);
                }
            for (var feature = 0; feature < featureCount; feature++)
                standardDeviation[feature] = (float)Math.Sqrt(standardDeviation[feature] / features.Length);

            random ??= new Random(settings.seed);
            var augmented = new float[features.Length * (settings.copiesPerSample + 1)][];
            for (var sampleIndex = 0; sampleIndex < features.Length; sampleIndex++)
            {
                augmented[sampleIndex] = features[sampleIndex];
                for (var copy = 1; copy <= settings.copiesPerSample; copy++)
                {
                    var jittered = (float[])features[sampleIndex].Clone();
                    for (var feature = 0; feature < featureCount; feature++)
                    {
                        var limit = standardDeviation[feature] * settings.standardDeviationScale *
                                    settings.maximumStandardDeviations;
                        jittered[feature] += NextGaussian(random) * standardDeviation[feature] *
                                             settings.standardDeviationScale;
                        jittered[feature] = Math.Max(features[sampleIndex][feature] - limit,
                            Math.Min(features[sampleIndex][feature] + limit, jittered[feature]));
                    }
                    augmented[copy * features.Length + sampleIndex] = jittered;
                }
            }
            return augmented;
        }

        public static float[] DuplicateTargets(float[] targets, CalibrationFeatureAugmentationSettings settings)
        {
            if (targets == null)
                throw new ArgumentNullException(nameof(targets));
            if (!IsEnabled(settings))
                return targets;

            var duplicated = new float[targets.Length * (settings.copiesPerSample + 1)];
            for (var copy = 0; copy <= settings.copiesPerSample; copy++)
                Array.Copy(targets, 0, duplicated, copy * targets.Length, targets.Length);
            return duplicated;
        }

        public static Vector2[] DuplicateTargets(Vector2[] targets, CalibrationFeatureAugmentationSettings settings)
        {
            if (targets == null)
                throw new ArgumentNullException(nameof(targets));
            if (!IsEnabled(settings))
                return targets;

            var duplicated = new Vector2[targets.Length * (settings.copiesPerSample + 1)];
            for (var copy = 0; copy <= settings.copiesPerSample; copy++)
                Array.Copy(targets, 0, duplicated, copy * targets.Length, targets.Length);
            return duplicated;
        }

        public static bool IsEnabled(CalibrationFeatureAugmentationSettings settings)
            => settings != null && settings.enabled && settings.copiesPerSample > 0 &&
               settings.standardDeviationScale > 0f;

        private static float NextGaussian(Random random)
        {
            //Ensure u1 is positive before applying Box-Muller.
            var u1 = Math.Max(double.Epsilon, random.NextDouble());
            var u2 = random.NextDouble();
            return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
    }
}
