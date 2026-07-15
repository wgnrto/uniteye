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
        //Default ON: augmentation is the default calibration approach. Each captured sample also yields
        //`copiesPerSample` bounded-jitter copies, which regularizes the fit and reduces overcommitment to
        //the exact captured samples. Holdout and evaluation samples are never augmented.
        [Tooltip("Add bounded zero-mean jitter to training features. Holdout and evaluation samples are never augmented.")]
        public bool enabled = true;
        [Range(1, 5)]
        public int copiesPerSample = 1;
        [Range(0f, 0.5f)]
        public float standardDeviationScale = 0.05f;
        [Range(0.1f, 5f)]
        public float maximumStandardDeviations = 3f;
        //Extra ABSOLUTE jitter for the head-pose feature slots named by headPoseFeatureIndices, given in
        //DEGREES (converted to the features' native radian scale internally). It models head-pose
        //measurement noise and guarantees the head yaw/pitch/roll features carry variance even when the
        //captured head barely moved, so the calibration does not overcommit to the exact calibration head
        //pose. Complements the head-movement capture stage (real variance); keep it small vs the real range.
        [Tooltip("Extra absolute jitter (degrees) added to the head-pose features only, modeling head-pose noise so the fit is less tied to the exact calibration head pose.")]
        [Range(0f, 15f)]
        public float headPoseJitterDegrees = 3f;
        public int seed = 12345;

        //Runtime hint (not serialized): which feature slots are head yaw/pitch/roll for the active gaze
        //backbone. Set by HomulerGazeCalibration before training; headPoseJitterDegrees applies to these.
        [NonSerialized]
        public int[] headPoseFeatureIndices;
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
            var headPoseIndices = settings.headPoseFeatureIndices;
            var headPoseJitter = settings.headPoseJitterDegrees;
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
                    //Extra absolute head-pose jitter on top of the proportional term (see the settings
                    //field comment). The head-pose features (head yaw/pitch/roll) are stored in RADIANS
                    //(FaceMeshSolution uses atan/atan2), so the degrees value is converted to radians here to
                    //match the feature scale. Bounded by the proportional limit PLUS the head-pose limit so
                    //the two jitter sources compose instead of the second clamp undoing the first.
                    if (headPoseIndices != null && headPoseJitter > 0f)
                    {
                        var headPoseJitterRad = headPoseJitter * Mathf.Deg2Rad;
                        foreach (var feature in headPoseIndices)
                        {
                            if (feature < 0 || feature >= featureCount)
                                continue;
                            jittered[feature] += NextGaussian(random) * headPoseJitterRad;
                            var limit = standardDeviation[feature] * settings.standardDeviationScale *
                                        settings.maximumStandardDeviations +
                                        headPoseJitterRad * settings.maximumStandardDeviations;
                            jittered[feature] = Math.Max(features[sampleIndex][feature] - limit,
                                Math.Min(features[sampleIndex][feature] + limit, jittered[feature]));
                        }
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
