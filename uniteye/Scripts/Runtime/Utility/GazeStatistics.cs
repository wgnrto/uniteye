using System.Collections.Generic;
using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// Standard eye-tracking signal statistics (static + testable): the ACCURACY / PRECISION / RMS-S2S
    /// decomposition every published tracker evaluation uses.
    ///
    /// Why this exists: a single RMSE conflates two very different error sources —
    ///   ACCURACY  = the systematic offset (bias) of the mean gaze from the target. Filtering and
    ///               fixation-averaging cannot reduce it; calibration/drift work can.
    ///   PRECISION = the sample scatter around the mean (SD / BCEA). Averaging N samples of a fixation
    ///               shrinks it ~sqrt(N) — IF the noise is white.
    ///   RMS-S2S   = root-mean-square of SAMPLE-TO-SAMPLE steps. For white noise RMS-S2S/SD == sqrt(2);
    ///               a materially smaller ratio means the noise is colored (already-filtered/drifting),
    ///               in which case averaging saturates far short of sqrt(N).
    /// The ratio therefore decides which half of the accuracy roadmap can pay: bias-dominated error wants
    /// calibration/drift/geometry work, jitter-dominated error wants resolution/aggregation/filtering.
    /// </summary>
    public static class GazeStatistics
    {
        /// <summary>Statistics of one fixation's samples against a known target.</summary>
        public struct FixationStats
        {
            public Vector2 bias;        // mean(sample) - target
            public Vector2 sd;          // per-axis standard deviation around the mean
            public float cov;           // covariance of x,y around the mean (for error-ellipse consumers)
            public float rmsS2S;        // RMS of consecutive-sample distances
            public float bcea;          // bivariate contour ellipse area (P=0.68), same units^2 as input
            public int count;
        }

        /// <summary>
        /// Computes bias/precision/RMS-S2S/BCEA for a run of samples belonging to ONE target.
        /// </summary>
        public static FixationStats Compute(IReadOnlyList<Vector2> samples, Vector2 target)
        {
            var stats = new FixationStats { count = samples?.Count ?? 0 };
            if (samples == null || samples.Count == 0)
                return stats;

            //Mean
            Vector2 mean = Vector2.zero;
            for (int i = 0; i < samples.Count; i++) mean += samples[i];
            mean /= samples.Count;
            stats.bias = mean - target;

            //Per-axis variance + covariance (population), and sample-to-sample steps
            float varX = 0f, varY = 0f, cov = 0f, s2s = 0f;
            for (int i = 0; i < samples.Count; i++)
            {
                var d = samples[i] - mean;
                varX += d.x * d.x;
                varY += d.y * d.y;
                cov += d.x * d.y;
                if (i > 0)
                    s2s += (samples[i] - samples[i - 1]).sqrMagnitude;
            }
            varX /= samples.Count;
            varY /= samples.Count;
            cov /= samples.Count;
            stats.cov = cov;
            stats.sd = new Vector2(Mathf.Sqrt(varX), Mathf.Sqrt(varY));
            stats.rmsS2S = samples.Count > 1 ? Mathf.Sqrt(s2s / (samples.Count - 1)) : 0f;

            //BCEA (P = 0.68 -> k = -ln(1-P) ≈ 1.14): 2kπ·sdx·sdy·sqrt(1-ρ²)
            float sdProduct = stats.sd.x * stats.sd.y;
            if (sdProduct > 1e-12f)
            {
                float rho = cov / sdProduct;
                float rhoTerm = Mathf.Sqrt(Mathf.Max(0f, 1f - rho * rho));
                stats.bcea = 2f * 1.14f * Mathf.PI * sdProduct * rhoTerm;
            }
            return stats;
        }

        /// <summary>
        /// Aggregates per-target stats into the session decomposition:
        /// accuracy = mean |bias|; precision = RMS of the per-target SDs (per axis, then magnitude);
        /// whiteness = mean RMS-S2S / mean SD-magnitude (≈1.41 for white noise, lower = colored).
        /// </summary>
        public static void Aggregate(IReadOnlyList<FixationStats> perTarget,
            out float accuracy, out Vector2 accuracyBias, out float precision, out float whiteness)
        {
            accuracy = 0f; precision = 0f; whiteness = 0f; accuracyBias = Vector2.zero;
            if (perTarget == null || perTarget.Count == 0) return;

            int used = 0;
            float sdSq = 0f, sdMagSum = 0f, s2sSum = 0f;
            for (int i = 0; i < perTarget.Count; i++)
            {
                var t = perTarget[i];
                if (t.count == 0) continue;
                used++;
                accuracy += t.bias.magnitude;
                accuracyBias += t.bias;
                sdSq += t.sd.sqrMagnitude;
                sdMagSum += t.sd.magnitude;
                s2sSum += t.rmsS2S;
            }
            if (used == 0) return;
            accuracy /= used;
            accuracyBias /= used;
            precision = Mathf.Sqrt(sdSq / used);
            float meanSd = sdMagSum / used;
            whiteness = meanSd > 1e-9f ? (s2sSum / used) / meanSd : 0f;
        }
    }
}
