using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// Turns a backbone's raw per-frame angular uncertainty (IGazeProvider.GazeAngularUncertainty) into
    /// numbers the rest of the pipeline can act on: a 0..1 confidence for weighting/gating, and a variance
    /// multiplier for the AOI error ellipse.
    ///
    /// WHY RELATIVE, NOT ABSOLUTE. The raw sigma is the spread of the model's own output distribution, and
    /// its scale is a property of the MODEL, not of the moment: the MobileGaze heads are broad on every
    /// frame (softmax max ~6x uniform), so any fixed threshold would be a per-backbone magic number tuned
    /// on one machine. What actually carries information is the DEVIATION from this session's own typical
    /// breadth — the frames where the crop went bad, the head turned away, or the lighting dropped. So the
    /// tracker keeps a slow per-axis baseline and scores each frame against it. That makes it
    /// self-calibrating across backbones, people and cameras, with no tuned constants.
    ///
    /// THE LIMITATION THIS IMPLIES, stated plainly: because the baseline adapts, a uniformly bad setup
    /// converges to "confident" — this detects transient degradation, not a globally poor configuration.
    /// Absolute quality is what the calibration holdout and the evaluation's error model measure; this is
    /// deliberately the complementary signal, not a replacement for either.
    ///
    /// Backbones with no distribution to measure (EyeMU) report NaN, which lands here as "no opinion":
    /// confidence 1, variance scale 1, <see cref="HasSignal"/> false. Nothing changes for them.
    /// </summary>
    public class GazeConfidence
    {
        /// <summary>Baseline EMA rate. ~100 samples of memory (a few seconds at typical gaze rates) —
        /// long enough that a handful of bad frames barely move it, short enough to follow a real change
        /// in seating or lighting.</summary>
        private const float BaselineAlpha = 0.01f;
        /// <summary>Frames observed before the baseline is trusted enough to score against.</summary>
        private const int WarmupSamples = 30;
        /// <summary>Confidence floor: a bad frame is down-weighted, never annihilated. Zero weight would
        /// let one pathological stretch silently drop every anchor and stall the drift corrector.</summary>
        private const float MinConfidence = 0.05f;
        /// <summary>Cap on ellipse inflation (3x sigma per axis). Beyond this the fixation is garbage and a
        /// bigger ellipse adds nothing but a wider spray of Monte-Carlo samples.</summary>
        private const float MaxVarianceScale = 9f;

        private float _baselineYaw, _baselinePitch;
        private int _count;

        /// <summary>True once a real (non-NaN) uncertainty has been observed and the warmup is over.</summary>
        public bool HasSignal { get; private set; }

        /// <summary>Latest 0..1 quality score: 1 at or below the session's typical spread, falling off as
        /// this frame's worst axis gets broader than usual. 1 when there is no signal.</summary>
        public float Confidence { get; private set; } = 1f;

        /// <summary>Per-axis multipliers for the error-ellipse VARIANCES (>= 1). 1 when there is no signal.</summary>
        public float VarianceScaleX { get; private set; } = 1f;
        public float VarianceScaleY { get; private set; } = 1f;

        /// <summary>Session baseline spread per axis in radians (diagnostics / UI). NaN before warmup.</summary>
        public Vector2 Baseline => _count >= WarmupSamples
            ? new Vector2(_baselineYaw, _baselinePitch)
            : new Vector2(float.NaN, float.NaN);

        public void Reset()
        {
            _baselineYaw = _baselinePitch = 0f;
            _count = 0;
            HasSignal = false;
            Confidence = 1f;
            VarianceScaleX = VarianceScaleY = 1f;
        }

        /// <summary>
        /// Feeds one frame's angular uncertainty (radians per axis; NaN = none) and refreshes the scores.
        /// Call once per FRESH gaze sample — feeding repeated render frames would shrink the effective
        /// baseline window without adding information.
        /// </summary>
        public void Observe(Vector2 sigmaRadians)
        {
            float sy = sigmaRadians.x, sp = sigmaRadians.y;
            if (float.IsNaN(sy) || float.IsNaN(sp) || float.IsInfinity(sy) || float.IsInfinity(sp) ||
                sy < 0f || sp < 0f)
            {
                //No opinion — and deliberately WITHOUT resetting the baseline: a backbone that drops a
                //frame (face lost) should not restart the warmup it already paid for.
                HasSignal = false;
                Confidence = 1f;
                VarianceScaleX = VarianceScaleY = 1f;
                return;
            }

            if (_count == 0)
            {
                //Seed on the first sample rather than ramping up from zero, which would make every early
                //frame look catastrophically broad against a near-zero baseline.
                _baselineYaw = sy;
                _baselinePitch = sp;
            }
            else
            {
                _baselineYaw += BaselineAlpha * (sy - _baselineYaw);
                _baselinePitch += BaselineAlpha * (sp - _baselinePitch);
            }
            _count++;

            if (_count < WarmupSamples)
            {
                HasSignal = false;
                Confidence = 1f;
                VarianceScaleX = VarianceScaleY = 1f;
                return;
            }

            HasSignal = true;
            float ratioYaw = Ratio(sy, _baselineYaw);
            float ratioPitch = Ratio(sp, _baselinePitch);

            //Variances scale with the SQUARE of the spread ratio, and only ever upward: a frame that is
            //sharper than usual is not evidence that the calibration's measured error shrank.
            VarianceScaleX = Mathf.Clamp(ratioYaw * ratioYaw, 1f, MaxVarianceScale);
            VarianceScaleY = Mathf.Clamp(ratioPitch * ratioPitch, 1f, MaxVarianceScale);

            //The worst axis governs the scalar score — a frame with a fine yaw and a hopeless pitch is a
            //bad frame.
            float worst = Mathf.Max(ratioYaw, ratioPitch);
            Confidence = Mathf.Clamp(1f / worst, MinConfidence, 1f);
        }

        //Spread relative to baseline, >= 1 by construction on the "worse" side; a degenerate baseline
        //(all-spike distributions) yields 1 rather than a division blow-up.
        private static float Ratio(float sigma, float baseline)
            => baseline > 1e-6f ? Mathf.Max(1f, sigma / baseline) : 1f;
    }
}
