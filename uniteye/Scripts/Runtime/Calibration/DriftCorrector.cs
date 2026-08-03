using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// Online drift correction: a 6-parameter affine transform stacked ON TOP of the frozen calibration
    /// (ridge/MLP + TPS), updated from validated anchors — moments where the gaze target is known
    /// (validated clicks, correlation-gated pursuit of game objects, attention events, the re-center
    /// marker). Published webcam baselines lose ~50% accuracy over a 20-minute session to seating/posture/
    /// lighting drift; a slowly-adapted affine layer recovers most of it, and because it sits AFTER the
    /// frozen mapping it can never corrupt the trained calibration — at worst it converges back toward
    /// identity.
    ///
    /// Design (per the recalibration literature: a 6-DOF affine recovers most calibration loss, and
    /// translation is the dominant component):
    ///  - Per-axis recursive least squares with exponential forgetting over anchor pairs
    ///    (predicted -> true), solved in closed form from decayed sufficient statistics.
    ///  - Staged DOF: translation-only until enough well-SPREAD anchors exist in the forgetting window
    ///    (a burst of clicks in one screen region cannot fit a meaningful gain), then full affine.
    ///  - Magnitude caps: the correction is clamped so a run of bad anchors cannot fling the mapping.
    ///  - Outlier gate: anchors whose residual exceeds a robust threshold (MAD-based) are rejected —
    ///    users provably do NOT look at their click point ~1/3 of the time.
    /// Coordinates are NORMALIZED (0..1 screen) so the state is resolution-independent and persistable.
    /// </summary>
    public class DriftCorrector
    {
        //Exponential forgetting per accepted anchor (~0.99 keeps a ~100-anchor memory).
        private const float Forgetting = 0.99f;
        //Anchors needed, and minimum anchor spread (normalized SD), before gain/shear DOFs unlock.
        private const int MinAnchorsForAffine = 8;
        private const float MinSpreadForAffine = 0.12f;
        //Caps: |translation| in normalized units; gain kept within [1-cap, 1+cap]; cross terms within ±cap.
        private const float TranslationCap = 0.15f;
        private const float GainCap = 0.25f;
        //Residual outlier gate: reject anchors whose residual exceeds max(this floor, k * robust scale).
        private const float ResidualFloor = 0.18f;
        private const float ResidualMadK = 3f;
        //EMA of |residual| standing in for the MAD (a true windowed MAD would need a sample buffer).
        private float _residualScale = 0.06f;
        private const float ResidualScaleAlpha = 0.05f;

        //The published correction: corrected = A * predicted + b (normalized space).
        private float _a11 = 1f, _a12, _a21, _a22 = 1f;
        private float _bx, _by;

        public int AcceptedAnchors { get; private set; }
        public int RejectedAnchors { get; private set; }
        /// <summary>Magnitude of the current translation correction in normalized units (session-health metric).</summary>
        public float TranslationMagnitude => new Vector2(_bx, _by).magnitude;
        /// <summary>True once the anchor cloud unlocked the full 6-DOF fit (vs translation-only).</summary>
        public bool AffineUnlocked { get; private set; }

        /// <summary>Resets the correction to identity and forgets all anchors (e.g. after recalibration).</summary>
        public void Reset()
        {
            _rw = _rpx = _rpy = _rtx = _rty = 0f;
            _rpxpx = _rpypy = _rpxpy = _rtxpx = _rtxpy = _rtypx = _rtypy = 0f;
            _a11 = 1f; _a12 = 0f; _a21 = 0f; _a22 = 1f; _bx = _by = 0f;
            _residualScale = 0.06f;
            AcceptedAnchors = RejectedAnchors = 0;
            AffineUnlocked = false;
        }

        /// <summary>Applies the current correction to a predicted gaze point (normalized 0..1).</summary>
        public Vector2 Apply(Vector2 predicted)
        {
            return new Vector2(
                _a11 * predicted.x + _a12 * predicted.y + _bx,
                _a21 * predicted.x + _a22 * predicted.y + _by);
        }

        /// <summary>
        /// Feeds one anchor: the calibrated (pre-correction) gaze <paramref name="predicted"/> at a moment
        /// the user was provably looking at <paramref name="target"/> (both normalized 0..1).
        /// <paramref name="weight"/> scales the anchor's influence (clicks ~1, pursuit ~0.5, events ~0.3).
        /// Returns false if the anchor was rejected as an outlier.
        /// </summary>
        public bool AddAnchor(Vector2 predicted, Vector2 target, float weight = 1f)
        {
            if (float.IsNaN(predicted.x) || float.IsNaN(predicted.y) ||
                float.IsNaN(target.x) || float.IsNaN(target.y) || weight <= 0f)
                return false;

            //Outlier gate on the residual AFTER the current correction (what the user experiences).
            float residual = (Apply(predicted) - target).magnitude;
            float gate = Mathf.Max(ResidualFloor, ResidualMadK * _residualScale);
            //Always keep the robust scale tracking, accepted or not, so a genuine shift (all residuals
            //large and consistent) grows the gate open instead of locking every future anchor out.
            _residualScale += ResidualScaleAlpha * (residual - _residualScale);
            if (residual > gate)
            {
                RejectedAnchors++;
                return false;
            }

            //Decay + accumulate the sufficient statistics, then re-solve. Decayed raw sums are all the
            //affine solve needs (12 numbers, no sample buffer); values are in [0..~1] so the naive-sums
            //formulation is numerically fine.
            AccumulateDecayedSums(predicted, target, weight);

            AcceptedAnchors++;
            Solve();
            return true;
        }

        //Decayed raw sums of the anchor cloud (weight, first moments, second moments of predicted p and
        //cross moments with target t).
        private float _rw, _rpx, _rpy, _rtx, _rty, _rpxpx, _rpypy, _rpxpy, _rtxpx, _rtxpy, _rtypx, _rtypy;

        private void AccumulateDecayedSums(Vector2 p, Vector2 t, float weight)
        {
            _rw = _rw * Forgetting + weight;
            _rpx = _rpx * Forgetting + weight * p.x;
            _rpy = _rpy * Forgetting + weight * p.y;
            _rtx = _rtx * Forgetting + weight * t.x;
            _rty = _rty * Forgetting + weight * t.y;
            _rpxpx = _rpxpx * Forgetting + weight * p.x * p.x;
            _rpypy = _rpypy * Forgetting + weight * p.y * p.y;
            _rpxpy = _rpxpy * Forgetting + weight * p.x * p.y;
            _rtxpx = _rtxpx * Forgetting + weight * t.x * p.x;
            _rtxpy = _rtxpy * Forgetting + weight * t.x * p.y;
            _rtypx = _rtypx * Forgetting + weight * t.y * p.x;
            _rtypy = _rtypy * Forgetting + weight * t.y * p.y;
        }

        //Decayed anchor weight at which the correction reaches full strength. Below it the whole
        //correction is scaled down proportionally: with the closed-form solve, ONE accepted anchor would
        //otherwise fully determine the translation and instantly shift the gaze by up to the caps — a
        //single misleading click (user did not look, residual under the gate) must not do that.
        private const float WarmupWeight = 5f;

        //Solves the correction from the decayed sums. Translation always (warmup-ramped); the 2x2 gain
        //matrix only once enough spread anchors exist (and always softly clamped).
        private void Solve()
        {
            if (_rw <= 1e-6f) return;
            float inv = 1f / _rw;
            float mpx = _rpx * inv, mpy = _rpy * inv, mtx = _rtx * inv, mty = _rty * inv;
            //Central covariances of predicted, and predicted-vs-target
            float cxx = _rpxpx * inv - mpx * mpx;
            float cyy = _rpypy * inv - mpy * mpy;
            float cxy = _rpxpy * inv - mpx * mpy;
            float ctxx = _rtxpx * inv - mtx * mpx;
            float ctxy = _rtxpy * inv - mtx * mpy;
            float ctyx = _rtypx * inv - mty * mpx;
            float ctyy = _rtypy * inv - mty * mpy;

            //LATCHED unlock: once the affine DOFs open they stay open (until Reset). Relocking on a
            //momentarily decayed spread snapped learned gains back to identity in a single frame — a
            //visible gaze jump; with the latch, thin recent data simply leaves the previous fit in place
            //(the det guard below skips degenerate updates).
            float spread = Mathf.Sqrt(Mathf.Max(0f, cxx) + Mathf.Max(0f, cyy));
            if (AcceptedAnchors >= MinAnchorsForAffine && spread >= MinSpreadForAffine)
                AffineUnlocked = true;

            if (AffineUnlocked)
            {
                float det = cxx * cyy - cxy * cxy;
                if (Mathf.Abs(det) > 1e-8f)
                {
                    float invDet = 1f / det;
                    //Row-wise least squares: [a11 a12] = [ctxx ctxy] * inv([[cxx cxy][cxy cyy]])
                    _a11 = (ctxx * cyy - ctxy * cxy) * invDet;
                    _a12 = (ctxy * cxx - ctxx * cxy) * invDet;
                    _a21 = (ctyx * cyy - ctyy * cxy) * invDet;
                    _a22 = (ctyy * cxx - ctyx * cxy) * invDet;
                }
            }

            //Clamp gains/shears so a pathological anchor set stays survivable.
            _a11 = Mathf.Clamp(_a11, 1f - GainCap, 1f + GainCap);
            _a22 = Mathf.Clamp(_a22, 1f - GainCap, 1f + GainCap);
            _a12 = Mathf.Clamp(_a12, -GainCap, GainCap);
            _a21 = Mathf.Clamp(_a21, -GainCap, GainCap);

            //Warmup ramp: blend the whole correction toward identity while the decayed anchor weight is
            //small, so the first few anchors move the gaze gradually instead of jumping to their solve.
            float warmup = Mathf.Clamp01(_rw / WarmupWeight);
            _a11 = 1f + (_a11 - 1f) * warmup;
            _a22 = 1f + (_a22 - 1f) * warmup;
            _a12 *= warmup;
            _a21 *= warmup;

            //Translation makes the (clamped) map pass through the anchor means, then is itself ramped + capped.
            _bx = Mathf.Clamp((mtx - (_a11 * mpx + _a12 * mpy)) * warmup, -TranslationCap, TranslationCap);
            _by = Mathf.Clamp((mty - (_a21 * mpx + _a22 * mpy)) * warmup, -TranslationCap, TranslationCap);
        }

        // ---- Persistence (session warm start). The state is small and normalized, so it transfers across
        // resolutions; it should NOT transfer across recalibrations (callers Reset() on recalibrate).
        [System.Serializable]
        private class State
        {
            public float a11 = 1f, a12, a21, a22 = 1f, bx, by;
            public float rw, rpx, rpy, rtx, rty, rpxpx, rpypy, rpxpy, rtxpx, rtxpy, rtypx, rtypy;
            public int accepted;
            public bool affine;
        }

        public string SaveToJson()
        {
            var s = new State
            {
                a11 = _a11, a12 = _a12, a21 = _a21, a22 = _a22, bx = _bx, by = _by,
                rw = _rw, rpx = _rpx, rpy = _rpy, rtx = _rtx, rty = _rty,
                rpxpx = _rpxpx, rpypy = _rpypy, rpxpy = _rpxpy,
                rtxpx = _rtxpx, rtxpy = _rtxpy, rtypx = _rtypx, rtypy = _rtypy,
                accepted = AcceptedAnchors,
                affine = AffineUnlocked,
            };
            return JsonUtility.ToJson(s);
        }

        public void LoadFromJson(string json)
        {
            if (string.IsNullOrEmpty(json)) return;
            try
            {
                var s = JsonUtility.FromJson<State>(json);
                if (s == null) return;
                _a11 = s.a11; _a12 = s.a12; _a21 = s.a21; _a22 = s.a22; _bx = s.bx; _by = s.by;
                _rw = s.rw; _rpx = s.rpx; _rpy = s.rpy; _rtx = s.rtx; _rty = s.rty;
                _rpxpx = s.rpxpx; _rpypy = s.rpypy; _rpxpy = s.rpxpy;
                _rtxpx = s.rtxpx; _rtxpy = s.rtxpy; _rtypx = s.rtypx; _rtypy = s.rtypy;
                AcceptedAnchors = s.accepted;
                AffineUnlocked = s.affine;
            }
            catch (System.Exception e)
            {
                UnitEyeLog.Exception(e);
                Reset();
            }
        }
    }
}
