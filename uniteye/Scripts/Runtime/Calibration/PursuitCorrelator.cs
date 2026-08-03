using System.Collections.Generic;
using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// Correlation-gated smooth-pursuit anchor source (Pursuit Calibration, Pfeuffer/Vidal UIST'13):
    /// when the game registers a MOVING object the player plausibly tracks (a projectile, a reward flying
    /// to the HUD), a sliding-window Pearson correlation between the gaze trajectory and the object
    /// trajectory decides — per axis — whether the player is really pursuing it. Only windows above the
    /// correlation threshold emit drift anchors; a pursuit segment then yields dense labeled samples along
    /// the path, including screen regions clicks never visit (corners!).
    ///
    /// The eye trails a moving target by ~100ms (the same pursuit lag the calibration sweeps compensate),
    /// so gaze at time t is paired with the object's position at t - lag.
    /// </summary>
    public class PursuitCorrelator
    {
        //Window and gate per the pursuit-selection literature: 0.75-1s windows, r ≈ 0.75+ on BOTH axes.
        public float windowSeconds = 0.8f;
        public float correlationThreshold = 0.75f;
        public float pursuitLagSeconds = 0.1f;
        //An axis with almost no motion has a meaningless correlation; it passes automatically when the
        //OTHER axis moves enough and correlates (a horizontally flying object shouldn't fail on flat y).
        public float minimumAxisTravel = 0.03f;   // normalized units over the window

        private struct Sample { public double t; public Vector2 gaze, target; }
        private readonly List<Sample> _window = new List<Sample>(64);
        //Object position ring for the lag pairing.
        private readonly List<(double t, Vector2 p)> _targetTrail = new List<(double, Vector2)>(64);
        private double _lastGazeTime = double.NegativeInfinity;

        /// <summary>
        /// Feeds one frame: the current gaze sample with its CAPTURE time, and the tracked object's
        /// current position with ITS OWN (render) time — the two live on different clocks: the gaze
        /// lags the render clock by the full pipeline latency (~100-300ms), so timestamping the object
        /// with the gaze's capture time would shift the lag pairing by that whole latency (inverting the
        /// intended ~100ms pursuit-lag compensation and displacing every anchor along the motion path).
        /// Returns true when the window certifies pursuit — the caller should then anchor
        /// <paramref name="pairedGaze"/> to <paramref name="pairedTarget"/>.
        /// Repeated calls with the same gaze capture time (render frames without a fresh camera sample)
        /// only refresh the target trail — they add nothing to the correlation window and never certify,
        /// so a certified pursuit emits ONE anchor per camera sample, not one per render frame.
        /// </summary>
        public bool Feed(Vector2 gaze, double gazeTime, Vector2 targetPosition, double targetTime,
            out Vector2 pairedGaze, out Vector2 pairedTarget)
        {
            pairedGaze = gaze;
            pairedTarget = targetPosition;

            //The object trail is stamped with the OBJECT's clock.
            _targetTrail.Add((targetTime, targetPosition));
            while (_targetTrail.Count > 1 && targetTime - _targetTrail[0].t > 2.0)
                _targetTrail.RemoveAt(0);

            //One correlation sample per fresh gaze sample.
            if (gazeTime <= _lastGazeTime)
                return false;
            _lastGazeTime = gazeTime;

            //Lag pairing on the shared wall clock: the eye at capture time gazeTime was following the
            //object as it was pursuitLagSeconds BEFORE that moment.
            double lagTime = gazeTime - pursuitLagSeconds;
            var lagged = _targetTrail[0].p;
            for (int i = _targetTrail.Count - 1; i >= 0; i--)
                if (_targetTrail[i].t <= lagTime) { lagged = _targetTrail[i].p; break; }
            pairedTarget = lagged;

            _window.Add(new Sample { t = gazeTime, gaze = gaze, target = lagged });
            while (_window.Count > 0 && gazeTime - _window[0].t > windowSeconds)
                _window.RemoveAt(0);
            if (_window.Count < 8)
                return false;

            //Per-axis Pearson r between gaze and (lagged) target trajectories.
            float rx = Correlation(true, out float travelX);
            float ry = Correlation(false, out float travelY);
            bool xOk = travelX < minimumAxisTravel || rx >= correlationThreshold;
            bool yOk = travelY < minimumAxisTravel || ry >= correlationThreshold;
            //At least one axis must actually move AND correlate — two flat axes = not a pursuit at all.
            bool anyMoving = (travelX >= minimumAxisTravel && rx >= correlationThreshold) ||
                             (travelY >= minimumAxisTravel && ry >= correlationThreshold);
            return xOk && yOk && anyMoving;
        }

        public void Reset()
        {
            _window.Clear();
            _targetTrail.Clear();
            _lastGazeTime = double.NegativeInfinity;
        }

        private float Correlation(bool xAxis, out float targetTravel)
        {
            int n = _window.Count;
            float mg = 0f, mt = 0f;
            float minT = float.MaxValue, maxT = float.MinValue;
            for (int i = 0; i < n; i++)
            {
                float g = xAxis ? _window[i].gaze.x : _window[i].gaze.y;
                float t = xAxis ? _window[i].target.x : _window[i].target.y;
                mg += g; mt += t;
                if (t < minT) minT = t;
                if (t > maxT) maxT = t;
            }
            mg /= n; mt /= n;
            targetTravel = maxT - minT;

            float sgg = 0f, stt = 0f, sgt = 0f;
            for (int i = 0; i < n; i++)
            {
                float g = (xAxis ? _window[i].gaze.x : _window[i].gaze.y) - mg;
                float t = (xAxis ? _window[i].target.x : _window[i].target.y) - mt;
                sgg += g * g; stt += t * t; sgt += g * t;
            }
            float denom = Mathf.Sqrt(sgg * stt);
            return denom > 1e-9f ? sgt / denom : 0f;
        }
    }
}
