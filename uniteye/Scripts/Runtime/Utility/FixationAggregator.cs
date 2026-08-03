using System.Collections.Generic;
using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// Runtime I-DT fixation detector + centroid aggregator for the calibrated gaze stream.
    ///
    /// Per-frame gaze is jitter-dominated; during a fixation (~200ms–1s, several camera samples) the
    /// noise is largely independent, so the fixation CENTROID cuts the precision component by ~sqrt(N).
    /// AOI hit logging therefore consumes the running centroid of the current fixation instead of the
    /// instantaneous sample (commercial trackers quote exactly this fixation-level signal), while the
    /// visible cursor keeps the responsive One-Euro output — the two consumers deliberately differ.
    ///
    /// Detection is dispersion-based (I-DT): the sample window of the last <see cref="windowSeconds"/>
    /// counts as one fixation while its bounding-box diagonal stays under a threshold expressed as a
    /// fraction of the screen diagonal (the same normalization the calibration's fixation gate uses,
    /// resolution-independent). A saccade (dispersion break) ends the fixation and starts a new window.
    /// </summary>
    public class FixationAggregator
    {
        //Same defaults as the calibration's capture gate (IsFixationStable): 0.035 of the diagonal at
        //~300ms. The runtime window is slightly longer so brief tracker glitches don't split fixations.
        public float windowSeconds = 0.35f;
        public float dispersionFraction = 0.035f;
        public int minimumSamples = 3;

        private readonly List<Vector2> _points = new List<Vector2>(64);
        private readonly List<double> _times = new List<double>(64);
        private Vector2 _sum;
        private Vector2 _fixationCentroid;
        private bool _inFixation;
        private double _fixationStartTime;

        /// <summary>True while the current window classifies as a fixation.</summary>
        public bool InFixation => _inFixation;
        /// <summary>Duration of the current fixation in seconds (0 when not fixating).</summary>
        public double FixationDuration { get; private set; }
        /// <summary>Number of samples in the current fixation window.</summary>
        public int SampleCount => _points.Count;

        /// <summary>
        /// Adds a gaze sample (pixels) at <paramref name="time"/> and returns the point AOI consumers
        /// should use: the running fixation centroid while fixating, the raw sample otherwise (during a
        /// saccade there is no meaningful aggregate — and AOI logging downstream can choose to ignore
        /// saccade samples entirely via <see cref="InFixation"/>).
        /// </summary>
        public Vector2 Add(Vector2 gazePixels, double time)
        {
            //Drop samples older than the window (the window slides; a stable fixation keeps ALL its
            //samples via the fixation branch below, so long fixations still average over their full span).
            if (!_inFixation)
            {
                while (_times.Count > 0 && time - _times[0] > windowSeconds)
                {
                    _sum -= _points[0];
                    _points.RemoveAt(0);
                    _times.RemoveAt(0);
                }
            }

            _points.Add(gazePixels);
            _times.Add(time);
            _sum += gazePixels;

            //Dispersion of the current window against the screen-diagonal fraction.
            float diagonal = Mathf.Sqrt((float)Screen.width * Screen.width + (float)Screen.height * Screen.height);
            float minX = float.MaxValue, minY = float.MaxValue, maxX = float.MinValue, maxY = float.MinValue;
            for (int i = 0; i < _points.Count; i++)
            {
                var p = _points[i];
                if (p.x < minX) minX = p.x;
                if (p.x > maxX) maxX = p.x;
                if (p.y < minY) minY = p.y;
                if (p.y > maxY) maxY = p.y;
            }
            float dispersion = new Vector2(maxX - minX, maxY - minY).magnitude;
            bool stable = _points.Count >= minimumSamples && dispersion <= dispersionFraction * diagonal;

            if (stable)
            {
                if (!_inFixation)
                {
                    _inFixation = true;
                    _fixationStartTime = _times[0];
                }
                _fixationCentroid = _sum / _points.Count;
                FixationDuration = time - _fixationStartTime;
                return _fixationCentroid;
            }

            if (_inFixation)
            {
                //Dispersion broke: the fixation ended. Restart the window from the samples that broke it
                //(the saccade tail), so the next fixation doesn't inherit the old cluster.
                _inFixation = false;
                FixationDuration = 0;
                var last = _points[_points.Count - 1];
                var lastT = _times[_times.Count - 1];
                Reset();
                _points.Add(last);
                _times.Add(lastT);
                _sum = last;
            }
            return gazePixels;
        }

        public void Reset()
        {
            _points.Clear();
            _times.Clear();
            _sum = Vector2.zero;
            _inFixation = false;
            FixationDuration = 0;
        }
    }
}
