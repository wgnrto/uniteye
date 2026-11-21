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
        private double _fixationSumX;
        private double _fixationSumY;
        private int _fixationCount;
        private Vector2 _fixationCentroid;
        private bool _inFixation;
        private double _fixationStartTime;

        /// <summary>True while the current window classifies as a fixation.</summary>
        public bool InFixation => _inFixation;
        /// <summary>Duration of the current fixation in seconds (0 when not fixating).</summary>
        public double FixationDuration { get; private set; }
        /// <summary>Number of samples in the current fixation (or candidate window when not fixating).</summary>
        public int SampleCount => _inFixation ? _fixationCount : _points.Count;

        /// <summary>
        /// Adds a gaze sample (pixels) at <paramref name="time"/> and returns the point AOI consumers
        /// should use: the running fixation centroid while fixating, the raw sample otherwise (during a
        /// saccade there is no meaningful aggregate — and AOI logging downstream can choose to ignore
        /// saccade samples entirely via <see cref="InFixation"/>).
        /// </summary>
        public Vector2 Add(Vector2 gazePixels, double time)
        {
            if (float.IsNaN(gazePixels.x) || float.IsNaN(gazePixels.y) ||
                float.IsInfinity(gazePixels.x) || float.IsInfinity(gazePixels.y) ||
                double.IsNaN(time) || double.IsInfinity(time))
            {
                Reset();
                return gazePixels;
            }

            //A face-loss gap longer than windowSeconds (or clock reset) cannot belong to the same fixation.
            if (_times.Count > 0)
            {
                double dt = time - _times[_times.Count - 1];
                if (dt > windowSeconds || dt < 0.0)
                    Reset();
            }

            //Keep _points bounded to the sliding windowSeconds tail so dispersion checks stay O(W) and
            //Gaussian extreme-value range growth never self-breaks a long steady stare.
            while (_times.Count > 0 && time - _times[0] > windowSeconds)
            {
                _points.RemoveAt(0);
                _times.RemoveAt(0);
            }

            _points.Add(gazePixels);
            _times.Add(time);

            float diagonal = Mathf.Sqrt((float)Screen.width * Screen.width + (float)Screen.height * Screen.height);
            if (diagonal <= 0f) diagonal = 1000f;
            float threshold = dispersionFraction * diagonal;
            float thresholdSq = threshold * threshold;
            int minSamples = Mathf.Max(1, minimumSamples);

            if (_inFixation)
            {
                float minX = float.MaxValue, minY = float.MaxValue, maxX = float.MinValue, maxY = float.MinValue;
                Vector2 winSum = Vector2.zero;
                for (int i = 0; i < _points.Count; i++)
                {
                    var p = _points[i];
                    if (p.x < minX) minX = p.x;
                    if (p.x > maxX) maxX = p.x;
                    if (p.y < minY) minY = p.y;
                    if (p.y > maxY) maxY = p.y;
                    winSum += p;
                }
                float dx = maxX - minX, dy = maxY - minY;
                Vector2 winMean = winSum / _points.Count;
                bool stillStable = (dx * dx + dy * dy <= thresholdSq)
                    && (gazePixels - _fixationCentroid).sqrMagnitude <= thresholdSq
                    && (winMean - _fixationCentroid).sqrMagnitude <= thresholdSq * 0.5625f;

                if (stillStable)
                {
                    _fixationSumX += gazePixels.x;
                    _fixationSumY += gazePixels.y;
                    _fixationCount++;
                    _fixationCentroid = new Vector2(
                        (float)(_fixationSumX / _fixationCount),
                        (float)(_fixationSumY / _fixationCount));
                    FixationDuration = time - _fixationStartTime;
                    return _fixationCentroid;
                }

                //Dispersion broke: end the fixation and keep only the latest sample so the next landing
                //cluster can form immediately without inheriting pre-landing points.
                Reset();
                _points.Add(gazePixels);
                _times.Add(time);
                return gazePixels;
            }

            //Not yet in a fixation: find the longest trailing cluster within threshold so a single prior
            //saccade sample in the buffer cannot blind fixation onset for a full windowSeconds.
            if (_points.Count >= minSamples)
            {
                float minX = float.MaxValue, minY = float.MaxValue, maxX = float.MinValue, maxY = float.MinValue;
                int bestStart = _points.Count;
                for (int i = _points.Count - 1; i >= 0; i--)
                {
                    var p = _points[i];
                    if (p.x < minX) minX = p.x;
                    if (p.x > maxX) maxX = p.x;
                    if (p.y < minY) minY = p.y;
                    if (p.y > maxY) maxY = p.y;
                    float dx = maxX - minX, dy = maxY - minY;
                    if (dx * dx + dy * dy > thresholdSq)
                        break;
                    bestStart = i;
                }

                int clusterCount = _points.Count - bestStart;
                if (clusterCount >= minSamples)
                {
                    if (bestStart > 0)
                    {
                        _points.RemoveRange(0, bestStart);
                        _times.RemoveRange(0, bestStart);
                    }
                    _inFixation = true;
                    _fixationStartTime = _times[0];
                    _fixationSumX = 0.0;
                    _fixationSumY = 0.0;
                    for (int i = 0; i < _points.Count; i++)
                    {
                        _fixationSumX += _points[i].x;
                        _fixationSumY += _points[i].y;
                    }
                    _fixationCount = _points.Count;
                    _fixationCentroid = new Vector2(
                        (float)(_fixationSumX / _fixationCount),
                        (float)(_fixationSumY / _fixationCount));
                    FixationDuration = time - _fixationStartTime;
                    return _fixationCentroid;
                }
            }

            return gazePixels;
        }

        public void Reset()
        {
            _points.Clear();
            _times.Clear();
            _fixationSumX = 0.0;
            _fixationSumY = 0.0;
            _fixationCount = 0;
            _inFixation = false;
            FixationDuration = 0;
        }
    }
}
