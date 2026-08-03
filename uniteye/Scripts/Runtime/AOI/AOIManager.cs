using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// This class is responsible for handling all the AOIs to be tracked.
    /// </summary>
    public class AOIManager
    {
        private List<AOI> _aoiList = new List<AOI>();
        private AOIVisualizer _aoiVisualizer;

        private bool _visualized;

        private Color _currentColor = Color.red;

        /// <summary>
        /// Add aoi to _aoiList.
        /// </summary>
        /// <param name="aoi"></param>
        public void AddAOI(AOI aoi)
        {
            //Debug.Log($"Adding AOI: {aoi.uID} | {aoi.inverted}");
            bool found = false;

            if (_aoiList.Count > 0)
            {
                foreach (AOI aoiL in _aoiList)
                {
                    if (aoiL.uID == aoi.uID) found = true;
                }
            }

            if (found) Debug.Log("AOI with same unique ID already in list!");
            else _aoiList.Add(aoi);
        }
        /// <summary>
        /// Remove aoi from _aoiList.
        /// </summary>
        /// <param name="aoi"></param>
        public void RemoveAOI(AOI aoi)
        {
            _aoiList.Remove(aoi);
            if (aoi != null) _dwell.Remove(aoi.uID);
        }
        /// <summary>
        /// Remove AOI from _aoiList by uID.
        /// </summary>
        /// <param name="uID"></param>
        public void RemoveAOI(string uID)
        {
            //RemoveAll instead of foreach+Remove: mutating the list inside a foreach over it throws
            //InvalidOperationException the moment a match is found (i.e. exactly in the success case).
            _aoiList.RemoveAll(aoi => aoi.uID == uID);
            _dwell.Remove(uID);
        }

        /// <summary>
        /// Get AOI from _aoiList by uID.
        /// </summary>
        /// <param name="uID"></param>
        /// <returns>AOI if found, null if not.</returns>
        public AOI GetAOIFromList(string uID)
        {
            foreach (AOI aoi in _aoiList)
            {
                if (aoi.uID == uID) return aoi;
            }

            return null;
        }

        /// <summary>
        /// Get all AOIs.
        /// </summary>
        /// <returns>Returns the list of all AOIs.</returns>
        public List<AOI> GetAOIs()
        {
            return _aoiList;
        }

        /// <summary>
        /// Check all AOI in _aoiList for point inclusion.
        /// </summary>
        /// <param name="point"></param>
        /// <returns>string list with all AOI.uID that contain point.</returns>
        public List<string> CheckAOIList(Vector2 point)
        {
            List<string> list = new List<string>();
            CheckAOIList(point, list);
            return list;
        }

        /// <summary>
        /// Allocation-free variant: clears and refills the caller-provided list instead of allocating a
        /// new one per call (the gaze pipeline runs this every frame). Callers that RETAIN the result
        /// (e.g. queueing it into a CSVData, which serializes later) must store their own copy — this
        /// same list instance is refilled on the next call.
        /// </summary>
        //---- Hysteresis + minimum dwell (the GazeGridQuantizer pattern generalized to all AOIs). A hit
        //only REGISTERS once the gaze has been inside for minimumDwellSeconds (human fixations are rarely
        //<100ms — shorter "hits" are noise), and an AOI that DID register stays active until the gaze has
        //been outside it for exitHysteresisSeconds (kills boundary flicker on noisy signals). An AOI that
        //never satisfied the dwell gets no hysteresis — a saccade sweeping through must not register a
        //phantom hit on the way out.
        [System.NonSerialized] public float minimumDwellSeconds = 0.1f;
        [System.NonSerialized] public float exitHysteresisSeconds = 0.12f;
        [System.NonSerialized] public bool dwellFiltering = true;
        private struct DwellState { public float insideSince, lastInside; public bool registered; }
        private readonly Dictionary<string, DwellState> _dwell = new Dictionary<string, DwellState>();

        public void CheckAOIList(Vector2 point, List<string> list)
        {
            list.Clear();
            float now = Time.unscaledTime;

            foreach (AOI aoi in _aoiList)
            {
                if (!aoi.enabled)
                {
                    //No stale dwell state may survive a disable — re-enabling must start a fresh dwell.
                    _dwell.Remove(aoi.uID);
                    if (aoi.focused) aoi.focused = false;
                    continue;
                }

                bool rawInside = aoi.CheckAOIWithMargin(point);
                bool inside = rawInside;
                //Raycast-backed AOIs (AOITagList) refresh their hit list inside CheckAOI — extending them
                //through hysteresis would report stale/empty object lists ("uID hit:" rows, and consumers
                //indexing the hit list would throw), so they get the dwell gate but no exit extension.
                bool extendable = !(aoi is AOITagList);
                if (dwellFiltering)
                {
                    _dwell.TryGetValue(aoi.uID, out var state);
                    if (rawInside)
                    {
                        if (state.insideSince <= 0f) state.insideSince = now;
                        state.lastInside = now;
                        inside = now - state.insideSince >= minimumDwellSeconds;
                        if (inside) state.registered = true;
                        _dwell[aoi.uID] = state;
                    }
                    else if (extendable && state.registered && now - state.lastInside <= exitHysteresisSeconds)
                    {
                        //Exit hysteresis: only an AOI that actually REGISTERED stays briefly active.
                        inside = true;
                        _dwell[aoi.uID] = state;
                    }
                    else
                    {
                        _dwell.Remove(aoi.uID);
                        inside = false;
                    }
                }

                if (inside)
                {
                    //Debug.Log($"User looking at AOI: {aoi.uID}");
                    //If aoi is AOITagList add hitTagList to string
                    if (aoi is AOITagList)
                    {
                        AOITagList aoit = (AOITagList)aoi;
                        list.Add($"{aoi.uID} hit: {string.Join(", ", aoit.hitNameList)}");
                    }
                    else
                    {
                        list.Add(aoi.uID);
                    }

                    //Set aoi focused to true, means it is being looked at
                    if (!aoi.focused) aoi.focused = true;
                }
                else
                {
                    //Set aoi focused to false, means it is not being looked at
                    if (aoi.focused) aoi.focused = false;
                }
            }
        }

        /// <summary>
        /// Probabilistic hit test: fills (uID, probability) for every enabled AOI whose hit probability
        /// under the given error ellipse exceeds <paramref name="minimumProbability"/>, sorted descending.
        /// This is what the CSV logger should record alongside (or instead of) boolean hits: with a
        /// ~2cm-sigma tracker, border fixations are genuinely ambiguous, and calibrated probabilities keep
        /// downstream dwell statistics honest where booleans manufacture certainty. Does not touch
        /// focused/hysteresis state (the boolean path owns interaction semantics).
        /// </summary>
        public void CheckAOIProbabilities(Vector2 mean, float covXX, float covXY, float covYY,
            List<(string uID, float probability)> results, float minimumProbability = 0.05f)
        {
            results.Clear();
            foreach (AOI aoi in _aoiList)
            {
                if (!aoi.enabled) continue;
                //The offscreen/inverted catch-all AOIs are not meaningful probability targets.
                if (aoi.inverted) continue;
                //Raycast-backed AOIs would fire 32 physics raycasts per probability — boolean-only there.
                if (aoi is AOITagList) continue;
                float p = aoi.HitProbability(mean, covXX, covXY, covYY);
                if (p >= minimumProbability)
                    results.Add((aoi.uID, p));
            }
            results.Sort((a, b) => b.probability.CompareTo(a.probability));
        }

        /// <summary>
        /// Try to attach an AOIVisualizer to main camera.
        /// </summary>
        public void EnableVisualize()
        {
            if (_visualized) return;
            if (!AttachVisualizer()) return;
            _visualized = true;
        }

        /// <summary>
        /// Destroy AOIVisualizer in main camera.
        /// </summary>
        public void DisableVisualize()
        {
            _aoiVisualizer = Camera.main?.gameObject?.GetComponent<AOIVisualizer>();
            if (_aoiVisualizer == null) return;

            Object.Destroy(_aoiVisualizer);
            _visualized = false;
        }

        /// <summary>
        /// Set visualized to true for every AOI in _aoiList, enabling it from being visualized.
        /// </summary>
        public void VisualizeAllAOIInList()
        {
            foreach (AOI aoi in _aoiList)
            {
                aoi.visualized = true;
            }
        }
        /// <summary>
        /// Set visualized to false for every AOI in _aoiList, disabling it from being visualized.
        /// </summary>
        public void UnvisualizeAllAOIInList()
        {
            foreach (AOI aoi in _aoiList)
            {
                aoi.visualized = false;
            }
        }

        /// <summary>
        /// Go through all AOIs in _aoiList and Visualize() each if enabled and visualized.
        /// </summary>
        public void VisualizeAOIList()
        {
            //Initial color seed for DeterministicRandomColor()
            _currentColor = Color.red;
            foreach (AOI aoi in _aoiList)
            {
                if (aoi.enabled && aoi.visualized)
                {
                    aoi.Visualize(DeterministicRandomColor());
                }
            }
        }

        /// <summary>
        /// Attach AOIVisualizer to main camera in scene.
        /// </summary>
        /// <returns>True if successfully attached.</returns>
        private bool AttachVisualizer()
        {
            _aoiVisualizer = Camera.main.gameObject.AddComponent<AOIVisualizer>();
            if (_aoiVisualizer == null) return false;

            _aoiVisualizer.aoiManager = this;
            return true;
        }

        /// <summary>
        /// Pseudo random but still deterministic color.
        /// </summary>
        /// <returns>Deterministic color shifted by changing HSV hue value.</returns>
        private Color DeterministicRandomColor()
        {
            float H, S, V;

            Color lastColor = _currentColor;
            Color.RGBToHSV(_currentColor, out H, out S, out V);

            //Debug.Log($"H {H} | S {S} | V {V}");
            H = (H + 0.069f) % 1.0f;

            _currentColor = Color.HSVToRGB(H, S, V);
            return lastColor;
        }
    }
}