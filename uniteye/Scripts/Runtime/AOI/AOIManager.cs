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
        public void CheckAOIList(Vector2 point, List<string> list)
        {
            list.Clear();

            foreach (AOI aoi in _aoiList)
            {
                if (aoi.enabled && aoi.CheckAOI(point))
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