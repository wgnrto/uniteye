using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// This class combines multiple AOIs
    /// </summary>
    public class AOICombined : AOI
    {
        private List<AOI> _aoiList = new List<AOI>();

        public AOICombined(string uID, bool inverted = false, bool enabled = true, bool visualized = true) : base(uID, inverted, enabled, visualized)
        {
            
        }

        public AOICombined(string uID, List<AOI> aoiList, bool inverted = false, bool enabled = true, bool visualized = true) : base(uID, inverted, enabled, visualized)
        {
            this._aoiList = aoiList ?? new List<AOI>();
        }

        /// <summary>
        /// Add AOI to AOICombined.
        /// </summary>
        /// <param name="aoi"></param>
        public void AddAOI(AOI aoi)
        {
            if (aoi != null) _aoiList.Add(aoi);
        }

        /// <summary>
        /// Remove AOI from AOICombined.
        /// </summary>
        /// <param name="aoi"></param>
        public void RemoveAOI(AOI aoi)
        {
            _aoiList.Remove(aoi);
        }

        /// <summary>
        /// Remove all AOIs with uID from AOICombined.
        /// </summary>
        /// <param name="aoi"></param>
        public void RemoveAOI(string uID)
        {
            //RemoveAll instead of foreach+Remove, which throws InvalidOperationException on a match.
            _aoiList.RemoveAll(aoi => aoi != null && aoi.uID == uID);
        }

        /// <summary>
        /// Check if point is in any of the AOI in _aoiList.
        /// </summary>
        /// <param name="point"></param>
        /// <returns>True if point is in the AOICombined, false if not. If this.inverted is true the return value is inverted.</returns>
        public override bool CheckAOI(Vector2 point)
        {
            bool aoiFound = false;

            foreach (AOI aoi in _aoiList)
            {
                if (aoi != null && aoi.enabled && aoi.CheckAOIWithMargin(point))
                {
                    aoiFound = true;
                    break;
                }
            }

            return this.inverted ? !aoiFound : aoiFound;
        }

        /// <summary>
        /// Visualize all AOI in the _aoiList.
        /// </summary>
        /// <param name="color"></param>
        public override void Visualize(Color color)
        {
            foreach (AOI aoi in _aoiList)
            {
                if (aoi != null && aoi.enabled && aoi.visualized)
                    aoi.Visualize(color);
            }
        }
    }
}