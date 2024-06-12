using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

namespace UnitEye
{
    public class AOITagList : AOI
    {
        private HashSet<string> _tagList = new HashSet<string>();
        public List<string> hitNameList = new List<string>();

        public RaycastHit hitRaycast;
        public List<RaycastHit> hitRaycastList = new List<RaycastHit>();

        private Vector3 _pointVector;

        public Camera camera = Camera.main;

        public int maxNumberOfRaycastHits = 20;

        public bool xray;
        public int layerMask;
        public QueryTriggerInteraction queryTriggerInteraction;

        public AOITagList(string uID, bool xray = false, bool inverted = false, int layerMask = Physics.DefaultRaycastLayers, QueryTriggerInteraction queryTriggerInteraction = QueryTriggerInteraction.Ignore, bool enabled = true, bool visualized = true) : base(uID, inverted, enabled, visualized)
        {
            this.xray = xray;
            this.layerMask = layerMask;
            this.queryTriggerInteraction = queryTriggerInteraction;
        }

        public AOITagList(string uID, HashSet<string> tagList, bool xray = false, bool inverted = false, int layerMask = Physics.DefaultRaycastLayers, QueryTriggerInteraction queryTriggerInteraction = QueryTriggerInteraction.Ignore, bool enabled = true, bool visualized = true) : base(uID, inverted, enabled, visualized)
        {
            _tagList = tagList;
            this.xray = xray;
            this.layerMask = layerMask;
            this.queryTriggerInteraction = queryTriggerInteraction;
        }

        /// <summary>
        /// Add tag to be checked for. Must not be in the _taglist already.
        /// </summary>
        /// <param name="tag"></param>
        public void AddTag(string tag)
        {
            //Debug.Log($"Adding Tag: {tag}");
            _tagList.Add(tag);
        }

        /// <summary>
        /// Remove tag from _taglist.
        /// </summary>
        /// <param name="tag"></param>
        public void RemoveTag(string tag)
        {
            _tagList.Remove(tag);
        }

        /// <summary>
        /// Check if Raycast hits a collider where the object has a tag from _tagList.
        /// </summary>
        /// <param name="point"></param>
        /// <returns>True if an object with a tag from _tagList was hit. If this.inverted is true the return value is inverted.</returns>
        public override bool CheckAOI(Vector2 point)
        {
            bool aoiFound = false;

            hitNameList.Clear();
            //Convert point to Vector3
            _pointVector = new Vector3(point.x, 1 - point.y, 0f);

            //If the ray should go through all objects call CheckRaycastXray(), if not call CheckRaycast()
            //aoiFound is true when a match in _tagList has been found
            //also sets public fields to match found RaycastHit or List<RaycastHit> for xray
            if (this.xray) 
                hitRaycastList = CheckRaycastXray(_pointVector, out aoiFound);
            else
                hitRaycast = CheckRaycast(_pointVector, out aoiFound);

            //Invert if AOI is inverted
            return this.inverted ? !aoiFound : aoiFound;
        }

        /// <summary>
        /// Check normal Raycast that stops after the first collision.
        /// </summary>
        /// <param name="point"></param>
        /// <param name="hitAOI">out bool for caller</param>
        /// <returns>RaycastHit when a hit with tag was found, null if nothing was hit</returns>
        private RaycastHit CheckRaycast(Vector3 point, out bool hitAOI)
        {
            RaycastHit hit;
            Ray ray = camera.ViewportPointToRay(point);

            hitAOI = false;

            //Normal Raycast
            if (Physics.Raycast(ray, out hit, Mathf.Infinity, layerMask))
            {
                var tag = hit.collider.transform.tag;
                //If tag in _tagList
                if (_tagList.Contains(tag))
                {
                    //Add string to hitTagList and return true
                    hitNameList.Add($"({tag}/{hit.collider.gameObject.name})");
                    hitAOI = true;
                }
            }

            if (hit.Equals(null))
                return new RaycastHit();
            else
                return hit;
        }

        /// <summary>
        /// Check RaycastNonAlloc which goes through all hit colliders and saves them to a list.
        /// </summary>
        /// <param name="point"></param>
        /// <param name="hitAOI">out bool for caller</param>
        /// <returns>List with all the found RaycastHit, is empty if nothing was found</returns>
        private List<RaycastHit> CheckRaycastXray(Vector3 point, out bool hitAOI)
        {
            RaycastHit[] hitList = new RaycastHit[maxNumberOfRaycastHits];
            List<RaycastHit> hitListWithTag = new List<RaycastHit>();
            Ray ray = camera.ViewportPointToRay(point);

            hitAOI = false;

            //Xray RaycastNonAlloc that puts all hit targets into an array
            int hits = Physics.RaycastNonAlloc(ray, hitList, Mathf.Infinity, layerMask);

            //Go through hitList array and check if hitList[i] has a tag in _tagList
            for (int i = 0; i < hits; i++)
            {
                var tag = hitList[i].collider.transform.tag;
                //If tag in _tagList
                if (_tagList.Contains(tag))
                {
                    //Add string to hitTagList and set tagFound to true
                    hitNameList.Add($"({tag}/{hitList[i].collider.gameObject.name})");
                    hitAOI = true;
                    hitListWithTag.Add(hitList[i]);
                }
            }

            return hitListWithTag;
        }

        /// <summary>
        /// Empty Visualize() to avoid compiler errors.
        /// </summary>
        /// <param name="color"></param>
        public override void Visualize(Color color)
        {
        }
    }
}