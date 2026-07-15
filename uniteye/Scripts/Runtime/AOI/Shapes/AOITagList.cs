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

        //Reused per-frame buffers for the xray raycast path (was a fresh RaycastHit[20] + List each call).
        private RaycastHit[] _raycastBuffer;
        private readonly List<RaycastHit> _hitListWithTag = new List<RaycastHit>();

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

            //(hit is default(RaycastHit) when nothing was hit; the old hit.Equals(null) branch was dead
            //because RaycastHit is a struct and never equals null.)
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
            //Reuse the buffers instead of allocating a RaycastHit[] + List every frame. Resize the
            //array only if maxNumberOfRaycastHits changed.
            if (_raycastBuffer == null || _raycastBuffer.Length != maxNumberOfRaycastHits)
                _raycastBuffer = new RaycastHit[maxNumberOfRaycastHits];
            _hitListWithTag.Clear();
            Ray ray = camera.ViewportPointToRay(point);

            hitAOI = false;

            //Xray RaycastNonAlloc that puts all hit targets into an array
            int hits = Physics.RaycastNonAlloc(ray, _raycastBuffer, Mathf.Infinity, layerMask);

            //Go through the hit array and check if _raycastBuffer[i] has a tag in _tagList
            for (int i = 0; i < hits; i++)
            {
                var tag = _raycastBuffer[i].collider.transform.tag;
                //If tag in _tagList
                if (_tagList.Contains(tag))
                {
                    //Add string to hitTagList and set tagFound to true
                    hitNameList.Add($"({tag}/{_raycastBuffer[i].collider.gameObject.name})");
                    hitAOI = true;
                    _hitListWithTag.Add(_raycastBuffer[i]);
                }
            }

            return _hitListWithTag;
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