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

        public Camera camera;

        public int maxNumberOfRaycastHits = 20;

        public bool xray;
        public int layerMask;
        public QueryTriggerInteraction queryTriggerInteraction;

        //Reused per-frame buffers for the xray raycast path (was a fresh RaycastHit[20] + List each call).
        private RaycastHit[] _raycastBuffer;
        private readonly List<RaycastHit> _hitListWithTag = new List<RaycastHit>();

        //Central per-frame raycast cache shared across all AOITagList / Gazeable instances querying the same
        //(camera, viewportPoint, layerMask, queryTriggerInteraction) in one frame, cutting N raycasts + N tag
        //string allocations down to 1.
        private static int s_singleFrame = -1, s_singleMask;
        private static Camera s_singleCam;
        private static Vector3 s_singlePoint;
        private static QueryTriggerInteraction s_singleQti;
        private static bool s_singleHasHit;
        private static RaycastHit s_singleHit;
        private static string s_singleTag, s_singleEntry;

        private static int s_xrayFrame = -1, s_xrayMask, s_xrayMaxHits, s_xrayCount;
        private static Camera s_xrayCam;
        private static Vector3 s_xrayPoint;
        private static QueryTriggerInteraction s_xrayQti;
        private static RaycastHit[] s_xrayHits;
        private static string[] s_xrayTags, s_xrayEntries;

        public AOITagList(string uID, bool xray = false, bool inverted = false, int layerMask = Physics.DefaultRaycastLayers, QueryTriggerInteraction queryTriggerInteraction = QueryTriggerInteraction.Ignore, bool enabled = true, bool visualized = true) : base(uID, inverted, enabled, visualized)
        {
            this.xray = xray;
            this.layerMask = layerMask;
            this.queryTriggerInteraction = queryTriggerInteraction;
        }

        public AOITagList(string uID, HashSet<string> tagList, bool xray = false, bool inverted = false, int layerMask = Physics.DefaultRaycastLayers, QueryTriggerInteraction queryTriggerInteraction = QueryTriggerInteraction.Ignore, bool enabled = true, bool visualized = true) : base(uID, inverted, enabled, visualized)
        {
            _tagList = tagList ?? new HashSet<string>();
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
        private Camera ResolveCamera()
        {
            if (camera == null)
                camera = Camera.main;
            return camera;
        }

        private RaycastHit CheckRaycast(Vector3 point, out bool hitAOI)
        {
            hitAOI = false;
            var cam = ResolveCamera();
            if (cam == null)
                return default;

            int frame = Time.frameCount;
            if (s_singleFrame != frame || !ReferenceEquals(s_singleCam, cam) || s_singleMask != layerMask ||
                s_singleQti != queryTriggerInteraction || s_singlePoint != point)
            {
                s_singleFrame = frame;
                s_singleCam = cam;
                s_singleMask = layerMask;
                s_singleQti = queryTriggerInteraction;
                s_singlePoint = point;

                Ray ray = cam.ViewportPointToRay(point);
                s_singleHasHit = Physics.Raycast(ray, out s_singleHit, Mathf.Infinity, layerMask, queryTriggerInteraction);
                if (s_singleHasHit && s_singleHit.collider != null)
                {
                    s_singleTag = s_singleHit.collider.transform.tag;
                    s_singleEntry = $"({s_singleTag}/{s_singleHit.collider.gameObject.name})";
                }
                else
                {
                    s_singleHasHit = false;
                    s_singleTag = null;
                    s_singleEntry = null;
                }
            }

            if (s_singleHasHit && s_singleHit.collider != null && _tagList.Contains(s_singleTag))
            {
                hitNameList.Add(s_singleEntry);
                hitAOI = true;
                return s_singleHit;
            }

            return s_singleHasHit ? s_singleHit : default;
        }

        /// <summary>
        /// Check RaycastNonAlloc which goes through all hit colliders and saves them to a list.
        /// </summary>
        /// <param name="point"></param>
        /// <param name="hitAOI">out bool for caller</param>
        /// <returns>List with all the found RaycastHit, is empty if nothing was found</returns>
        private List<RaycastHit> CheckRaycastXray(Vector3 point, out bool hitAOI)
        {
            _hitListWithTag.Clear();
            hitAOI = false;

            var cam = ResolveCamera();
            int maxHits = Mathf.Max(1, maxNumberOfRaycastHits);
            if (cam == null)
                return _hitListWithTag;

            int frame = Time.frameCount;
            if (s_xrayFrame != frame || !ReferenceEquals(s_xrayCam, cam) || s_xrayMask != layerMask ||
                s_xrayQti != queryTriggerInteraction || s_xrayMaxHits != maxHits || s_xrayPoint != point)
            {
                s_xrayFrame = frame;
                s_xrayCam = cam;
                s_xrayMask = layerMask;
                s_xrayQti = queryTriggerInteraction;
                s_xrayMaxHits = maxHits;
                s_xrayPoint = point;

                if (s_xrayHits == null || s_xrayHits.Length != maxHits)
                {
                    s_xrayHits = new RaycastHit[maxHits];
                    s_xrayTags = new string[maxHits];
                    s_xrayEntries = new string[maxHits];
                }

                Ray ray = cam.ViewportPointToRay(point);
                s_xrayCount = Physics.RaycastNonAlloc(ray, s_xrayHits, Mathf.Infinity, layerMask, queryTriggerInteraction);
                for (int i = 0; i < s_xrayCount; i++)
                {
                    var col = s_xrayHits[i].collider;
                    if (col != null)
                    {
                        string tag = col.transform.tag;
                        s_xrayTags[i] = tag;
                        s_xrayEntries[i] = $"({tag}/{col.gameObject.name})";
                    }
                    else
                    {
                        s_xrayTags[i] = null;
                        s_xrayEntries[i] = null;
                    }
                }
            }

            for (int i = 0; i < s_xrayCount; i++)
            {
                if (s_xrayHits[i].collider != null && s_xrayTags[i] != null && _tagList.Contains(s_xrayTags[i]))
                {
                    hitNameList.Add(s_xrayEntries[i]);
                    hitAOI = true;
                    _hitListWithTag.Add(s_xrayHits[i]);
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