using UnityEngine;
using UnitEye;
using System.Collections;
namespace UnitEye
{

    [RequireComponent(typeof(Collider))]
    public class GazeableXray : MonoBehaviour
    {
        /// <summary>
        /// Checks whether or not this game object is currently looked at.
        /// </summary>
        public bool HasGazeFocus => IsFocused();

        [Tooltip("If enabled, the Raycast that checks if the collider is being looked at will go through all colliders, if disabled the Raycast will stop at the first collider.")]
        public bool xray = true;

        [Tooltip("If enabled, all objects with matching tags will influence HasGazeFocus boolean even if they don't have a Gazeable component.")]
        public bool checkAllObjectsWithMatchingTag = false;

        private AOITagList _tagList;
        private bool _isQuitting;
        private bool _hasGazeFocus;
        private string _oldTag;
        private static int s_current = 0;

        void Awake()
        {
            EnsureInitialized();
        }

        void Start()
        {
            EnsureInitialized();
        }

        private void EnsureInitialized()
        {
            if (_tagList != null) return;
            if (gameObject.CompareTag("Untagged"))
                Debug.Log("Gazeable GameObject must have a tag different than \"Untagged\"");

            _oldTag = gameObject.tag;
            _tagList = new AOITagList(gameObject.name + "Xray" + s_current.ToString(), xray);
            s_current++;

            _tagList.AddTag(_oldTag);
        }

        void Update()
        {
            if (!string.IsNullOrEmpty(_oldTag) && !gameObject.CompareTag(_oldTag))
            {
                _tagList.RemoveTag(_oldTag);
                _oldTag = gameObject.tag;
                _tagList.AddTag(_oldTag);
            }

            //Make AOI aware of xray change
            if (_tagList != null && xray != _tagList.xray)
                _tagList.xray = xray;
        }

        void OnEnable()
        {
            EnsureInitialized();
            //Add AOI in the next frame to allow Gaze component to initialize
            StartCoroutine(AddAOI());
        }

        void OnDisable()
        {
            if (!_isQuitting && _tagList != null)
                UnitEyeAPI.TryGetAOIManagerInstance()?.RemoveAOI(_tagList);
        }

        private void OnApplicationQuit()
        {
            //bool flag to avoid running OnDisable() when the application is quitting, which would
            // result in a NullReferenceException because OnDisable() runs last in the execution order
            _isQuitting = true;
        }

        /// <summary>
        /// Helper function to check if this gameObject is focused.
        /// </summary>
        /// <returns>true if focused, false if not focused</returns>
        private bool IsFocused()
        {
            _hasGazeFocus = false;

            if (_tagList == null || !_tagList.focused || !this.enabled)
            {
                return false;
            }

            if (!checkAllObjectsWithMatchingTag)
            {
                //Check for gameObject equality
                bool found = false;

                if (_tagList.xray)
                {
                    //If xray is enabled, go through all raycast hits and check if one of them matches this gameObject
                    foreach (var raycasthit in _tagList.hitRaycastList)
                    {
                        if (raycasthit.collider?.gameObject == gameObject)
                            found = true;
                    }
                }
                else
                {
                    //Else check if the one raycast hit matches this gameObject
                    if (_tagList.hitRaycast.collider?.gameObject == gameObject)
                        found = true;
                }

                _hasGazeFocus = found;
            }
            else
            {
                //If xray is true check if atleast one hit in AOITagList
                _hasGazeFocus = _tagList.xray ? _tagList.hitRaycastList.Count > 0 : true;
            }

            return _hasGazeFocus;
        }

        IEnumerator AddAOI()
        {
            //Yield until next frame to allow Gaze component to be initialized
            yield return 0;

            UnitEyeAPI.GetAOIManagerInstance()?.AddAOI(_tagList);
        }
    }
}
