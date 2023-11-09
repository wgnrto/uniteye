using UnityEngine;
using UnitEye;
using System.Collections;

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

    void Start()
    {
        if (gameObject.tag == "Untagged")
        {
            Debug.Log("Gazeable GameObject must have a tag different than \"Untagged\"");
        }

        _oldTag = gameObject.tag;
        _tagList = new AOITagList(gameObject.name + "Xray" + s_current.ToString(), xray);
        s_current++;

        _tagList.AddTag(gameObject.tag);
    }

    void Update()
    {
        //Make AOI aware of tag change
        if (_oldTag != gameObject.tag)
        {
            _tagList.RemoveTag(_oldTag);
            _tagList.AddTag(gameObject.tag);
            _oldTag = gameObject.tag;
        }

        //Make AOI aware of xray change
        if (xray != _tagList.xray)
            _tagList.xray = xray;
    }

    void OnEnable()
    {
        //Add AOI in the next frame to allow Gaze component to initialize
        StartCoroutine(AddAOI());
    }

    void OnDisable()
    {
        //Only run when app is not currently quitting to avoid NullReferenceException
        if (!_isQuitting) UnitEyeAPI.GetAOIManagerInstance()?.RemoveAOI(_tagList);
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

        //Return false if the tag is not being focused on any gameObject or the script is not enabled
        if (!_tagList.focused || !this.enabled)
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
