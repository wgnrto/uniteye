using UnityEngine;
using UnitEye;
using System.Collections;

[RequireComponent(typeof(Collider))]
public class Gazeable : MonoBehaviour
{
    /// <summary>
    /// Checks whether or not this game object is currently looked at.
    /// </summary>
    public bool HasGazeFocus => _tagList.focused && this.enabled && checkAllObjectsWithMatchingTag ? true : _tagList.hitRaycast.collider?.gameObject == gameObject;

    [Tooltip("If enabled, all objects with matching tags will influence HasGazeFocus boolean even if they don't have a Gazeable component.")]
    public bool checkAllObjectsWithMatchingTag = false;

    private AOITagList _tagList;
    private bool _isQuitting;
    private string _oldTag;
    private static int s_current = 0;

    void Start() 
    {
        if (gameObject.tag == "Untagged")
        {
            Debug.Log("Gazeable GameObject must have a tag different than \"Untagged\"");
        }

        _oldTag = gameObject.tag;
        _tagList = new AOITagList(gameObject.name + s_current.ToString());
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

    IEnumerator AddAOI()
    {
        //Yield until next frame to allow Gaze component to be initialized
        yield return 0;

        UnitEyeAPI.GetAOIManagerInstance()?.AddAOI(_tagList);
    }
}
