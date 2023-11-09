using UnitEye;
using UnityEngine;

/// <summary>
/// This class showcases a small game where multiple objects can be moved with your eyes.
/// In this class the UnitEye pipeline is accessed by using the UnitEyeAPI.
/// In most cases this is the preferred and easier way to access UnitEye instead of using the Gaze script directly.
/// </summary>
public class GazeGameAPI : MonoBehaviour
{
    private int focusedCount;

    bool moving;

    GameObject hitObject = null;

    void Start()
    {
    }

    void Update()
    {
        if (focusedCount > 29)
            focusedCount = 30;
        else if (UnitEyeAPI.GetFocusedGameObject() != null)
            focusedCount++;
        else if (!moving)
            focusedCount = 0;

        if (!moving && focusedCount > 29)
        {
            hitObject = UnitEyeAPI.GetFocusedGameObject();
            hitObject.GetComponent<Renderer>().material.color = Color.green;
            moving = true;
        }
        else if (UnitEyeAPI.IsBlinking())
        {
            focusedCount = 0;
            if (hitObject != null) hitObject.GetComponent<Renderer>().material.color = Color.grey;
            hitObject = null;
            moving = false;
        }

        if (hitObject != null)
        {
            var vector = hitObject.transform.localPosition;
            var gazeVector = UnitEyeAPI.GetGazeLocationInWorld();
            vector.x = gazeVector.x;
            vector.y = gazeVector.y;
            hitObject.transform.localPosition = vector;
        }
    }
    void OnGUI()
    {
        GUI.Label(new Rect(100, 300, 100, 100), $"{focusedCount}");
        var focusedObject = UnitEyeAPI.GetFocusedGameObject();
        GUI.Label(new Rect(100, 320, 100, 100), $"{(focusedObject != null ? focusedObject.name : "None")}");
    }
}
