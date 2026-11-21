using UnitEye;
using UnityEngine;
namespace UnitEye
{

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
            var focused = UnitEyeAPI.GetFocusedGameObject();
            if (focusedCount > 29)
                focusedCount = 30;
            else if (focused != null)
                focusedCount++;
            else if (!moving)
                focusedCount = 0;

            if (!moving && focusedCount > 29 && focused != null)
            {
                hitObject = focused;
                var rend = hitObject.GetComponent<Renderer>();
                if (rend != null) rend.material.color = Color.green;
                moving = true;
            }
            else if (UnitEyeAPI.IsBlinking())
            {
                focusedCount = 0;
                if (hitObject != null)
                {
                    var rend = hitObject.GetComponent<Renderer>();
                    if (rend != null) rend.material.color = Color.grey;
                }
                hitObject = null;
                moving = false;
            }

            if (hitObject != null)
            {
                var cam = Camera.main;
                float depth = cam != null
                    ? Mathf.Max(cam.nearClipPlane + 0.01f, Vector3.Dot(hitObject.transform.position - cam.transform.position, cam.transform.forward))
                    : -1f;
                var gazeVector = UnitEyeAPI.GetGazeLocationInWorld(depth);
                var pos = hitObject.transform.position;
                pos.x = gazeVector.x;
                pos.y = gazeVector.y;
                hitObject.transform.position = pos;
            }
        }
        void OnGUI()
        {
            GUI.Label(new Rect(100, 300, 100, 100), $"{focusedCount}");
            var focusedObject = UnitEyeAPI.GetFocusedGameObject();
            GUI.Label(new Rect(100, 320, 100, 100), $"{(focusedObject != null ? focusedObject.name : "None")}");
        }
    }
}
