using UnitEye;
using UnityEngine;
namespace UnitEye
{

    /// <summary>
    /// This class showcases a small game where multiple objects can be moved with your eyes.
    /// In this class the UnitEye pipeline is accessed by using a reference to the Gaze object.
    /// Alternatively you could also inherit from Gaze itself to access its features.
    /// </summary>
    public class GazeGame : MonoBehaviour /*Gaze*/
    {
        private int focusedCount;
        private AOITagList tagList;

        bool moving;

        //When inheriting, remove gaze reference
        public HomulerGaze gaze;

        RaycastHit hitObject = new RaycastHit();

        void Start()
        {
            if (gaze == null) gaze = FindAnyObjectByType<HomulerGaze>();
            tagList = new AOITagList("gazeGameRay");
            tagList.AddTag("gazeGame");
            if (gaze != null)
                gaze.AOIManager.AddAOI(tagList);
        }

        void Update()
        {
            if (gaze == null || tagList == null) return;

            if (focusedCount > 29)
                focusedCount = 30;
            else if (tagList.focused)
                focusedCount++;
            else if (!moving)
                focusedCount = 0;

            if (!moving && focusedCount > 29 && tagList.hitRaycast.collider != null)
            {
                hitObject = tagList.hitRaycast;
                var rend = hitObject.collider.gameObject.GetComponent<Renderer>();
                if (rend != null) rend.material.color = Color.green;
                moving = true;
            }
            else if (gaze.Blinking)
            {
                focusedCount = 0;
                if (hitObject.collider != null)
                {
                    var rend = hitObject.collider.gameObject.GetComponent<Renderer>();
                    if (rend != null) rend.material.color = Color.grey;
                }
                hitObject = new RaycastHit();
                moving = false;
            }

            if (hitObject.collider != null)
            {
                var cam = Camera.main;
                if (cam != null)
                {
                    float depth = Vector3.Dot(hitObject.transform.position - cam.transform.position, cam.transform.forward);
                    if (depth <= cam.nearClipPlane) depth = cam.nearClipPlane + 1f;
                    var gazeVector = new Vector3(gaze.gazeLocation.x, Screen.height - gaze.gazeLocation.y, depth);
                    var vectorGaze = cam.ScreenToWorldPoint(gazeVector);
                    var pos = hitObject.transform.position;
                    pos.x = vectorGaze.x;
                    pos.y = vectorGaze.y;
                    hitObject.transform.position = pos;
                }
            }
        }

        void OnGUI()
        {
            GUI.Label(new Rect(100, 300, 100, 100), $"{focusedCount}");
            GUI.Label(new Rect(100, 320, 200, 100), $"{(tagList != null && tagList.hitRaycast.collider != null ? tagList.hitRaycast.collider.gameObject.name : "None")}");
        }
    }
}
