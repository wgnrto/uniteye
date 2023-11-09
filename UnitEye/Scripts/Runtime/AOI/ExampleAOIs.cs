using System.Collections.Generic;
using UnityEngine;
using UnitEye;
using Unity.Mathematics;

/// <summary>
/// Example class to quicly add multiple AOIs to the scene.
/// </summary>
public class ExampleAOIs : MonoBehaviour
{
    void Start()
    {
        //Get AOIManager instance from first Gaze component in scene using the API call
        AOIManager aoiManager = UnitEyeAPI.GetAOIManagerInstance();

        //----------------------------------------------------------------------------------------------------
        // Test AOIs
        //----------------------------------------------------------------------------------------------------

        //Save the reference to AOIBox
        AOIBox middleBox = new AOIBox("Middle", new Vector2(0.5f, 0.5f), new Vector2(0.9f, 0.9f), false, true, true);
        aoiManager.AddAOI(middleBox);
        //Change middleBox
        middleBox.startpoint = new Vector2(0.1f, 0.1f);
        //You can add AOIs to the aoiManager without saving the reference aswell
        aoiManager.AddAOI(new AOIBox("Edge", new Vector2(0.1f, 0.1f), new Vector2(0.9f, 0.9f), true, true, true));
        aoiManager.AddAOI(new AOIBox("Monkey", new Vector2(0.8925573587f, 0.14528101802f), new Vector2(1f, 0.48144220572f), true, true, true));
        //And get the with GetAOIFromList(uID)
        AOIBox monkey = (AOIBox) aoiManager.GetAOIFromList("Monkey");
        monkey.inverted = false;
        aoiManager.AddAOI(new AOICircle("CircleTopLeft", new Vector2(0f, 0f), 0.1f, false, true, true));
        aoiManager.AddAOI(new AOICircle("CircleMiddle", new Vector2(0.5f, 0.5f), 0.2f));
        aoiManager.AddAOI(new AOICircle("CircleTopRight", new Vector2(1f, 0f), 0.1f, false, true, true));
        aoiManager.AddAOI(new AOICircle("CircleBottomLeft", new Vector2(0f, 1f), 0.1f, false, true, true));
        aoiManager.AddAOI(new AOICircle("CircleBottomRight", new Vector2(1f, 1f), 0.1f, false, true, true));
        aoiManager.AddAOI(new AOICapsule("LeftEdgeCapsule", new Vector2(0.1f, 0.09f), new Vector2(0.1f, 0.89f), 0.11f, false, true, true));
        aoiManager.AddAOI(new AOICapsule("RightEdgeCapsule", new Vector2(0.9f, 0.09f), new Vector2(0.9f, 0.89f), 0.11f, false, true, true));
        aoiManager.AddAOI(new AOICapsuleBox("LeftHalfCapsuleBox", new Vector2(0.25f, 0f), new Vector2(0.25f, 1f), 0.25f));
        aoiManager.AddAOI(new AOICapsuleBox("RightHalfCapsuleBox", new Vector2(0.75f, 0f), new Vector2(0.75f, 1f), 0.25f));
        aoiManager.AddAOI(new AOICapsuleBox("DiagonalTopLeftToBottomRight", new Vector2(0.2f, 0.2f), new Vector2(0.8f, 0.8f), 0.1f, false, true, true));

        List<AOI> combinedTestList = new List<AOI>()
        {
            new AOIBox("Box1", new Vector2(0.0f, 0.0f), new Vector2(0.5f, 0.2f)),
            new AOIBox("Box2", new Vector2(0.0f, 0.2f), new Vector2(0.5f, 0.4f)),
            new AOIBox("Box3", new Vector2(0.0f, 0.4f), new Vector2(0.5f, 0.6f)),
            new AOIBox("Box4", new Vector2(0.0f, 0.6f), new Vector2(0.5f, 0.8f)),
            new AOIBox("Box5", new Vector2(0.0f, 0.8f), new Vector2(0.5f, 1.0f))
        };
        aoiManager.AddAOI(new AOICombined("CombinedLeft", combinedTestList));

        AOITagList tagList = new AOITagList("TagList", true);
        tagList.AddTag("banana");
        tagList.AddTag("cucumber");
        aoiManager.AddAOI(tagList);

        List<Vector2> polypoints = new List<Vector2>()
        {
            new Vector2(0.1f, 0.2f),
            new Vector2(0.1f, 0.3f),
            new Vector2(0.2f, 0.5f),
            new Vector2(0.3f, 0.4f),
            new Vector2(0.4f, 0.5f),
            new Vector2(0.7f, 0.25f),
            new Vector2(0.5f, 0.1f)
        };
        AOIPolygon aoiPoly = new AOIPolygon("Polygon", polypoints, false, true, true);
        aoiManager.AddAOI(aoiPoly);

        //----------------------------------------------------------------------------------------------------
    }
}
