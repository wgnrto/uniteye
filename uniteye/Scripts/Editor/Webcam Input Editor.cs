using UnityEditor;
using UnityEngine;

namespace UnitEye
{

    [CustomEditor(typeof(WebCamInput))]
    class WebcamInputEditor : Editor
    {
        static readonly GUIContent SelectLabel = new GUIContent("Select");

        SerializedProperty webCamName;
        SerializedProperty webCamResolution;
        SerializedProperty rawImage;
        SerializedProperty staticInput;
        SerializedProperty targetFramerate;
        SerializedProperty mirrorImage;
        bool mirrorImageBool = false;

        void OnEnable()
        {
            webCamName = serializedObject.FindProperty("webCamName");
            webCamResolution = serializedObject.FindProperty("webCamResolution");
            targetFramerate = serializedObject.FindProperty("targetFramerate");
            staticInput = serializedObject.FindProperty("staticInput");
            rawImage = serializedObject.FindProperty("rawImage");
            mirrorImage = serializedObject.FindProperty("mirrorImage");
            mirrorImageBool = mirrorImage.boolValue;
        }

        void ShowDeviceSelector(Rect rect)
        {
            var menu = new GenericMenu();

            foreach (var device in WebCamTexture.devices)
                menu.AddItem(new GUIContent(device.name), false,
                             () => {
                                 serializedObject.Update();
                                 webCamName.stringValue = device.name;
                                 serializedObject.ApplyModifiedProperties();
                             });

            menu.DropDown(rect);
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            EditorGUILayout.BeginHorizontal();

            //Initialize first available device if none is selected or device is not found and reset resolution to WebCamInput.DefaultResolution
            if (webCamName.stringValue == "" || !WebCamInput.CheckValidWebCam(webCamName.stringValue))
            {
                if (WebCamTexture.devices.Length > 0)
                    webCamName.stringValue = WebCamTexture.devices[0].name;
                webCamResolution.vector2Value = WebCamInput.DefaultResolution;
            }

            EditorGUILayout.PropertyField(webCamName, new GUIContent("Webcam Name"));

            var rect = EditorGUILayout.GetControlRect(false, GUILayout.Width(60));
            if (EditorGUI.DropdownButton(rect, SelectLabel, FocusType.Keyboard))
                ShowDeviceSelector(rect);

            EditorGUILayout.EndHorizontal();

            EditorGUILayout.PropertyField(webCamResolution, new GUIContent("Webcam Resolution"));
            EditorGUILayout.PropertyField(targetFramerate, new GUIContent("Maximum Framerate"));

            EditorGUILayout.Separator();

            EditorGUILayout.LabelField(new GUIContent("Optional settings:"));
            EditorGUILayout.PropertyField(rawImage, new GUIContent("Image to draw webcam on"));
            EditorGUILayout.PropertyField(staticInput, new GUIContent("Static Input"));

            mirrorImageBool = EditorGUILayout.Toggle("Mirror image horizontally", mirrorImageBool);
            mirrorImage.boolValue = mirrorImageBool;

            serializedObject.ApplyModifiedProperties();
        }
    }

}
