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

        void OnEnable()
        {
            webCamName = serializedObject.FindProperty("webCamName");
            webCamResolution = serializedObject.FindProperty("webCamResolution");
            targetFramerate = serializedObject.FindProperty("targetFramerate");
            staticInput = serializedObject.FindProperty("staticInput");
            rawImage = serializedObject.FindProperty("rawImage");
            mirrorImage = serializedObject.FindProperty("mirrorImage");
        }

        void ShowDeviceSelector(Rect rect)
        {
            var menu = new GenericMenu();
            var devices = WebCamTexture.devices;
            if (devices.Length == 0)
            {
                menu.AddDisabledItem(new GUIContent("No webcams detected"));
            }
            else
            {
                foreach (var device in devices)
                {
                    var selectedName = device.name;
                    menu.AddItem(new GUIContent(selectedName), webCamName.stringValue == selectedName, () =>
                    {
                        serializedObject.Update();
                        webCamName.stringValue = selectedName;
                        serializedObject.ApplyModifiedProperties();
                    });
                }
            }

            menu.DropDown(rect);
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            EditorGUILayout.BeginHorizontal();

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

            //Bind directly (was mirrored through a cached bool, breaking Undo/multi-object editing).
            EditorGUILayout.PropertyField(mirrorImage, new GUIContent("Mirror image horizontally"));

            serializedObject.ApplyModifiedProperties();
        }
    }

}
