using UnityEditor;
using UnityEngine;
using Mediapipe.Unity;

namespace UnitEye
{

    [CustomEditor(typeof(WebCamSource))]
    class HomulerWebcamInput : Editor
    {
        static readonly GUIContent SelectLabel = new GUIContent("Select");

        SerializedProperty _name;
        SerializedProperty _preferableDefaultWidth;
        SerializedProperty _requestedFps;

        void OnEnable()
        {
            _name = serializedObject.FindProperty("_name");
            _preferableDefaultWidth = serializedObject.FindProperty("_preferableDefaultWidth");
            _requestedFps = serializedObject.FindProperty("_requestedFps");
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
                    menu.AddItem(new GUIContent(selectedName), _name.stringValue == selectedName, () =>
                    {
                        serializedObject.Update();
                        _name.stringValue = selectedName;
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

            //Do NOT query WebCamTexture.devices or overwrite _name on every GUI pass: enumerating DirectShow
            //devices every frame stalls the Inspector, and mutating _name on selection marks scenes/prefabs dirty.
            EditorGUILayout.PropertyField(_name, new GUIContent("Webcam Name"));

            var rect = EditorGUILayout.GetControlRect(false, GUILayout.Width(60));
            if (EditorGUI.DropdownButton(rect, SelectLabel, FocusType.Keyboard))
                ShowDeviceSelector(rect);

            EditorGUILayout.EndHorizontal();

            EditorGUILayout.PropertyField(_preferableDefaultWidth, new GUIContent("Webcam Resolution (Width)"));
            if (_requestedFps != null)
                EditorGUILayout.PropertyField(_requestedFps, new GUIContent("Target Camera FPS"));

            serializedObject.ApplyModifiedProperties();
        }
    }

}
