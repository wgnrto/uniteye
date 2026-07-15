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

        void OnEnable()
        {
            _name = serializedObject.FindProperty("_name");
            _preferableDefaultWidth = serializedObject.FindProperty("_preferableDefaultWidth");
        }

        void ShowDeviceSelector(Rect rect)
        {
            var menu = new GenericMenu();

            foreach (var device in WebCamTexture.devices)
                menu.AddItem(new GUIContent(device.name), false,
                             () => {
                                 serializedObject.Update();
                                 _name.stringValue = device.name;
                                 serializedObject.ApplyModifiedProperties();
                             });

            menu.DropDown(rect);
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            EditorGUILayout.BeginHorizontal();

            //Initialize first available device if none is selected or device is not found and reset resolution to WebCamInput.DefaultResolution
            if (_name.stringValue == "" || !WebCamInput.CheckValidWebCam(_name.stringValue))
            {
                if (WebCamTexture.devices.Length > 0)
                    _name.stringValue = WebCamTexture.devices[0].name;
            }

            EditorGUILayout.PropertyField(_name, new GUIContent("Webcam Name"));

            var rect = EditorGUILayout.GetControlRect(false, GUILayout.Width(60));
            if (EditorGUI.DropdownButton(rect, SelectLabel, FocusType.Keyboard))
                ShowDeviceSelector(rect);

            EditorGUILayout.EndHorizontal();

            EditorGUILayout.PropertyField(_preferableDefaultWidth, new GUIContent("Webcam Resolution"));

            serializedObject.ApplyModifiedProperties();
        }
    }

}
