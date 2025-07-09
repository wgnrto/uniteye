using UnityEngine;
using UnityEditor;
using System.IO;

namespace UnitEye
{
    [CustomEditor(typeof(CSVLogger))]
    class CSVLoggerEditor : Editor
    {
        SerializedProperty _baseFolderPath;
        SerializedProperty _baseFileName;
        SerializedProperty useDefaultFolder;
        bool useDefaultFolderBool = true;
        SerializedProperty timeUntilWrite;
        SerializedProperty logsPerSecond;

        void OnEnable()
        {
            _baseFileName = serializedObject.FindProperty("_baseFileName");
            _baseFolderPath = serializedObject.FindProperty("_baseFolderPath");
            useDefaultFolder = serializedObject.FindProperty("useDefaultFolder");
            useDefaultFolderBool = useDefaultFolder.boolValue;
            timeUntilWrite = serializedObject.FindProperty("timeUntilWrite");
            logsPerSecond = serializedObject.FindProperty("logsPerSecond");
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            //File Name
            EditorGUILayout.LabelField("Filename without extension (timestamp will be appended automatically):");
            EditorGUILayout.PropertyField(_baseFileName, new GUIContent(""));

            GUILayout.Space(15);

            //Folder Selection
            if (!Directory.Exists(_baseFolderPath.stringValue)) _baseFolderPath.stringValue = "";

            EditorGUILayout.LabelField("Use Application.dataPath/CSVLogs folder?");
            useDefaultFolderBool = EditorGUILayout.Toggle(" ", useDefaultFolderBool);
            useDefaultFolder.boolValue = useDefaultFolderBool;

            if (!useDefaultFolderBool)
            {
                //Default to Application.dataPath/CSVLogs/ folder
                //For editor this is Assets/Logs, windows build is executablename_Data/CSVLogs/, Android uses persistentDataPath which points to /storage/emulated/0/Android/data/<packagename>/files/CSVLogs
                _baseFolderPath.stringValue = Application.platform == RuntimePlatform.Android ? $"{Application.persistentDataPath}/CSVLogs" : _baseFolderPath.stringValue == "" ? $"{Application.dataPath}/CSVLogs" : _baseFolderPath.stringValue;
                EditorGUILayout.PropertyField(_baseFolderPath, new GUIContent("Current folder: "));
                EditorGUILayout.LabelField($"Make sure the folder will be valid when building!");
                if (GUILayout.Button("Select folder"))
                {
                    var newPath = EditorUtility.OpenFolderPanel("Select folder to store CSV files in", "", "");
                    if (!newPath.Equals(""))
                        _baseFolderPath.stringValue = newPath;
                }
            }

            GUILayout.Space(15);

            //timeUntilWrite
            EditorGUILayout.PropertyField(timeUntilWrite, new GUIContent("Write File every X seconds:"));
            //logsPerSecond
            EditorGUILayout.PropertyField(logsPerSecond, new GUIContent("Log up to Y times per second:"));

            serializedObject.ApplyModifiedProperties();
        }
    }
}
