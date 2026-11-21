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
        SerializedProperty timeUntilWrite;
        SerializedProperty logsPerSecond;

        void OnEnable()
        {
            _baseFileName = serializedObject.FindProperty("_baseFileName");
            _baseFolderPath = serializedObject.FindProperty("_baseFolderPath");
            useDefaultFolder = serializedObject.FindProperty("useDefaultFolder");
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

            EditorGUILayout.LabelField("Use default CSVLogs folder (Editor: Assets/CSVLogs, Builds: persistentDataPath/CSVLogs)?");
            //Bind directly (was mirrored through a cached bool, breaking Undo/multi-object editing).
            EditorGUILayout.PropertyField(useDefaultFolder, new GUIContent(" "));

            if (!useDefaultFolder.boolValue)
            {
                EditorGUILayout.PropertyField(_baseFolderPath, new GUIContent("Current folder: "));
                EditorGUILayout.LabelField("Leave empty to use the platform default folder, or select a custom folder:");
                if (GUILayout.Button("Select folder"))
                {
                    var newPath = EditorUtility.OpenFolderPanel("Select folder to store CSV files in", _baseFolderPath.stringValue, "");
                    if (!string.IsNullOrEmpty(newPath))
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
