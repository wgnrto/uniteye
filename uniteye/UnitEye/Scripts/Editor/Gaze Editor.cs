using UnityEngine;
using UnityEditor;

namespace UnitEye
{
    [CustomEditor(typeof(Gaze))]
    class GazeEditor : Editor
    {
        SerializedProperty webCamInput;
        SerializedProperty dot;
        SerializedProperty csvLogger;
        SerializedProperty drawDot;
        bool drawDotBool = true;
        SerializedProperty showEyes;
        bool showEyesBool = true;
        SerializedProperty visualizeAOI;
        bool visualizeAOIBool = false;
        SerializedProperty showGazeUI;
        bool showGazeUIBool = false;

        SerializedProperty calibrations;
        SerializedProperty gazeLocation;
        SerializedProperty filtering;

        SerializedProperty easefactor;
        SerializedProperty q;
        SerializedProperty r;
        SerializedProperty beta;
        SerializedProperty mincutoff;
        SerializedProperty dcutoff;


        void OnEnable()
        {
            webCamInput = serializedObject.FindProperty("webCamInput");
            dot = serializedObject.FindProperty("dot");
            csvLogger = serializedObject.FindProperty("csvLogger");
            drawDot = serializedObject.FindProperty("drawDot");
            drawDotBool = drawDot.boolValue;
            showEyes = serializedObject.FindProperty("showEyes");
            showEyesBool = showEyes.boolValue;
            visualizeAOI = serializedObject.FindProperty("visualizeAOI");
            visualizeAOIBool = visualizeAOI.boolValue;
            showGazeUI = serializedObject.FindProperty("showGazeUI");
            showGazeUIBool = showGazeUI.boolValue;
            calibrations = serializedObject.FindProperty("_calibrations");
            gazeLocation = serializedObject.FindProperty("gazeLocation");
            filtering = serializedObject.FindProperty("_filtering");

            easefactor = serializedObject.FindProperty("easefactor");
            q = serializedObject.FindProperty("Q");
            r = serializedObject.FindProperty("R");
            beta = serializedObject.FindProperty("beta");
            mincutoff = serializedObject.FindProperty("mincutoff");
            dcutoff = serializedObject.FindProperty("dcutoff");
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            EditorGUILayout.PropertyField(webCamInput);
            EditorGUILayout.Separator();

            EditorGUILayout.PropertyField(dot, new GUIContent("Gaze Location Dot:"));
            EditorGUILayout.Separator();

            EditorGUILayout.PropertyField(csvLogger, new GUIContent("CSV Logger:"));
            EditorGUILayout.Separator();

            drawDotBool = EditorGUILayout.Toggle("Draw Dot?", drawDotBool);
            drawDot.boolValue = drawDotBool;

            showEyesBool = EditorGUILayout.Toggle("Show Eyecrops?", showEyesBool);
            showEyes.boolValue = showEyesBool;

            visualizeAOIBool = EditorGUILayout.Toggle("Visualize AOIs?", visualizeAOIBool);
            visualizeAOI.boolValue = visualizeAOIBool;

            showGazeUIBool = EditorGUILayout.Toggle("Show Gaze UI button?", showGazeUIBool);
            showGazeUI.boolValue = showGazeUIBool;
            EditorGUILayout.Separator();

            EditorGUILayout.PropertyField(calibrations, new GUIContent("Calibration Type"));
            EditorGUILayout.PropertyField(gazeLocation);
            EditorGUILayout.Separator();

            EditorGUILayout.PropertyField(filtering);
            switch ((Filtering)filtering.intValue)
            {
                case Filtering.Kalman:
                    EditorGUILayout.PropertyField(q);
                    EditorGUILayout.PropertyField(r);
                    break;
                case Filtering.Easing:
                    EditorGUILayout.PropertyField(easefactor);
                    break;
                case Filtering.KalmanEasing:
                    EditorGUILayout.PropertyField(easefactor);
                    EditorGUILayout.PropertyField(q);
                    EditorGUILayout.PropertyField(r);
                    break;
                case Filtering.EasingKalman:
                    EditorGUILayout.PropertyField(q);
                    EditorGUILayout.PropertyField(r);
                    EditorGUILayout.PropertyField(easefactor);
                    break;
                case Filtering.OneEuro:
                    EditorGUILayout.PropertyField(beta);
                    EditorGUILayout.PropertyField(mincutoff);
                    EditorGUILayout.PropertyField(dcutoff);
                    break;
                default:
                    break;
            }

            serializedObject.ApplyModifiedProperties();
        }
    }
}
