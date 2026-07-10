using UnityEngine;
using UnityEditor;

namespace UnitEye
{
    [CustomEditor(typeof(HomulerGaze))]
    class HomulerGazeEditor : Editor
    {
        SerializedProperty mediaPipeGO;
        SerializedProperty calibration;
        SerializedProperty dot;
        SerializedProperty csvLogger;
        SerializedProperty drawDot;
        SerializedProperty showEyes;
        SerializedProperty visualizeAOI;
        SerializedProperty showGazeUI;

        SerializedProperty calibrations;
        SerializedProperty gazeLocation;
        SerializedProperty filtering;

        SerializedProperty easefactor;
        SerializedProperty q;
        SerializedProperty r;
        SerializedProperty beta;
        SerializedProperty mincutoff;
        SerializedProperty dcutoff;

        SerializedProperty frameRate;
        SerializedProperty gazeBackbone;


        void OnEnable()
        {
            mediaPipeGO = serializedObject.FindProperty("_mediaPipeGO");
            gazeBackbone = serializedObject.FindProperty("_gazeBackbone");
            calibration = serializedObject.FindProperty("_calibrationScript");
            dot = serializedObject.FindProperty("dot");
            csvLogger = serializedObject.FindProperty("_csvLogger");
            drawDot = serializedObject.FindProperty("drawDot");
            showEyes = serializedObject.FindProperty("showEyes");
            visualizeAOI = serializedObject.FindProperty("visualizeAOI");
            showGazeUI = serializedObject.FindProperty("showGazeUI");
            calibrations = serializedObject.FindProperty("_calibrations");
            gazeLocation = serializedObject.FindProperty("gazeLocation");
            filtering = serializedObject.FindProperty("_filtering");

            easefactor = serializedObject.FindProperty("easefactor");
            q = serializedObject.FindProperty("Q");
            r = serializedObject.FindProperty("R");
            beta = serializedObject.FindProperty("beta");
            mincutoff = serializedObject.FindProperty("mincutoff");
            dcutoff = serializedObject.FindProperty("dcutoff");

            frameRate = serializedObject.FindProperty("_frameRate");
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            EditorGUILayout.PropertyField(mediaPipeGO);
            EditorGUILayout.PropertyField(calibration);
            EditorGUILayout.PropertyField(gazeBackbone, new GUIContent("Gaze Backbone (model)"));
            EditorGUILayout.Separator();

            EditorGUILayout.PropertyField(dot, new GUIContent("Gaze Location Dot:"));
            EditorGUILayout.Separator();

            EditorGUILayout.PropertyField(csvLogger, new GUIContent("CSV Logger:"));
            EditorGUILayout.Separator();

            //Bind the SerializedProperty directly (was mirrored through cached bools, which broke
            //Undo/Reset and multi-object editing and ignored external changes to the value).
            EditorGUILayout.PropertyField(drawDot, new GUIContent("Draw Dot?"));
            EditorGUILayout.PropertyField(showEyes, new GUIContent("Show Eyecrops?"));
            EditorGUILayout.PropertyField(visualizeAOI, new GUIContent("Visualize AOIs?"));
            EditorGUILayout.PropertyField(showGazeUI, new GUIContent("Show Gaze UI button?"));
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

            EditorGUILayout.Separator();
            EditorGUILayout.PropertyField(frameRate);

            serializedObject.ApplyModifiedProperties();
        }
    }
}
