using System;
using System.Collections.Generic;
using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// This class allows easy access to all required eye tracking results.
    /// If you want to implement eye tracking using UnitEye in your application, this should be the way to go.
    /// </summary>
    public class UnitEyeAPI
    {
        // Migrated to the Barracuda-free HomulerGaze pipeline (native MediaPipe landmarks + Inference Engine).
        // Private: this is internal caching state, not API surface. It is resolved lazily on first use.
        private static HomulerGaze s_gazeScript;
        private static Camera s_mainCamera;

        internal static AOIManager TryGetAOIManagerInstance()
        {
            return s_gazeScript != null ? s_gazeScript.AOIManager : null;
        }

        /// <summary>
        /// Checks if an instance of the HomulerGaze script is in the scene and activated.
        /// </summary>
        /// <exception cref="InvalidOperationException">If no HomulerGaze script is found</exception>
        private static void CheckIsInitialized()
        {
            // Search for the HomulerGaze script when this is first called (FindFirstObjectByType:
            // FindObjectOfType is obsolete in Unity 6).
            if (s_gazeScript == null)
            {
                s_gazeScript = UnityEngine.Object.FindFirstObjectByType<HomulerGaze>();
            }
            if (s_gazeScript == null)
            {
                throw new InvalidOperationException(
                    "UnitEye not initialized. Please place an instance of the UnitEye prefab in the Scene"
                );
            }
        }

        /// <summary>
        /// Enables UnitEyes eye tracking.
        /// <remarks>
        /// Note that this has an influnce on the runtime performance.
        /// Enable only if you currently need to track the gaze.
        /// </remarks>
        /// </summary>
        public static void EnableGaze()
        {
            CheckIsInitialized();

            s_gazeScript.enabled = true;
        }

        /// <summary>
        /// Disables UnitEyes eye tracking.
        /// <remarks>
        /// Enabled by default if you have a UnitEye Prefab in the scene.
        /// </remarks>
        /// </summary>
        public static void DisableGaze()
        {
            CheckIsInitialized();

            s_gazeScript.enabled = false;
        }

        /// <summary>
        /// Checks if a user is present (face landmarks are currently available).
        /// </summary>
        /// <param name="faceDetectionScoreThreshold">Unused on the native MediaPipe pipeline (kept for source compatibility)</param>
        /// <returns>true if user present, false if absent</returns>
        public static bool IsUserPresent(float faceDetectionScoreThreshold = 0.5f)
        {
            CheckIsInitialized();

            return s_gazeScript.IsUserPresent;
        }

        /// <summary>
        /// Gets the location of the current gaze in GUI coordinates where (0, 0) is the top-left corner.
        /// </summary>
        /// <returns>The gaze location in GUI coordinates</returns>
        public static Vector2 GetGazeLocationInGUI()
        {
            CheckIsInitialized();

            return s_gazeScript.gazeLocation;
        }

        /// <summary>
        /// Gets the location of the current gaze in Screen coordinates where (0, 0) is the bottom-left corner.
        /// </summary>
        /// <returns>The gaze location in screen coordinates</returns>
        public static Vector2 GetGazeLocationInScreen()
        {
            CheckIsInitialized();

            return new Vector2(s_gazeScript.gazeLocation.x, Screen.height - s_gazeScript.gazeLocation.y);
        }

        /// <summary>
        /// Gets the location of the current gaze in 3D world coordinates.
        /// </summary>
        /// <param name="depth">z distance from the camera plane, default is -1.0f which defaults to the camera nearClipPlane</param>
        /// <returns>The gaze location in world coordinates</returns>
        public static Vector3 GetGazeLocationInWorld(float depth = -1.0f)
        {
            CheckIsInitialized();

            if (s_mainCamera == null || !s_mainCamera.isActiveAndEnabled)
                s_mainCamera = Camera.main;
            if (s_mainCamera == null)
                return Vector3.zero;

            if (depth == -1.0f)
                depth = s_mainCamera.nearClipPlane;

            var gazeScreen = new Vector3(s_gazeScript.gazeLocation.x,
                                            Screen.height - s_gazeScript.gazeLocation.y,
                                            depth);
            return s_mainCamera.ScreenToWorldPoint(gazeScreen);
        }

        /// <summary>
        /// Gets the timestamp of the last gaze
        /// </summary>
        /// <returns>Timestamp of the last gaze</returns>
        public static long GetLastGazeLocationTimestamp()
        {
            CheckIsInitialized();

            return s_gazeScript.LastGazeLocationTimeUnix;
        }

        /// <summary>
        /// Checks if the gaze is not on the screen.
        /// </summary>
        /// <returns>true if gaze offscreen, false if onscreen</returns>
        public static bool IsGazeOffscreen()
        {
            CheckIsInitialized();

            return s_gazeScript.OffscreenAOI != null && s_gazeScript.OffscreenAOI.focused;
        }

        /// <summary>
        /// Checks if user is blinking.
        /// </summary>
        /// <returns>true if blinking, false if not</returns>
        public static bool IsBlinking()
        {
            CheckIsInitialized();

            return s_gazeScript.Blinking;
        }

        /// <summary>
        /// Checks if user is drowsy.
        /// </summary>
        /// <returns>true if drowsy, false if not</returns>
        public static bool IsDrowsy()
        {
            CheckIsInitialized();

            return s_gazeScript.Drowsy;
        }

        /// <summary>
        /// Gets the distance between the user and the screen
        /// </summary>
        /// <returns>The distance from the screen</returns>
        public static float GetDistanceFromScreen()
        {
            CheckIsInitialized();

            return s_gazeScript.Distance;
        }

        /// <summary>
        /// Check if CSV logging is currently paused.
        /// </summary>
        /// <returns>True if csv logging is paused, false otherwise</returns>
        public static bool IsCSVLoggingPaused()
        {
            CheckIsInitialized();

            return s_gazeScript.PauseCSVLogging;
        }

        /// <summary>
        /// Pauses the CSV logging.
        /// </summary>
        public static void PauseCSVLogging()
        {
            CheckIsInitialized();

            s_gazeScript.PauseCSVLogging = true;
        }

        /// <summary>
        /// Unpauses the CSV logging.
        /// </summary>
        public static void UnpauseCSVLogging()
        {
            CheckIsInitialized();

            s_gazeScript.PauseCSVLogging = false;
        }

        /// <summary>
        /// Returns an instance to the CSV logger
        /// </summary>
        /// <returns>The csv logger used by the gaze script</returns>
        public static CSVLogger GetCSVLogger()
        {
            CheckIsInitialized();

            return s_gazeScript.CSVLogger;
        }

        /// <summary>
        /// Writes a line to the beginning of the CSV file. This rewrites the entire file and can freeze the program for a while if the csv file is large!
        /// </summary>
        /// <param name="line">Line to write</param>
        public static void PrependLineToCSV(string line)
        {
            CheckIsInitialized();

            if (s_gazeScript.CSVLogger != null)
                s_gazeScript.CSVLogger.PrependLine(line);
        }

        /// <summary>
        /// Returns the reference to the HomulerGaze script in the scene.
        /// </summary>
        /// <returns>HomulerGaze instance</returns>
        public static HomulerGaze GetGazeReference()
        {
            CheckIsInitialized();

            return s_gazeScript;
        }

        /// <summary>
        /// Returns the instance of an AOIManager.
        /// </summary>
        /// <returns>AOIManager instance</returns>
        public static AOIManager GetAOIManagerInstance()
        {
            CheckIsInitialized();

            return s_gazeScript.AOIManager;
        }

        /// <summary>
        /// Adds the AOI to the AOIManager
        /// </summary>
        /// <param name="aoi">The AOI to add</param>
        public static void AddAOI(AOI aoi)
        {
            CheckIsInitialized();

            s_gazeScript.AOIManager.AddAOI(aoi);
        }

        /// <summary>
        /// Removes the AOI from the AOIManager
        /// </summary>
        /// <param name="aoi">The AOI to remove</param>
        public static void RemoveAOI(AOI aoi)
        {
            CheckIsInitialized();

            s_gazeScript.AOIManager.RemoveAOI(aoi);
        }

        /// <summary>
        /// Removes the AOI with the same uID from the AOIManager
        /// </summary>
        /// <param name="uID">The ID to remove</param>
        public static void RemoveAOI(string uID)
        {
            CheckIsInitialized();

            s_gazeScript.AOIManager.RemoveAOI(uID);
        }

        /// <summary>
        /// Returns the head pose.
        /// </summary>
        /// <returns>The head rotation as a Quaternion/returns>
        public static Quaternion GetHeadPose()
        {
            CheckIsInitialized();
            if (s_gazeScript.Provider == null)
                return Quaternion.identity;

            //IGazeProvider.HeadPoseEuler is in RADIANS (native provider derives it via Mathf.Atan on
            //the landmarks), but Quaternion.Euler expects DEGREES — convert, or the rotation is off by
            //a ~57x factor and meaningless.
            var pose = s_gazeScript.Provider.HeadPoseEuler * Mathf.Rad2Deg;
            return Quaternion.Euler(pose.x, pose.y, pose.z);
        }

        /// <summary>
        /// Gets a texture of the left eye of the user.
        /// </summary>
        /// <returns>Left eye render texture</returns>
        public static RenderTexture GetLeftEyeTexture()
        {
            CheckIsInitialized();

            return s_gazeScript.Provider?.LeftEyeTexture;
        }

        /// <summary>
        /// Gets a texture of the right eye of the user.
        /// </summary>
        /// <returns>Right eye render texture</returns>
        public static RenderTexture GetRightEyeTexture()
        {
            CheckIsInitialized();

            return s_gazeScript.Provider?.RightEyeTexture;
        }

        /// <summary>
        /// Get the game object that is currently looked at.
        /// Requires the game object to have a "Gazeable" component which is enabled.
        /// </summary>
        /// <returns>The currently focused object or null if no object is focused</returns>
        public static GameObject GetFocusedGameObject()
        {
            CheckIsInitialized();

            foreach (AOI aoi in s_gazeScript.AOIManager.GetAOIs())
            {
                var tagList = aoi as AOITagList;
                if (tagList == null || !tagList.focused || tagList.xray)
                    continue;
                var collider = tagList.hitRaycast.collider;
                if (collider == null)
                    continue;
                var gameObject = collider.gameObject;
                var gazeable = gameObject.GetComponent<Gazeable>();
                if (gazeable != null && gazeable.enabled)
                    return gameObject;
            }
            return null;
        }

        /// <summary>
        /// Get a list of all game objects that are currently looked at.
        /// Requires the game object to have a "GazeableXray" component which is enabled.
        /// </summary>
        /// <returns>A list with the currently focused objects or null if no object is focused</returns>
        public static List<GameObject> GetFocusedGameObjects()
        {
            CheckIsInitialized();

            List<GameObject> objectList = null;
            foreach (AOI aoi in s_gazeScript.AOIManager.GetAOIs())
            {
                var tagList = aoi as AOITagList;
                if (tagList == null || !tagList.focused || !tagList.xray)
                    continue;
                foreach (var raycasthit in tagList.hitRaycastList)
                {
                    if (raycasthit.collider == null)
                        continue;
                    var gameObject = raycasthit.collider.gameObject;
                    var gazeablexray = gameObject.GetComponent<GazeableXray>();
                    if (gazeablexray != null && gazeablexray.enabled)
                    {
                        objectList ??= new List<GameObject>();
                        if (!objectList.Contains(gameObject))
                            objectList.Add(gameObject);
                    }
                }
            }
            return objectList;
        }
    }
}
