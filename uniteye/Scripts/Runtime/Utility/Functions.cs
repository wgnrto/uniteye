using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// Shared, inference-backend-agnostic utility functions used across UnitEye.
    /// The Holistic/Barracuda-specific eye-crop helpers were removed with the Barracuda pipeline; the
    /// homuler path crops on the GPU in HomulerEyeMURunner using HomulerFunctions.GetEyeCropRect(...).
    /// </summary>
    public class Functions
    {
        //Note: FlipTexture and PreprocessImage previously lived here too but were dead duplicates of the
        //versions in HomulerFunctions (which the homuler inference path actually calls); removed to keep
        //a single implementation of each. This class keeps the inference-agnostic helpers below.

        /// <summary>
        /// Converts pixels to mm using Unity Screen.dpi
        /// </summary>
        /// <param name="pixels"></param>
        /// <returns>mm in float</returns>
        public static float PixelsToMm(float pixels)
        {
            return PixelsToMm(pixels, Screen.dpi);
        }
        /// <summary>
        /// Converts pixels to mm using custom dpi
        /// </summary>
        /// <param name="pixels"></param>
        /// <param name="dpi"></param>
        /// <returns>mm in float</returns>
        public static float PixelsToMm(float pixels, float dpi)
        {
            return pixels * 25.4f / dpi;
        }

        /// <summary>
        /// Quits the application. If in Editor it just stops playing
        /// </summary>
        public static void Quit()
        {
            //If in editor stop the editor
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#else
        //If in build just quit the Application
        Application.Quit();
#endif
        }
    }
}