using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// Shared, inference-backend-agnostic utility functions used across UnitEye.
    /// The Holistic/Barracuda-specific GetEyeTexture(HolisticPipeline, ...) was removed with
    /// the Barracuda pipeline; the homuler path uses HomulerFunctions.GetEyeTexture(landmarks, ...).
    /// </summary>
    public class Functions
    {
        //Buffer Texture2D to avoid memory leak
        private static Texture2D _flipTextureBuffer = new Texture2D(256, 256);
        /// <summary>
        /// Horizontally flips a Texture that can be converted into a Texture2D (not a RenderTexture)
        /// </summary>
        /// <param name="source">Texture to be flipped </param>
        /// <returns>Flipped Texture</returns>
        public static Texture FlipTexture(Texture source)
        {
            //If source is size 0 return 1x1 pixel dummy texture
            if (source.width == 0 || source.height == 0)
            {
                //Reinitialize buffer as empty 1x1 texture
                _flipTextureBuffer.Reinitialize(1, 1);
                return (Texture)_flipTextureBuffer;
            }

            //Preset variables to not call methods in the for loop
            var sourceWidth = source.width;
            var sourceHeight = source.height;

            //Reinitialize buffer Texture2D with new size, does not duplicate to avoid memory leak
            _flipTextureBuffer.Reinitialize(source.width, source.height);

            //Write pixels to Color32[] array
            var pixelArray = ((Texture2D)source).GetPixels32();

            //Use System.Array.Reverse to reverse each horitontal pixel chunk in the pixelArray
            for (int i = 0; i < sourceHeight; i++)
            {
                System.Array.Reverse(pixelArray, i * sourceWidth, sourceWidth);
            }

            //Return a flipped Texture
            _flipTextureBuffer.SetPixels32(pixelArray);
            _flipTextureBuffer.Apply();
            return (Texture)_flipTextureBuffer;
        }

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
        /// Preprocess Image using a shader to provide the correct image format for the model
        /// </summary>
        /// <param name="source">Source RenderTexture</param>
        /// <param name="destination">Destination RenderTexture</param>
        /// <param name="preprocessCS">Shader to use</param>
        /// <param name="imageSize">Square image size to use, default 128x128</param>
        /// <returns>Processed RenderTexture</returns>
        public static RenderTexture PreprocessImage(RenderTexture source, RenderTexture destination, ComputeShader preprocessCS, int imageSize = 128)
        {
            preprocessCS.SetTexture(0, "_Texture", source);
            preprocessCS.SetTexture(0, "_Tensor", destination);
            preprocessCS.SetInt("_ImageSize", imageSize);
            preprocessCS.Dispatch(0, imageSize, imageSize, 1);

            return destination;
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