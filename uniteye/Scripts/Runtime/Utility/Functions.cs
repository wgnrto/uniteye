using MediaPipe.Holistic;
using Unity.Mathematics;
using UnityEngine;

namespace UnitEye
{
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

        //Buffer Texture2D to avoid memory leak
        private static Texture2D _getEyeTextureBuffer = new Texture2D(256,256);
        /// <summary>
        /// Calculates EyeCrops similar to EyeMU and returns them as a Texture
        /// </summary>
        /// <param name="holisticPipeline">HolisticPipeline reference to access vertex data</param>
        /// <param name="source">WebCamTexture to source from</param>
        /// <param name="leftVertex">Left vertex index from face mesh</param>
        /// <param name="rightVertex">Right vertex index from face mesh</param>
        /// <param name="imageSize">Square image size to use, default 128x128</param>
        /// <returns>EyeCrop Texture</returns>
        public static Texture GetEyeTexture(HolisticPipeline holisticPipeline, WebCamTexture source, int leftVertex, int rightVertex, int imageSize = 128)
        {
            //Get xy Coords of eye Vertices
            float2 leftCorner = holisticPipeline.facePipeline.GetFaceRegionVertex(leftVertex).xy;
            float2 rightCorner = holisticPipeline.facePipeline.GetFaceRegionVertex(rightVertex).xy;

            //Preset final coordinates and size
            int cropSize = imageSize;
            int leftX = 0;
            int yBot = 0;

            //scaleFactor because y coordinates are in a square with width as side length
            float scaleFactor = (float)source.height / (float)source.width;

            //Calculation similar to EyeMU approach
            float eyeLength = leftCorner.x - rightCorner.x;
            float xShift = eyeLength * 0.2f;
            eyeLength += 2f * xShift;
            float yShift = eyeLength * 0.5f;
            float yRef = (leftCorner.y + rightCorner.y) * 0.5f;

            //Map yRef to our height because of square xy coordinates
            yRef -= yShift;
            yRef = (yRef - ((1f - scaleFactor) * 0.5f)) / scaleFactor;
            //Clamp so that GetPixels doesn't throw a fit
            yRef = Mathf.Clamp(yRef, 0.0f, 1.0f);

            //Calculate coordinates and size
            cropSize = (int)(eyeLength * (float)source.width);
            leftX = (int)((rightCorner.x - xShift) * (float)source.width);
            yBot = (int)(yRef * (float)source.height);

            //If crop is about to be size 0 skip SetPixels
            if (source != null && cropSize > 0 && leftX >= 0 && leftX <= source.width - cropSize && yBot >= 0 && yBot <= source.height - cropSize)
            {
                //Reinitialize buffer Texture2D with new size, does not duplicate to avoid memory leak
                _getEyeTextureBuffer.Reinitialize((int)(eyeLength * (float)source.width), (int)(eyeLength * (float)source.width));

                //Copy relevant pixels from source to croppedSource
                _getEyeTextureBuffer.SetPixels(source.GetPixels(leftX, yBot, cropSize, cropSize));
                _getEyeTextureBuffer.Apply();
            }
            //return croppedSource as Texture
            return (Texture)_getEyeTextureBuffer;
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