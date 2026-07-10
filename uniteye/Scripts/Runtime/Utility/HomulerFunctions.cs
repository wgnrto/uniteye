// Excluded from WebGL player builds: depends on the native MediaPipe plugin (Mediapipe.Runtime
// has no wasm library, so IL2CPP linking fails). Kept for the Editor regardless of build target.
#if !UNITY_WEBGL || UNITY_EDITOR
using Mediapipe;
using Mediapipe.Unity.FaceMesh;
using System;
using System.Collections.Generic;
using UnityEngine;

namespace UnitEye
{
    public static class HomulerFunctions
    { 
        //Reusable buffer for FlipTexture. This runs every frame on the left eye crop; the old code
        //allocated a `new Texture2D` per call and never Destroy()'d it, leaking native GPU memory every
        //frame. It is a *separate* buffer from _getEyeTextureBuffer so a flip does not clobber the
        //right-eye crop that GetEyeTexture leaves in that shared buffer.
        private static Texture2D _flipTextureBuffer = new Texture2D(256, 256);
        /// <summary>
        /// Horizontally flips a Texture that can be converted into a Texture2D (not a RenderTexture)
        /// </summary>
        /// <param name="source">Texture to be flipped </param>
        /// <returns>Flipped Texture (a shared reused buffer; do not cache the reference across frames)</returns>
        public static Texture FlipTexture(Texture source)
        {
            //If source is size 0 return the reused buffer as a dummy texture
            if (source.width == 0 || source.height == 0)
                return _flipTextureBuffer;

            //Reinitialize buffer Texture2D with new size, reusing the same native texture (no per-frame leak)
            _flipTextureBuffer.Reinitialize(source.width, source.height);

            //Write pixels to Color32[] array
            var pixelArray = ((Texture2D)source).GetPixels32();

            //Use System.Array.Reverse to reverse each horitontal pixel chunk in the pixelArray
            for (int i = 0; i < source.height; i++)
                Array.Reverse(pixelArray, i * source.width, source.width);

            //Return a flipped Texture
            _flipTextureBuffer.SetPixels32(pixelArray);
            _flipTextureBuffer.Apply();
            return _flipTextureBuffer;
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
        public static Texture GetEyeTexture(IList<NormalizedLandmark> landmarks, WebCamTexture source, int leftVertex, int rightVertex)
        {
            var crop = GetEyeCropRect(landmarks, leftVertex, rightVertex, source.width, source.height);

            //If crop is about to be size 0 skip SetPixels
            if (source != null && crop.width > 0 && crop.x >= 0 && crop.x <= source.width - crop.width && crop.y >= 0 && crop.y <= source.height - crop.height)
            {
                //Reinitialize buffer Texture2D with new size, does not duplicate to avoid memory leak
                _getEyeTextureBuffer.Reinitialize(crop.width, crop.height);

                //Copy relevant pixels from source to croppedSource
                _getEyeTextureBuffer.SetPixels(source.GetPixels(crop.x, crop.y, crop.width, crop.height));
                _getEyeTextureBuffer.Apply();
            }
            //return croppedSource as Texture
            return _getEyeTextureBuffer;
        }

        /// <summary>
        /// Computes the eye crop rectangle in source pixel coordinates (bottom-left origin, as used
        /// by GetPixels) from two eye-corner landmarks in MediaPipe convention (normalized, y-down).
        /// Must not write to the landmarks: NormalizedLandmark is a protobuf reference type shared
        /// with EyeCorners (an EyeMU model input) and the annotation layer, so an in-place Y flip
        /// here corrupts those consumers and toggles the convention on every extra call per graph
        /// output (the pre-2026 behavior).
        /// </summary>
        public static RectInt GetEyeCropRect(IList<NormalizedLandmark> landmarks, int leftVertex, int rightVertex, int sourceWidth, int sourceHeight)
        {
            var leftCorner = landmarks[leftVertex];
            var rightCorner = landmarks[rightVertex];
            return GetEyeCropRect(leftCorner.X, leftCorner.Y, rightCorner.X, rightCorner.Y, sourceWidth, sourceHeight);
        }

        public static RectInt GetEyeCropRect(float leftX, float leftY, float rightX, float rightY, int sourceWidth, int sourceHeight)
        {
            //Convert y-down (MediaPipe) to y-up (Unity texture space) on locals only
            float leftYUp = 1f - leftY;
            float rightYUp = 1f - rightY;

            //Calculation similar to EyeMU approach
            float eyeLength = rightX - leftX;
            float xShift = eyeLength * 0.2f;
            eyeLength += 2f * xShift;
            float yShift = eyeLength * 0.5f;
            float yRef = (leftYUp + rightYUp) * 0.5f;
            yRef -= 2f * yShift;

            //Clamp so that GetPixels doesn't throw a fit
            yRef = Mathf.Clamp(yRef, 0.0f, 1.0f);

            //Calculate coordinates and size
            var cropSize = (int)(eyeLength * sourceWidth);
            var leftPx = (int)((leftX - xShift) * sourceWidth);
            var yBot = (int)(yRef * sourceHeight);

            return new RectInt(leftPx, yBot, cropSize, cropSize);
        }

        //Note: PixelsToMm and Quit were dead duplicates of the versions in Functions (which callers use)
        //and were removed. This class keeps only the MediaPipe/inference-specific helpers.

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
    }
}
#endif
