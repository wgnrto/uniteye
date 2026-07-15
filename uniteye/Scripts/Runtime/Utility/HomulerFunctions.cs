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
        //Note: the CPU eye-crop path (GetEyeTexture's GetPixels readback + FlipTexture's GetPixels32
        //round-trip, both run every frame) was replaced by a GPU Graphics.Blit crop in
        //HomulerEyeMURunner.ComputeEyes/BlitEyeCrop, so those two methods and their Texture2D buffers
        //were removed. GetEyeCropRect (below) still computes the crop rectangle used by the GPU blit.

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
            //Dispatch counts THREAD GROUPS, and the kernel is [numthreads(8,8,1)] — dispatching
            //imageSize x imageSize groups launched 64x more threads than pixels (~1M threads for a
            //128x128 image, twice per frame). Ceil-divide so every pixel is still covered when
            //imageSize is not a multiple of 8.
            int groups = (imageSize + 7) / 8;
            preprocessCS.Dispatch(0, groups, groups, 1);

            return destination;
        }
    }
}
#endif
