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

        //Face-mesh landmark indices for the iris gaze features (face_landmarker_v2, 478 landmarks:
        //0..467 mesh, then two 5-point iris blocks at 468..472 and 473..477; the FIRST index of each
        //block is that iris' CENTER).
        //
        //WHICH BLOCK BELONGS TO WHICH EYE: MediaPipe documents the blocks as "left"/"right" using
        //IMAGE-relative sides, while the eye-corner constants below use SUBJECT-relative sides (the
        //canonical face-mesh naming: 33/133 = subject's right eye, which appears on the image LEFT).
        //The two conventions are mirror images, so pairing them by name pairs each iris with the WRONG
        //eye. Verified against real landmarks: index 468 lies between corners 33 and 133, and index 473
        //lies between corners 362 and 263. These constants are therefore named for the eye each iris
        //actually belongs to, so FillIrisFeatures' pairing below is correct by construction.
        public const int LeftEyeInnerCorner = 362;
        public const int LeftEyeOuterCorner = 263;
        public const int RightEyeOuterCorner = 33;
        public const int RightEyeInnerCorner = 133;
        public const int LeftIrisCenter = 473;
        public const int RightIrisCenter = 468;

        /// <summary>
        /// Fills 4 calibration features with the per-eye NORMALIZED iris offset — the position of the iris
        /// center relative to the eye-corner midpoint, divided by the corner distance:
        ///   [leftOffsetX, leftOffsetY, rightOffsetX, rightOffsetY]  written at dest[start..start+3].
        /// The iris position within the eye opening is the classic direct webcam gaze cue (it is what
        /// model-based trackers regress on); MediaPipe tracks it every frame but until now it was only used
        /// for blink/distance, never fed to the gaze calibration. Normalizing by the corner distance makes
        /// the features scale-invariant (head distance / face size cancel out). Coordinates are MediaPipe
        /// normalized (y-down); x and y are normalized by different frame axes, so the offsets fold in the
        /// camera aspect — constant within a session, absorbed by the calibration's standardization.
        /// Writes zeros when landmarks are missing or an eye is degenerate (corner distance ~ 0).
        /// </summary>
        public static void FillIrisFeatures(IList<NormalizedLandmark> landmarks, float[] dest, int start)
        {
            //Guard on the highest index actually read, not on one particular iris constant — which of
            //the two is larger depends on the eye mapping above.
            if (landmarks == null || landmarks.Count <= Mathf.Max(LeftIrisCenter, RightIrisCenter))
            {
                dest[start] = dest[start + 1] = dest[start + 2] = dest[start + 3] = 0f;
                return;
            }

            FillOneEyeIrisOffset(landmarks[LeftEyeInnerCorner], landmarks[LeftEyeOuterCorner],
                landmarks[LeftIrisCenter], dest, start);
            FillOneEyeIrisOffset(landmarks[RightEyeOuterCorner], landmarks[RightEyeInnerCorner],
                landmarks[RightIrisCenter], dest, start + 2);
        }

        private static void FillOneEyeIrisOffset(NormalizedLandmark cornerA, NormalizedLandmark cornerB,
            NormalizedLandmark iris, float[] dest, int index)
        {
            float midX = (cornerA.X + cornerB.X) * 0.5f;
            float midY = (cornerA.Y + cornerB.Y) * 0.5f;
            float dx = cornerB.X - cornerA.X;
            float dy = cornerB.Y - cornerA.Y;
            float cornerDistance = Mathf.Sqrt(dx * dx + dy * dy);
            if (cornerDistance < 1e-5f)
            {
                dest[index] = dest[index + 1] = 0f;
                return;
            }
            dest[index] = (iris.X - midX) / cornerDistance;
            dest[index + 1] = (iris.Y - midY) / cornerDistance;
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
