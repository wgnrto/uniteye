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

            //Calculation similar to EyeMU approach.
            //KNOWN QUIRK (kept deliberately): eyeLength/yShift are WIDTH-normalized but yShift is applied
            //to the HEIGHT-normalized y coordinate, so the eye's vertical placement inside the crop varies
            //with the camera aspect ratio (at 16:9 the eye sits ~24% from the crop top — the framing the
            //browser pipeline verified and the shipped calibrations were trained against). Constant within
            //a session -> absorbed by calibration; "fixing" it would silently change EyeMU's input framing
            //for every existing calibration, so any change must ship together with a forced recalibration
            //and a webcam hand-test, plus the same change in webgl/uniteye-core.js eyeCropRect.
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

        // ---------------------------------------------------------------------------------------------
        // Shared "context" feature block appended to every backbone's calibration vector. Layout (17):
        //   [tx, ty, dist, eyeLook x8, gazeA·headYaw, gazeB·headPitch, gazeA·tx, gazeB·ty, gazeA·dist,
        //    gazeB·dist]
        // tx/ty/dist are the metric head translation + depth (transformation matrix, in metres; falls
        // back to the face-bbox centre offset with dist 0 when the matrix is unavailable) — they close the
        // "lateral head shift is invisible" gap. The eyeLook blendshapes are a second, independently
        // trained gaze estimate. The interaction terms give the LINEAR ridge the multiplicative structure
        // of the physical map (screen_x ≈ eyePos_x + D·tan(yaw + headYaw)): without them the head-rotation
        // calibration stage collects variance the per-axis linear-in-pose model family cannot exploit
        // beyond an additive shift. gazeA/gazeB are the backbone's primary horizontal/vertical gaze terms
        // (yaw/pitch for the direction models, the normalized point for EyeMU).
        // ---------------------------------------------------------------------------------------------

        /// <summary>Number of features in the shared context block.</summary>
        public const int ContextFeatureCount = 17;
        /// <summary>Number of tail slots the context source values occupy (tx, ty, dist, 8 eyeLook).</summary>
        public const int ContextTailCount = 11;

        /// <summary>
        /// Snapshots the context source values (head translation, depth, eyeLook blendshapes) from the
        /// face mesh into a feature TAIL at <paramref name="start"/> (11 slots). Runners snapshot tails at
        /// inference-schedule time so async readback publishes internally consistent vectors.
        /// </summary>
        public static void FillTailContext(FaceMeshSolution faceMesh, float[] dest, int start)
        {
            if (faceMesh != null && faceMesh.HasTransformMatrix)
            {
                var t = faceMesh.HeadTranslation;      // canonical-face cm, camera space
                dest[start] = t.x * 0.01f;             // metres — keeps magnitudes in a sane range
                dest[start + 1] = t.y * 0.01f;
                dest[start + 2] = Mathf.Abs(t.z) * 0.01f;
            }
            else if (faceMesh != null && faceMesh.FaceLandmarks != null)
            {
                //Fallback: normalized face-bbox centre offset (proportional to lateral translation), no depth.
                var bounds = faceMesh.FaceBoundsNormalized;
                dest[start] = bounds.center.x - 0.5f;
                dest[start + 1] = bounds.center.y - 0.5f;
                dest[start + 2] = 0f;
            }
            else
            {
                dest[start] = dest[start + 1] = dest[start + 2] = 0f;
            }

            var eyeLook = faceMesh != null && faceMesh.HasBlendshapes ? faceMesh.EyeLookBlendshapes : null;
            for (int i = 0; i < 8; i++)
                dest[start + 3 + i] = eyeLook != null ? eyeLook[i] : 0f;
        }

        /// <summary>
        /// Fills the 17-feature context block at <paramref name="start"/> of the calibration vector from a
        /// tail whose context source values begin at <paramref name="tailContextStart"/> (see
        /// FillTailContext). gazeA/gazeB are the backbone's primary gaze terms; headYaw/headPitch come from
        /// the same tail snapshot as everything else so the products are single-frame consistent.
        /// </summary>
        public static void FillContextFeatures(float[] f, int start, float gazeA, float gazeB,
            float headYaw, float headPitch, float[] tail, int tailContextStart)
        {
            float tx = tail[tailContextStart];
            float ty = tail[tailContextStart + 1];
            float dist = tail[tailContextStart + 2];
            f[start] = tx;
            f[start + 1] = ty;
            f[start + 2] = dist;
            for (int i = 0; i < 8; i++)
                f[start + 3 + i] = tail[tailContextStart + 3 + i];
            f[start + 11] = gazeA * headYaw;
            f[start + 12] = gazeB * headPitch;
            f[start + 13] = gazeA * tx;
            f[start + 14] = gazeB * ty;
            f[start + 15] = gazeA * dist;
            f[start + 16] = gazeB * dist;
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
