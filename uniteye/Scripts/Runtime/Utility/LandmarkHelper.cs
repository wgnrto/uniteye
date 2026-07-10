// Excluded from WebGL player builds: depends on the native MediaPipe plugin (Mediapipe.Runtime
// has no wasm library, so IL2CPP linking fails). Kept for the Editor regardless of build target.
#if !UNITY_WEBGL || UNITY_EDITOR
using Mediapipe;
using System.Collections.Generic;
namespace UnitEye
{

    public static class LandmarkHelper
    {
        public static float[] GetEyeCorners(IList<NormalizedLandmark> landmarks)
        {
            var leftUpperCorner = landmarks[263];
            var leftLowerCorner = landmarks[362];
            var rightLowerCorner = landmarks[33];
            var rightUpperCorner = landmarks[133];

            return new float[] { leftUpperCorner.X, leftUpperCorner.Y, leftLowerCorner.X, leftUpperCorner.Y,
                                 rightLowerCorner.X, rightLowerCorner.Y, rightUpperCorner.X, rightUpperCorner.Y };
        }
    }
}
#endif
