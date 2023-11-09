using Mediapipe;
using System.Collections.Generic;

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