using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// OPTIONAL capability a gaze provider may implement to expose the extra per-frame data a dataset
    /// recording needs (landmarks, blendshapes, camera imagery). Deliberately separate from IGazeProvider:
    /// recording is not part of producing a gaze estimate, WebGLGazeProvider has none of this to give, and a
    /// future provider should not have to stub six members it will never use. Consumers obtain it with
    /// <c>provider as IGazeRecordingSource</c> and fall back to the feature-only tier when that is null.
    ///
    /// Every Try* method follows the same contract as IGazeProvider.GetFeatures(): the source owns reused
    /// per-frame buffers, so values are copied into a caller-owned destination rather than handed out by
    /// reference. Returning a reference here would silently alias the live frame — MediaPipe mutates its 478
    /// landmark objects in place every frame, so a retained list shows the LATEST frame, not the recorded one.
    /// </summary>
    public interface IGazeRecordingSource
    {
        /// <summary>
        /// Whether the imagery below belongs to the SAME camera frame as the current RawGaze/GetFeatures().
        /// False under async GPU readback, where the backbone publishes frame N-1's gaze while the crop
        /// textures already hold frame N — recording both would pair pixels with someone else's label, and
        /// nothing downstream could detect it. The recorder refuses imagery tiers when this is false rather
        /// than writing a plausible-looking but mislabeled dataset.
        /// </summary>
        bool ImageryInSyncWithFeatures { get; }

        /// <summary>Landmarks available this frame (478 for MediaPipe FaceMesh), or 0 if none.</summary>
        int LandmarkCount { get; }

        /// <summary>
        /// Copies landmarks as consecutive x,y,z triplets into <paramref name="dest"/>, which must hold at
        /// least LandmarkCount*3 floats. Returns the number of floats written, 0 if unavailable.
        /// Coordinates are MediaPipe-normalized (0..1, y-down, top-left) relative to the CAMERA FRAME —
        /// denormalize with FrameWidth/FrameHeight, never with Screen.width/height.
        /// </summary>
        int TryCopyLandmarks(float[] dest);

        /// <summary>
        /// Copies the 8 eyeLook* scores followed by eyeBlinkLeft/eyeBlinkRight (10 floats) into
        /// <paramref name="dest"/>. Returns false when this frame carried no blendshapes — in which case
        /// dest is left untouched and the caller must OMIT the field rather than write zeros. The underlying
        /// buffer is NOT cleared when blendshapes drop out; it keeps the last successful frame's values, so
        /// an unconditional copy silently writes stale measurements into the dataset.
        /// </summary>
        bool TryCopyEyeBlendshapes(float[] dest);

        /// <summary>Camera frame dimensions the landmarks are normalized against; 0 when no camera.</summary>
        int FrameWidth { get; }
        int FrameHeight { get; }

        /// <summary>The live camera texture, or null. Valid only for the current frame; blit, do not retain.</summary>
        Texture CameraTexture { get; }

        /// <summary>Face bounding box in the same normalized space as the landmarks; zero rect if no face.</summary>
        Rect FaceBoundsNormalized { get; }

        /// <summary>
        /// Whether the 6 gaze landmarks (4 eye corners + 2 iris centres) were One-Euro smoothed while the
        /// other 472 are raw. A real caveat for anyone training on the landmark block, so it is recorded.
        /// </summary>
        bool LandmarksSmoothed { get; }

        /// <summary>
        /// The flips applied when handing the camera frame to MediaPipe. Recorded, not corrected: landmark
        /// -to-pixel registration depends on them, and silently "fixing" the orientation is how a published
        /// dataset ends up with points that do not sit on the face.
        /// </summary>
        bool FrameFlippedHorizontally { get; }
        bool FrameFlippedVertically { get; }
    }
}
