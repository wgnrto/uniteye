namespace UnitEye
{
    /// <summary>
    /// How much of a calibration session may be written to disk, ordered least- to most-identifying.
    /// Every tier includes everything below it. The ladder exists because "record the calibration" spans a
    /// huge privacy range — a 36-value feature vector and a video of someone's living room are not the same
    /// ask — and a participant can only give meaningful consent to a specific point on it.
    ///
    /// The numeric values are persisted in consent.json and session.json; do not renumber them.
    /// </summary>
    public enum GazeRecordingTier
    {
        /// <summary>Record nothing. The default, and what a participant who declines gets.</summary>
        Off = 0,

        /// <summary>
        /// The backbone's feature vector + the on-screen label + per-sample quality flags. No imagery and no
        /// face geometry. Enough to retrain or re-benchmark the CALIBRATION HEAD (ridge / MLP), which is what
        /// the existing evaluation measures — but not enough to retrain a gaze BACKBONE.
        /// </summary>
        Features = 1,

        /// <summary>
        /// Adds the 478 MediaPipe face landmarks, the eye blendshapes and head pose. Note for consent
        /// wording: this is a 3D face template, i.e. still biometric data, even though it contains no picture.
        /// ~1434 floats/sample against Features' ~36, so it is also ~40x the size.
        /// </summary>
        Landmarks = 2,

        /// <summary>
        /// Adds the two 128x128 eye crops the gaze model actually consumes. The highest value-per-byte tier
        /// for improving gaze accuracy, and periocular rather than whole-face: eye, lid, brow edge, some
        /// cheek, and glasses if worn. See GazeSessionRecorder for the crop's geometry caveats.
        /// </summary>
        EyeCrops = 3,

        /// <summary>
        /// Adds frames cropped to the face bounding box. Shows the whole face but excludes the room and
        /// anyone walking through it — strictly preferable to FullFrames unless the background is genuinely
        /// needed, because no consent wording can un-record a bystander.
        /// </summary>
        FaceVideo = 4,

        /// <summary>
        /// Adds the entire camera frame: face, room, and whoever else is in view. The most sensitive tier;
        /// the consent flow requires a second, separate confirmation against a live preview before selecting
        /// it, because participants routinely forget what is behind them.
        /// </summary>
        FullFrames = 5,
    }
}
