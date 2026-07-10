namespace UnitEye
{
    /// <summary>
    /// Which gaze-estimation model the native provider runs. See docs/GAZE-BACKBONES.md.
    /// </summary>
    public enum GazeBackbone
    {
        /// <summary>EyeMU (FIGLAB, CHI 2021): eye crops + corners + head geometry -> screen point. The
        /// verified default; ships with the package.</summary>
        EyeMU,

        /// <summary>A direction-based model (e.g. yakhyo/gaze-estimation): a face crop -> gaze pitch/yaw,
        /// mapped to the screen. Requires you to add the ONNX model and hand-test — see the doc.</summary>
        GazeEstimation,
    }
}
