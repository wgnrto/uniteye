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

        /// <summary>yakhyo/gaze-estimation MobileOne-s0: a face crop -> gaze pitch/yaw (fast), mapped to
        /// the screen. Uses Resources/ONNX/GazeEstimation/mobileone_s0_gaze.onnx.</summary>
        GazeMobileOne,

        /// <summary>yakhyo/gaze-estimation MobileNetV2: a face crop -> gaze pitch/yaw, mapped to the
        /// screen. Uses Resources/ONNX/GazeEstimation/mobilenetv2_gaze.onnx.</summary>
        GazeMobileNetV2,

        /// <summary>yakhyo/gaze-estimation (uniface "MobileGaze") ResNet-34: a face crop -> gaze
        /// pitch/yaw. Larger + more accurate than the mobile nets (uniface's default), same 448x448 /
        /// 90-bin I/O. Uses Resources/ONNX/GazeEstimation/resnet34_gaze.onnx.</summary>
        GazeResNet34,
    }
}
