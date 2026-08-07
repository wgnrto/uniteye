using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// Whether the pixels UnitEye measures in are the pixels <see cref="Screen.dpi"/> describes.
    ///
    /// The whole pipeline works in <c>Screen.width</c>/<c>Screen.height</c>, which is the RENDER SURFACE —
    /// in the Editor that is the Game view, not the monitor. Everything stays internally consistent in those
    /// units (targets, gaze, the normalized labels in a recording), so a calibration is perfectly valid for
    /// the viewport it was made in.
    ///
    /// What is NOT valid is the conversion to physical units. <c>Functions.PixelsToMm</c> is
    /// <c>pixels * 25.4 / Screen.dpi</c>, which silently assumes one render pixel covers one physical pixel.
    /// Run a 1920x1080 Game view inside a 900px-wide panel and every centimetre figure — the reported RMSE,
    /// the accuracy verdict, <c>LastHoldoutRmseCm</c>, a recording's screen size, and any degrees-of-visual-
    /// angle computed from it — is wrong by that ratio, with nothing to indicate it.
    /// </summary>
    public static class ScreenGeometry
    {
        /// <summary>Display resolution as the OS reports it (in the Editor: the desktop, not the Game view).</summary>
        public static int DisplayWidth => Screen.currentResolution.width;
        public static int DisplayHeight => Screen.currentResolution.height;

        /// <summary>
        /// True when the render surface matches the reported display resolution, i.e. the pixel->mm
        /// conversion is at least plausible.
        ///
        /// Deliberately named for what it checks rather than what one wishes it proved. It does NOT
        /// guarantee physical accuracy: a fullscreen build at a non-native resolution passes this while
        /// still stretching render pixels across the panel, and <c>Screen.dpi</c> is itself frequently wrong
        /// (many Windows setups report a flat 96) — which is why the calibration screen has always said so.
        /// It catches the common, silent case: an Editor Game view that is not the whole screen.
        /// </summary>
        public static bool RenderMatchesDisplay =>
            Screen.width == DisplayWidth && Screen.height == DisplayHeight;

        /// <summary>
        /// A warning for the user, or empty when nothing looks off. Kept here rather than at the call sites
        /// so the calibration, the recorder and any future consumer describe the same condition identically.
        /// </summary>
        public static string PhysicalScaleWarning()
        {
            if (RenderMatchesDisplay && !Application.isEditor)
                return "";

            var where = Application.isEditor ? "the Game view" : "the window";
            return $"UnitEye: {where} is {Screen.width}x{Screen.height} but the display reports " +
                   $"{DisplayWidth}x{DisplayHeight}. Calibration and gaze stay correct in these units, but " +
                   "anything in CENTIMETRES (the reported RMSE, the accuracy verdict, and a recording's " +
                   "screen size and degrees-of-visual-angle) assumes one render pixel is one physical pixel " +
                   "and will be wrong by that ratio. For measurements you intend to compare or donate, run a " +
                   "fullscreen build, or a Game view at scale 1x with Maximize On Play.";
        }
    }
}
