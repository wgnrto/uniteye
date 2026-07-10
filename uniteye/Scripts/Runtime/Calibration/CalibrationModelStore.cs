using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// Owns the loaded calibration model(s) — the RidgeRegression X/Y pair or the SimpleMLP — and turns a
    /// raw EyeMU gaze estimate into a calibrated one. Extracted from HomulerGaze so the pipeline component
    /// isn't also responsible for calibration-model loading and Predict dispatch (part of splitting that
    /// god class). The numeric behaviour is unchanged from the old HomulerGaze.RefineGazeLocation.
    /// </summary>
    public class CalibrationModelStore
    {
        private RidgeRegression _xModel, _yModel;
        private SimpleMLP _mlp;

        /// <summary>
        /// (Re)loads the model(s) for the given calibration type. RidgeRegression.Load / SimpleMLP.Load
        /// already handle the expected "no calibration file yet" case (they return null and the pipeline
        /// falls back to raw gaze); only genuinely unexpected failures — a corrupt/incompatible JSON, an
        /// IO error — are surfaced here instead of being silently swallowed.
        /// </summary>
        public void Load(Calibrations calibrations)
        {
            try
            {
                switch (calibrations)
                {
                    case Calibrations.RidgeRegression:
                        _xModel = RidgeRegression.LoadX("Reg_X.json");
                        _yModel = RidgeRegression.LoadY("Reg_Y.json");
                        break;
                    case Calibrations.MLCalibration:
                        _mlp = SimpleMLP.Load("MLP.json");
                        break;
                }
            }
            catch (System.Exception e)
            {
                UnitEyeLog.Error($"Failed to load the {calibrations} calibration model; falling back to raw gaze.");
                UnitEyeLog.Exception(e);
            }
        }

        /// <summary>
        /// Applies the calibration model for <paramref name="calibrations"/> to the raw gaze. Falls back to
        /// the raw gaze when there is no feature vector, no loaded model, or the model returns NaN (a
        /// model/feature dimensionality mismatch).
        /// </summary>
        public Vector2 Refine(Vector2 rawGaze, Calibrations calibrations, float[] features, int screenWidth, int screenHeight)
        {
            //No feature vector (e.g. a browser provider streaming only raw gaze) -> calibration cannot
            //apply, use the raw gaze location directly
            if (features == null || features.Length == 0)
                return rawGaze;

            Vector2 refinedGaze = Vector2.zero;
            switch (calibrations)
            {
                case Calibrations.None:
                    refinedGaze = rawGaze;
                    break;
                case Calibrations.RidgeRegression:
                    //Fall back to the raw gaze location if no calibration model is loaded
                    if (_xModel == null || _yModel == null)
                        return rawGaze;
                    refinedGaze.x = _xModel.Predict(features) * screenWidth;
                    refinedGaze.y = _yModel.Predict(features) * screenHeight;
                    break;
                case Calibrations.MLCalibration:
                    //Fall back to the raw gaze location if no calibration model is loaded
                    if (_mlp == null)
                        return rawGaze;
                    refinedGaze = _mlp.Predict(features);
                    break;
            }

            //If calibration produced no usable value (NaN, e.g. model/feature mismatch), fall back to raw gaze
            if (float.IsNaN(refinedGaze.x) || float.IsNaN(refinedGaze.y))
                return rawGaze;

            return refinedGaze;
        }
    }
}
