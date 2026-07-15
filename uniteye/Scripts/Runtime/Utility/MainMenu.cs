using UnityEngine;
using UnityEngine.SceneManagement;
namespace UnitEye
{

    // Legacy menu helper. The standalone HolisticBarracuda-era scenes (GazeScene, GazeGame,
    // GazeEvaluation, GazeMainMenu) were removed in the Barracuda -> Inference Engine migration, so the
    // loaders that pointed at them would just throw "scene not in build settings". The two that map to a
    // surviving scene are repointed; the rest are kept as no-ops (with a warning) so existing UI wiring
    // doesn't NRE. Any target scene must still be added to Build Settings by the consuming project.
    public class MainMenu : MonoBehaviour
    {
        public static void LoadCalibrationScene()
        {
            SceneManager.LoadScene("HomulerGazeCalibration");
        }

        public static void LoadGazeScene()
        {
            SceneManager.LoadScene("HomulerGazeScene");
        }

        public static void LoadGazeGameScene()
        {
            Debug.LogWarning("MainMenu: the standalone GazeGame scene was removed in the migration. Add GazeGame to HomulerGazeScene instead (see README).");
        }

        public static void LoadGazeEvaluationScene()
        {
            Debug.LogWarning("MainMenu: the standalone GazeEvaluation scene was removed in the migration. Evaluation now runs inside HomulerGazeScene via the Gaze UI.");
        }

        public static void LoadMainMenuScene()
        {
            Debug.LogWarning("MainMenu: the standalone menu scene was removed in the migration; there is no menu scene to load.");
        }

        public static void QuitApplication()
        {
            UnitEye.Functions.Quit();
        }
    }
}
