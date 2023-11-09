using UnityEngine;
using UnityEngine.SceneManagement;

public class MainMenu : MonoBehaviour
{
    public static void LoadCalibrationScene()
    {
        SceneManager.LoadScene("GazeCalibration");
    }

    public static void LoadGazeScene()
    {
        SceneManager.LoadScene("GazeScene");
    }

    public static void LoadGazeGameScene()
    {
        SceneManager.LoadScene("GazeGame");
    }

    public static void LoadGazeEvaluationScene()
    {
        SceneManager.LoadScene("GazeEvaluation");
    }

    public static void LoadMainMenuScene()
    {
        SceneManager.LoadScene("GazeMainMenu");
    }

    public static void QuitApplication()
    {
        UnitEye.Functions.Quit();
    }
}
