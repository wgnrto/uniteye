using UnityEngine;

public class MainMenuButton : MonoBehaviour
{
    private void OnGUI()
    {
        //MainMenu Button
        if (GUI.Button(new Rect(Screen.width * 0.95f - Screen.height * 0.05f, Screen.height - Screen.height * 0.1f, Screen.width * 0.05f, Screen.height * 0.05f), $"Main Menu"))
            MainMenu.LoadMainMenuScene();
    }
}
