# UnitEye: Introducing a User-Friendly Plugin to Democratize Eye Tracking Technology in Unity Environments ([MuC '24]([https://muc2024.mensch-und-computer.de/en/]))

 [Tobias Wagner*](https://scholar.google.de/citations?user=uqCJ2qsAAAAJ&hl=de&oi=ao), [Mark Colley*](https://scholar.google.de/citations?user=Kt5I7wYAAAAJ&hl=de&oi=ao), Daniel Breckel, Michael Kösel, Enrico Rukzio (*=equal contribution)

Full paper; doi: [10.1145/3670653.3670655](https://dl.acm.org/doi/10.1145/3670653.3670655)

Webcam-based eye-tracking for Unity


## Our Features
* Easy-to-use webcam-based eye tracker for Unity projects that require no specialized hardware other than a webcam
* Filtering of the gaze location to obtain more stable results
* Calibration to make it work for your setup
* Evaluation to test the accuracy
* Area of interest system to designate areas or objects on the screen to track
* CSV logging to log all the data
* Distance to the camera, blinking, and drowsiness detection
* Built-in GUI for runtime configuration
* Easy API to get started quickly

> :warning: **Known Issues**
> * Your GPU might be incompatible with our eye tracking pipeline and Direct3D11, to fix this follow our [Graphics API Troubleshooting](#graphics-api-troubleshooting).
> * **Barracuda → Inference Engine migration:** the pipeline now runs the EyeMU model on Unity's [Inference Engine](https://docs.unity3d.com/Packages/com.unity.ai.inference@latest) and gets its face-mesh landmarks from the native [MediaPipe Unity Plugin](https://github.com/homuler/MediaPipeUnityPlugin) instead of Barracuda/HolisticBarracuda. The primary component/scene is now `HomulerGaze` / `HomulerGazeScene` (the `UnitEyeUsingHomulerMediapipe` prefab); the old HolisticBarracuda `Gaze` component, its scenes and `UnitEye.prefab` were removed. The native plugin has no WebGL build, so on **WebGL** the computer vision runs in the **browser** instead: `webgl/` contains a JavaScript pipeline (MediaPipe FaceLandmarker + EyeMU on onnxruntime-web) bridged into Unity via `WebGLGazeProvider` — Unity WebGL builds are verified on Unity 6.3 and 6.5 (see `docs/WEBGL.md` and `webgl/README.md`). The model's names/shapes are verified to match under Inference Engine, but the eye-crop pixel normalization and RGB/BGR channel order could not be verified headlessly — confirm gaze accuracy at runtime with a webcam. The `HomulerGazeCalibration` scene still references five landmark-annotation scripts that were never committed and logs missing-script warnings when opened (they are not needed for tracking).
> * Since we use a webcam and no infrared lighting, accuracy heavily depends on the quality of your webcam and the lighting in your room. A dark room and a cheap 720p webcam will most likely not result in usable accuracy. The upside of our approach is that we are fairly resistant to glasses and contact lenses, which is an issue for infrared-based approaches.
> * The eye tracking performance degrades when you're not in the center of the webcam image. This is due to our underlying neural network from [EyeMU](https://github.com/FIGLAB/EyeMU) being very sensitive to positional changes and probably also due to optical distortion towards the edges. This can partly be mitigated by moving your head around while calibrating.
> * The current [EyeMU](https://github.com/FIGLAB/EyeMU) model is unable to generate gaze locations outside of the screen (it is, however, shifted to the bottom right for us, resulting in pseudo off-screen values). We do include an 'off-screen' AOI in our Gaze component, but it mostly doesn't do much. This is probably due to model structure and/or insufficient training.
> * The tracking in the middle of the screen can at times be inaccurate, this is probably due to insufficient training of the [EyeMU](https://github.com/FIGLAB/EyeMU) model across a wide range of setups and webcams.
> * Since this is as first version, the accuracy is not up to par with commercial infrared-based eye trackers. However, during our testing, if the room (mainly the face) is well lit through frontal lighting and the distance and position doesn't change a lot from the calibration, we've seen RMSEs of 2.5cm x and y on a 24" monitor with a distance of 60cm to the webcam (Logitech C920 running at 960x540 resolution). This equates to a visual angle of ~2.4°.
> * We use the face mesh from the native [MediaPipe Unity Plugin](https://github.com/homuler/MediaPipeUnityPlugin) to get the locations for our eye crops that we feed into the [EyeMU](https://github.com/FIGLAB/EyeMU) NN. This runs Google's reference MediaPipe FaceMesh (468 landmarks + iris) natively, which should give more stable eye crops than the previous HolisticBarracuda (Barracuda .onnx) approach, though heavy filtering of the gaze location is still recommended.
> * The current eye tracking pipeline runs blocking synchronously, meaning the frame rate of a project can only be as high as our neural network pipeline runs at. This was 40 FPS in Editor and 70 FPS in a build on a system with an i7 3820 (ancient), 16GB of DDR3 RAM (also ancient) and an AMD Vega56 GPU.
> * We've attempted to run the eye tracker on Android, however the performance was quite lackluster and Unity handles the front camera on Android in a weird way which would require us to rewrite parts of the eye tracking pipeline to handle camera rotation. This was deemed out of scope for the current project.
> * During development, we encountered a bug on one of our systems where the webcam (namely a **Logitech C920**) would only deliver around 1-2 fps when the requested resolution was set to the full 1080p. This was not reproduced on other systems and appears to be a fairly old bug in [Unity](https://answers.unity.com/questions/1426135/hd-webcam-is-slow-in-pc.html). Upgrading to newer Unity versions did not fix the bug. We suspect this is a rare bug that only happens with Logitech webcams (that do not have USB 3.0) and is out of our control.

## Used Sources and Libraries
* [Unity Inference Engine](https://docs.unity3d.com/Packages/com.unity.ai.inference@latest) (`com.unity.ai.inference`, formerly Sentis, the successor to the now-deprecated Barracuda), Unity's neural-network inference library based around the [.onnx](https://onnx.ai/) file type. UnitEye runs the [EyeMU](https://github.com/FIGLAB/EyeMU) gaze model on it. **Note:** as of the Barracuda→Inference Engine migration, the old HolisticBarracuda landmark pipeline and Barracuda itself have been removed (Barracuda and the Inference Engine cannot coexist — both register an importer for `.onnx`).
* [MediaPipe Unity Plugin](https://github.com/homuler/MediaPipeUnityPlugin) (homuler), which runs Google's native MediaPipe FaceMesh (468 landmarks + iris). This replaced HolisticBarracuda as the eye-crop landmark source, so the pipeline no longer depends on Barracuda. It uses native binaries (Windows/macOS/Linux/Android) and therefore **does not support WebGL**.
* [EyeMU](https://github.com/FIGLAB/EyeMU), we use their model (`.onnx`) for our eye tracking pipeline
* Other smaller code sources are referenced in code comments

(The former [BrightWire](https://github.com/jdermody/brightwire-v2) dependency was removed: the `ML Calibration` MLP is now a small dependency-free C# implementation — same 12→32→16→2 architecture — which trains faster and, with proper input standardization and honest holdout evaluation, more accurately than the BrightWire version it replaces.)

## Quick Demo
Create a new (empty 3D) Unity project with Unity 6.3 LTS (6000.3), go through [Installation](#installation) to add our packages, and open the `HomulerGazeScene` scene from the UnitEye package. [Getting Started](#getting-started) will walk you through the features.

## Graphics API Troubleshooting
If the eye tracking isn't working at all, your GPU is most likely incompatible with our eye tracking pipeline in [Barracuda](https://docs.unity3d.com/Packages/com.unity.barracuda@2.0/manual/index.html) when using the Direct3D11 Graphics API. To fix this go through the following steps:

![](./uniteye/Documentation~/Images/GraphicsAPI.png)

Open your Project Settings window (`Edit` -> `Project Settings`). Select the `Player` settings, scroll down and open the `Other Settings` for the Windows, Mac and Linux player. Untick the `Auto Graphics API for Windows` checkbox (1), add either `Vulkan` or `OpenGLCore` through the `+` menu (2 and 3) and rearrange them so they are above `Direct3D11` in the list (4). Unity will then ask you to reopen the project. Afterward, the eye tracking should be working. We recommend testing between the two alternative APIs to find the one with the better performance. This bug isn't mentioned in the Unity Barracuda [documentation](https://docs.unity3d.com/Packages/com.unity.barracuda@2.0/manual/SupportedPlatforms.html) for Windows.

## Installation
UnitEye targets Unity 6.3 LTS (6000.3) and is also verified on Unity 6.5 (6000.5) — the full editor test suite passes on both (`unity` in `package.json` declares the 6000.3 minimum). Since the Barracuda→Inference Engine migration you only need **two** package folders: `uniteye` and `com.github.homuler.mediapipe`. Copy them from the repository root into your own project's `Packages/` folder (they become embedded packages), or reference them as local packages via `file:` entries in your `Packages/manifest.json`.

The only registry dependency is Unity's Inference Engine (`com.unity.ai.inference`), which resolves from Unity's own package registry — no scoped registries are needed anymore (HolisticBarracuda and the `jp.keijiro`/`jp.ikep` scoped registries are gone). Your manifest's `dependencies` should include:
```json
{
    "dependencies": {
        "de.uniulm.uniteye": "file:../../path/to/uniteye",
        "com.github.homuler.mediapipe": "file:../../path/to/com.github.homuler.mediapipe",
        "com.unity.ai.inference": "2.6.1"
    }
}
```

### Required: install the MediaPipe model files
The native MediaPipe FaceMesh loads its model files (`.bytes`) from your project's `StreamingAssets` folder at runtime, and on a desktop build it will throw a `FileNotFoundException` (and produce no gaze) if they are missing. After adding the packages, run **`UnitEye ▸ Install MediaPipe StreamingAssets`** from the Unity menu once — it copies the required models (`face_detection_short_range.bytes`, `face_landmark_with_attention.bytes`, `face_landmark.bytes`) from the homuler package into `Assets/StreamingAssets/`. (In CI you can run it headless with `-executeMethod MediaPipeAssetInstaller.Install`.) These files are then included automatically in your builds.

## Getting started

Once you're done with the [Installation](#installation), start up your Unity project and wait for the Package Manager to sort out the packages. After startup, you should have a window open automatically to inform you about the two newly added scoped registries. You're now ready to use UnitEye in your project! But how do you use it?

Included in the package are example scenes, the starting point is `HomulerGazeScene`. Use your project browser to navigate to `UnitEye/Scenes/` (the package's display name in the Project window). This scene includes our `UnitEyeUsingHomulerMediapipe` prefab with the `HomulerGaze` component (the main eye-tracking pipeline) and a `Mediapipe` GameObject that runs the native face-mesh. Let's have a look at the main pieces:

#### Web Cam Input: 
This component manages the connection between Unity and your webcam.

![](./uniteye/Documentation~/Images/WebCamInputInspector.png)

You can select your webcam by pressing the `Select` button, which opens a drop-down list with all the plugged-in webcams that Unity can see. Below that, you can select a desired resolution, and Unity will use the closest available resolution that your webcam can provide. The default here is 1080p. You can also set a maximum frame rate limit if you run into a bug where your webcam freezes when the application runs at several hundred frames per second.

Optional settings include a RawImage reference if you wish to have the webcam image drawn in your scene, a static input image for debugging purposes and a toggle box to mirror the image horizontally, depending on your webcam you may want to use this if you're looking at the top right corner but the webcam image is showing you looking to the top left. This mirroring is purely visual and does not influence the tracking.

#### Visualizer:
This is a component from the preexisting [HolisticBarracuda](https://github.com/creativeIKEP/HolisticBarracuda) package. We use a custom version of this package, so you shouldn't download it separately!

![](./uniteye/Documentation~/Images/VisuallizerInspector.png)

The main purpose of this component is to visualize all the tracking features of HolisticBarracuda, namely the face mesh, pose, and hand tracking. We currently only utilize the face mesh for our eye tracking, but you might want to use more features from HolisticBarracuda in your project! This component is disabled by default as its main purpose is debugging and will degrade the frame rate when enabled. Most settings should be left standard, but you can play around with the `Holistic Inference Type` to check out the different tracking options.

#### Gaze:
This is our main component that handles the entire eye-tracking pipeline.

![](./uniteye/Documentation~/Images/GazeInspector.png)

The first setting is a reference to the [Web Cam Input](#web-cam-input) component you wish to use. After that you can select a texture to use as the gaze location dot, we include a simple cross-hair in our package. If you wish to use our [CSV Logger](#csv-logger), simply reference the CSV Logger component from the scene, our prefab already has one included. Below that you have 4 toggle boxes, these can be used to toggle the gaze location dot, show or hide the eye crops, toggle the visualization of our [Area Of Interest](#area-of-interest) system and show the button to toggle our runtime [Gaze UI](#gaze-ui).

Moving on, you can select between our [Calibration](#calibration) types. We currently offer `None`, which only uses the raw neural network output of the underlying [EyeMU](https://github.com/FIGLAB/EyeMU) model, `Ridge Regression`, which uses a regularized linear regression to refine the gaze location and `ML Calibration` where we use our own machine learning multilayer perceptron that we train when calibrating. `Ridge Regression` is the default and recommended calibration: it is deterministic, works well with the amount of data a calibration produces, standardizes its input features, and its reported accuracy comes from a proper held-out test set with a cross-validated regularization strength. The `ML Calibration` MLP (a small dependency-free implementation) now uses the same input standardization and honest random-holdout evaluation; it can outperform ridge when the gaze mapping is genuinely nonlinear, at the cost of non-determinism and a few seconds of training. Calibration files from pre-2026 versions of the ML calibration use an incompatible format — simply recalibrate. The current gaze location in x and y pixels coordinates (with zero at the top left corner) is displayed below the calibration type.

The next setting is the filtering selection. We currently offer a [Kalman](https://en.wikipedia.org/wiki/Kalman_filter) filter, a simple Easing filter, which is a weighted sum filter between the last and the current gaze location, combinations of those two and a [One Euro](https://gery.casiez.net/1euro/) filter. Depending on the selected filter, you can also tinker with the relevant filter values to suit your needs. We recommend the One Euro filter as it offers the best smoothing performance while being quick when larger location changes happen, but you're welcome to try out the other options.

Finally, the Gaze component holds the last gaze location while you blink (`Hold Gaze During Blink`), because the eye crops during a blink are unreliable and would otherwise push a spike through calibration and filtering. The hold duration is capped (`Max Blink Hold Seconds`, default 0.5s) so a miscalibrated blinking threshold can never freeze the gaze location.

#### CSV Logger:
This is the last component in our prefab and handles our CSV logging. If you do not wish to use the CSV Logging, simply disable this component, and nothing will be logged!

![](./uniteye/Documentation~/Images/CSVLoggerInspector.png)

The first setting is the base filename you want to give to the .csv files we create. When you hit play in the Editor or run a built application, we automatically append the current timestamp to the filename. This is done to prevent accidental file overwriting when you have multiple runs. An example name would be `UnitEyeLog_20221021_162115.csv`, the formatting being `BaseName_YYYYMMDD_HHMMSS.csv`.

You have the choice to either use our default folder by ticking the checkbox. This equates to `/Assets/CSVLogs/` in the editor and `/ProjectName_Data/CSVLogs/` in a built application.
When you untick the checkbox, you can select your folder. Take care that this folder is accurate when using it in a build.

Our CSV Logger uses a write queue to aggregate data entries and flush them to a file every so often. The next text box is the time in seconds between queue flushes. If you only want to log our data Y times per second instead of once per frame, just specify the maximum number of entries per second in the last text box.

We log the following data in our .csv files:
* Filtered gaze location in pixel and normalized (to the screen size) values
* Unfiltered gaze location as normalized values
* Distance to the camera in millimeters
* Eye aspect ratio, can be used for drowsiness detection
* Blinking true/false
* Timestamp of the data entry in formatted time with milliseconds and Unix timestamp in milliseconds
* A list of all the areas of interest that were being looked at when the data entry was made
* Additional notes, currently used for runtime calibration and evaluation messages

#### Main Menu Button:
Our example scenes also include a main menu system to allow you to click through our scenes with ease. This component is just a simple GUI button that will load the `GazeMainMenu` scene from our project. If you wish to use this feature, you have to include all of the 5 scenes in `/Packages/UnitEye/Scenes/` in your editor build settings. However, this main menu is completely optional and not included in the prefab!

## Gaze UI
We offer a built-in GUI overlay which gives you access to some settings at runtime. When the `Show Gaze` UI button ` checkbox in a [Gaze](#gaze) component is ticked, you will see a button to toggle the GUI. When you click `Show Gaze UI` at runtime, you will see the following overlay:

<img src="./uniteye/Documentation~/Images/GazeUI.png" width="500" height="495">

This entire window is draggable with the bar at the top. You can go through all the webcams that are currently available in Unity using the `Webcam controls` section. Below that, you can `Toggle UI Overlays` similar to the Inspector settings in the [Gaze](#gaze) component. 

Moving on, you can calibrate our `Distance to camera` feature. This calibration is required when you first set up your webcam or change to a different webcam. To calibrate, simply sit so that your eyes are around 50cm away from the webcam and hit the `Calibrate Distance to Camera` button. The calibrated value in the background is saved in the PlayerPrefs of your Unity project/built app, so this calibration ideally only needs to be done once when changing webcams and persists through multiple runs.

After that, you can calibrate our `Blinking and Drowsiness` system. To calibrate the blinking threshold, close your eyes to whatever degree you want us to detect as blinking and click on the `Calibrate Blinking Threshold` button. The same applies to the drowsiness calibration. Close your eyes to the degree that you want us to detect as being drowsy and click on the `Calibrate Drowsiness Baseline` button. We then sample the eye aspect ratio of your eyes over the next 60 frames and calculate an average baseline to compare to. When your eyes are below this baseline (meaning your eyes are more shut) for a while (several seconds), we detect you as being drowsy. Both these calibrations are saved in PlayerPrefs as well.

The next section controls the used `Calibration type` and `Filtering type`. You can select between all of our currently offered calibration and filtering types and play with the relevant filter values for the current filter. The filter values are currently not being saved and will reset between runs!

Lastly, you have the option to start our `Calibration` and `Evaluation` sequence at runtime. These buttons will start a [Calibration](#calibration) or [Evaluation](#evaluation) without loading a new scene. When you start a sequence through the Gaze UI, you can cancel and return from it at any point by clicking the right mouse button.

> **Blurry Gaze UI text in the editor on a high-DPI display?** The overlay is drawn with Unity's on-screen IMGUI, which renders at the Game view's resolution and is then scaled to fit — so on a high-DPI screen (e.g. a 3200×2000 laptop) the text can look soft while the editor's own UI stays sharp. In the Game view, set the aspect to **Free Aspect** and enable **"Low Resolution Aspect Ratios"** (in the Game view's aspect/scale dropdown); this restored crisp text on such a display. A standalone build renders the overlay crisp regardless. The buttons themselves are unaffected by this — if a toggle appears not to respond in the editor, make sure you are running a build from this branch (an earlier bug made the toggles revert every frame after a calibration had run; it is fixed here).

## Calibration
Every room and computer setup needs calibration to ensure good tracking accuracy. You should also calibrate when your seating position changes drastically or when you notice a loss in accuracy (which can also be caused by a change in lighting). To do that you have two options:
* Load a new scene similar to our `UnitEye/Scenes/HomulerGazeCalibration` scene and add the `HomulerGazeCalibration` component (uniteye/Scripts/Runtime/HomulerGazeCalibration.cs) next to the `HomulerGaze` component
* Do a runtime calibration through our [Gaze UI](#gaze-ui). When you do this, we load the GazeCalibration component dynamically with mostly default settings and the currently selected calibration type. You also have the option to cancel and return with right click. The calibration accuracy will be included in the .csv file if you are logging data.

When you load a new scene similar to `GazeCalibration` and add the component, you have access to the following settings in the inspector:

![](./uniteye/Documentation~/Images/GazeCalibrationInspector.png)

The Points drop-down will give you a list of all the point coordinates for the current calibration round. After that you can select a texture to use as the calibration dot, we include a default CalibrationDot with our package. The speed that the dot moves can also be changed, as well as the padding around the edges in pixels. The current Round shows the current round when calibrating.

Max Rounds Per Preset sets the maximum rounds the calibration goes through based on how many pattern presets are included. Currently, the calibration runs through the corners clockwise, then a vertical zig-zag pattern, followed by a horizontal zig-zag pattern and a counterclockwise corner pattern. When Max Rounds is set to eg. 3, it will run through those four patterns sequentially and then repeat that 2 more times, for a total of 12 rounds. We recommend 2 Max Rounds Per Preset, but you are welcome to test and change the `_presets` in `UnitEye/Scripts/Runtime/GazeCalibration.cs`.

Rounding off, you can select the Calibration Type to calibrate against (in a runtime calibration, this is the current calibration type on the Gaze component), choose whether or not to save a (and potentially overwrite an existing) calibration file, choose whether or not the calibration stops after every point (to allow your eyes to relax and you to blink if needed), and choose whether you want to quit the application after the calibration is complete.

When you start a calibration sequence either through a new scene or at runtime through [Gaze UI](#gaze-ui), you will see the following screen:

![](./uniteye/Documentation~/Images/CalibrationScreen.png)

Once you left click with your mouse, the calibration will start and your task is to follow the calibration dot with your eyes. After each round, the calibration will pause, requiring you to press left click again to continue with the next round. As you can see we show ghost dots to visualize where the dot will be moving to. You can blink when the dot is stationary and between rounds, but try not to blink while the dot is moving! Frames during blinks, frames without a fresh webcam image and frames without a detected face are automatically excluded from the training data (`Skip Unreliable Samples` on the GazeCalibration component), so occasional blinks no longer poison the calibration.

When all the rounds are done, the training will start. If you want to stop the calibration early but still perform training press the `S` key. Depending on how many rounds you calibrate for and the speed you select, this can take quite a while (even several minutes on very long calibrations). Do not quit the application before the training is done with the corresponding message on the screen or the calibration file will not be saved! After the training is done we display the Root Mean Squared Error (RMSE) as a centimeter error based on your screen size. This is also included in the .csv file when you do a runtime calibration.

For the RidgeRegression calibration the reported RMSE is measured on a randomly held-out 20% of the samples, and the regularization strength is selected with 5-fold cross validation on the training portion only. This is an honest accuracy estimate, so it can read slightly higher than the numbers reported by older UnitEye versions, which evaluated on the chronologically first samples of the session.

Speaking of calibration files: eye tracking only works well once *you* calibrate. We no longer ship a default fit, because calibration is inherently per-person — the old defaults were trained on one person and extrapolated off-screen for everyone else (the gaze dot would get stuck in a screen corner). So if you start the eye tracker before calibrating, we log a one-time console warning and fall back to the **raw, uncalibrated gaze** from the model: it tracks your eyes roughly and stays on-screen, but it is not accurate until you calibrate. When you finish your calibration, we save the new files in `StreamingAssets/Calibration Files/` in subfolders for each calibration type. This means that when you calibrate in the editor and then build your project into an app, the calibration files will be included! (Tip: if you want to sanity-check tracking before calibrating, set the calibration type to `None` in the [Gaze UI](#gaze-ui) to view the raw gaze directly.)

## Evaluation
To test the accuracy of our eye tracker, we offer an evaluation sequence. The general procedure is similar to a [Calibration](#calibration), but instead of moving, the evaluation dot jumps between random points on the screen that are on a set grid. You once again have two options:
* Add the `HomulerGazeEvaluation` component (uniteye/Scripts/Runtime/HomulerGazeEvaluation.cs) next to the `HomulerGaze` component
* Do a runtime evaluation through our [Gaze UI](#gaze-ui). When you do this, we load the GazeEvaluation component dynamically with mostly default settings and the currently selected calibration type. You also have the option to cancel and return with right click. The evaluation accuracy will be included in the .csv file if you log data.

When you load a new scene similar to `GazeEvaluation` and add the component, you have access to the following settings in the inspector:

![](./uniteye/Documentation~/Images/GazeEvaluationInspector.png)

First, you can select a texture to use as the evaluation dot. Then you can choose the Duration in seconds, this being the time that the dot will appear at each location. We only use data from the middle 50% of the duration (so from 25% to 75% of the duration) for the evaluation to give the user time to find the new location. Padding around the edges in pixels and the pixel size of the dot are the next settings. You can then define the number of rows and columns for our point grid. Finally, choose if you want to display the potential points on the grid as ghost dots and whether or not to quit the app after evaluating.

We evaluate all calibration types other than none at the same time, so there's no need to specify the type.

When you start an evaluation sequence either through a new scene or at runtime through [Gaze UI](#gaze-ui), you will see the following screen (the screenshot includes ghost dots to visualize the grid, this is not the default setting!):

![](./uniteye/Documentation~/Images/EvaluationScreen.png)

Once you left-click with your mouse, the evaluation will start, and your task is to look at the evaluation dot. When the duration left hits 0, the dot will appear at a new random location on the grid and you should once again look at it. This will repeat until we've gone through `rows * columns` locations or until you press the `S` key to stop early. Afterward, we calculate the RMSE of each calibration type of our eye tracker and display it on the screen as a centimeter error based on your screen size. This is also included in the .csv file when you do a runtime evaluation.

## Area Of Interest
Our custom Area Of Interest (AOI) system allows you to do interesting things with our eye tracker. The [AOIManager](uniteye/Scripts/Runtime/AOI/AOIManager.cs) class handles all the AOIs we want to track through a List<AOI>, which contains all the currently tracked AOIs. The manager includes functions to add, remove, and get AOIs from the list.  If you want to add an AOI, you have to add it to the AOIManager in the [Gaze](#gaze) component. To do so, you need to have a reference to that AOIManager, our [GazeGameAPI.cs](uniteye/Scripts/Runtime/GazeGameAPI.cs) and [GazeGame.cs](uniteye/Scripts/Runtime/GazeGame.cs) include several ways to do so, either through the [UnitEyeAPI](#uniteyeapi), via reference or inheritance.

We currently offer 7 different shapes for you to use as you please which will be explained in detail in this chapter. [ExampleAOIs.cs](uniteye/Scripts/Runtime/AOI/ExampleAOIs.cs) contains examples for all of the AOI shapes.

The base class that all of our AOI shapes inherit from is [AOI.cs](uniteye/Scripts/Runtime/AOI/AOI.cs). This class has several relevant public fields:
* ```readonly string uID``` is a unique identifier, this should be a unique string, not shared between different AOI objects and is immutable after you create a new shape
* ```bool inverted``` if this is true, the AOI shape will essentially be inverted, ie. if the shape is a simple box the AOI will be focused if the user is looking anywhere but the box
* ```bool enabled``` if this is false, the AOI will not be considered by the AOI system, almost equal to disabling a component in the Unity inspector
* ```bool visualized``` if this is false, the AOI will be excluded from the AOI visualization system
* ```bool focused``` this bool tells you if the AOI is currently being looked at (true) or not (false)

All of the AOI shapes in [UnitEye/Scripts/Runtime/AOI/Shapes/](uniteye/Scripts/Runtime/AOI/Shapes/) include these fields, they are the main way to interact with the eye tracker. The AOI class also contains all the methods to check for point inclusion in the inheriting AOI shapes. The actual managing of the AOIs is done through the AOIManager, but this runs entirely in the background and doesn't need any setup.

> __Do note that all the following points are in normalized Vector2, `(0f,0f)` being the top left of the screen and `(1f,1f)` being the bottom right of the screen, this means they are influenced by the aspect ratio! They are almost identical to Unity's Vector2.__

We will now go through all the shapes and explain them:

#### AOIBox
[AOIBox](uniteye/Scripts/Runtime/AOI/Shapes/AOIBox.cs) is the simplest shape, a 2D box (who would have guessed!). Relevant fields are:
* ```Vector2 startpoint``` is the startpoint of the box, usually the top left corner
* ```Vector2 endpoint``` is the endpoint of the box, usually the bottom right corner

#### AOICircle
[AOICircle](uniteye/Scripts/Runtime/AOI/Shapes/AOICircle.cs) is a 2D circle. Relevant fields are:
* ```Vector2 center``` is the center point of the circle
* ```float radius``` is the radius in float, this is also influenced by the aspect ratio, meaning the circle might not look round unless your aspect ratio is 1:1

#### AOICapsule
[AOICapsule](uniteye/Scripts/Runtime/AOI/Shapes/AOICapsule.cs) is a 2D capsule. Relevant fields are:
* ```Vector2 startpoint``` is the startpoint of the capsule in the middle of the shaft
* ```Vector2 endpoint``` is the endpoint of the capsule in the middle of the shaft
* ```float radius``` is the radius in float, meaning the thickness of the middle box is `2 * radius` and the semicircles on each end have a radius of `radius`

#### AOICapsuleBox
[AOICapsuleBox](uniteye/Scripts/Runtime/AOI/Shapes/AOICapsuleBox.cs) is almost identical to an AOICapsule, the difference being that the semicircles on each end are removed, essentially creating a box that you can rotate. Relevant fields are:
* ```Vector2 startpoint``` is the startpoint of the capsule in the middle of the shaft
* ```Vector2 endpoint``` is the endpoint of the capsule in the middle of the shaft
* ```float radius``` is the radius in float, meaning the thickness of the box is `2 * radius` and the semicircles on each end have a radius of `radius`

#### AOIPolygon
[AOIPolygon](uniteye/Scripts/Runtime/AOI/Shapes/AOIPolygon.cs) is a 2D polygon. Relevant fields are:
* ```List<Vector2> points``` is a List of Vector2 points that define the polygon, the first entry being the first corner in the polygon, the last being the last

To use this shape you have to add Vector2 points with the ```AddPoint(Vector2 point)``` or ```InsertPoint(Vector2 point, int i)``` method and can remove them either with ```RemovePoint(Vector2 point)``` or ```RemoveAllPoints(Vector2 point)```. You can also make your ```List<Vector2>``` and construct the shape with that list included. Since this list is public you can do whatever you want with it. Currently, the AOIPolygon does not allow for holes (though it might still work depending on the shape) and only allows for straight lines between the points

#### AOICombined
[AOICombined](uniteye/Scripts/Runtime/AOI/Shapes/AOICombined.cs) is a container to combines multiple AOI shapes under one uID. Relevant fields are:
* ```private List<AOI> _aoiList``` is a List of AOIs

To use this shape you have to add AOI shapes with the ```AddAOI(AOI aoi)``` method and can remove them either with ```RemoveAOI(AOI aoi)``` or ```RemoveAOI(string uID)```. You can also make your own `List<AOI>` and construct the shape with that list included.

#### AOITagList
[AOITagList](uniteye/Scripts/Runtime/AOI/Shapes/AOITagList.cs) is a shape that allows you to interact with GameObjects in Unity. What it does is throw a RayCast into the scene at the gaze location and return hit objects that match predefined tags from a list. Relevant fields are:
* ```private List<string> _tagList``` is a list of tag names to match against
* ```List<string> hitNameList``` contains a list of the names of GameObjects with the correct tag that was hit
* ```RaycastHit hitRaycast``` contains the currently hit RaycastHit with matching tag when using xray == false
* ```List<RaycastHit> hitRaycastList``` contains a list of all hit RaycastHit with matching tag when using xray == true
* ```Camera camera``` default is Camera.main, if you use a custom camera in your scene you might want to overwrite this
* ```int maxNumberOfRaycastHits``` the maximum number of RaycastHit, 20 being default (mostly relevant for xray RayCasts)
* ```bool xray``` if this is true, the RayCast will go through Colliders until maxNumberOfRaycastHits is reached, if false, RayCast will stop at the first hit object
* ```int layerMask``` the layer mask to use for RayCasting, see [Physics.RayCast](https://docs.unity3d.com/ScriptReference/Physics.Raycast.html)
* ```QueryTriggerInteraction queryTriggerInteraction``` specifies whether the RayCast query should hit Triggers, see [Physics.RayCast](https://docs.unity3d.com/ScriptReference/Physics.Raycast.html), the default is Ignore

To use this shape you have to add tag names with the `AddTag(string tag)` method and can remove them with `RemoveTag(string tag)`, these tags being the `Tag` of the GameObject with a collider. You can also make your own tag `List<string>` and construct the shape with that list included. Do note that this AOI depends on Unity's RayCast functionality, so your GameObject can only be detected if it has a Collider component attached.

All these shapes can be used however you like. Also the positions are not static, meaning that you can change the centerpoint of an AOICircle at runtime and that will functionally move the Circle as far as our AOI system is concerned.

For a few examples of how you can create the shapes, take a look at [ExampleAOIs.cs](uniteye/Scripts/Runtime/AOI/ExampleAOIs.cs). Our example [Gaze Game](#gaze-game) utilizes to AOITagList to interact with GameObjects.

## UnitEyeAPI
We offer an API in [UnitEyeAPI.cs](uniteye/Scripts/Runtime/UnitEyeAPI.cs). This API allows you to access most of the relevant fields and functions of a [Gaze](#gaze) component by calling API functions. Since this script is well-documented and most functions are very short, we invite you to take a look at the code if you intend to use it!
The API is the preferred way to access UnitEye since it is the easiest option.
__Do make sure to include `using UnitEye;` at the top of the script you want to use our UnitEyeAPI!__

## Coarse Gaze Targets: GazeGridQuantizer
Webcam eye tracking is inherently jittery. If your application only needs coarse gaze regions (for example "which third of the screen is the user looking at"), do not consume the raw gaze location directly. Use the [GazeGridQuantizer](uniteye/Scripts/Runtime/Utility/GazeGridQuantizer.cs): it quantizes the gaze into a grid of cells and only switches cells after the gaze has clearly (hysteresis margin) and steadily (dwell time) settled in another cell. Sub-cell jitter disappears entirely and cells do not flicker at the borders, which makes the output far more stable than thresholding the raw location yourself.

```cs
using UnitEye;
using UnityEngine;

public class CoarseGazeExample : MonoBehaviour
{
    private GazeGridQuantizer _quantizer = new GazeGridQuantizer(columns: 3, rows: 3);

    void Update()
    {
        var gaze = UnitEyeAPI.GetGazeLocationInGUI();
        var normalized = new Vector2(gaze.x / Screen.width, gaze.y / Screen.height);

        if (_quantizer.Update(normalized, Time.unscaledTime))
            Debug.Log($"Now looking at cell {_quantizer.CurrentColumn}, {_quantizer.CurrentRow}");
    }
}
```

Both the hysteresis margin and the dwell time are constructor parameters, the defaults (0.15 cells, 0.1s) are a good starting point. Increase them for even more stability, decrease them for faster reactions.

## Making GameObjects Gaze Aware
One way to make GameObjects gaze-aware is by using the `AOIManager`` and a `AOITagList`. Alternatively, you can add the `Gazeable` script to a GameObject to allow it to be tracked.
Make sure to set its tag to something other than `Untagged`. Also as with the `AOITagList`, make sure your GameObject has a collider script attached.
Here is a small script highlighting how to use the `Gazeable` component:
```cs
using UnityEngine;
using UnitEye;
 
[RequireComponent(typeof(Gazeable))]
public class GazeableExample : MonoBehaviour
{
    private Gazeable _gazeable;
 
    void Start()
    {
        _gazeable = GetComponent<Gazeable>();
    }
 
    void Update()
    {
        if (_gazeable.HasGazeFocus)
        {
            // Object is being looked at 
        }
    }
}
```

## Gaze Game
A small demo where you can move GameObjects around by looking at them for more than 30 frames, then "let go" by blinking. [GazeGame.cs](uniteye/Scripts/Runtime/GazeGame.cs) shows the basics via a reference to the gaze component, and [GazeGameAPI.cs](uniteye/Scripts/Runtime/GazeGameAPI.cs) does the same through the [UnitEyeAPI](#uniteyeapi).

> Note: the standalone `GazeGame` demo scene was removed together with the other HolisticBarracuda-based scenes during the Barracuda→Inference Engine migration. `GazeGame.cs` now references the `HomulerGaze` component; to try it, add it to a copy of `HomulerGazeScene` with a tagged, collidered GameObject.

## License
* Unity Inference Engine (`com.unity.ai.inference`) is licensed under the [Unity Companion License](https://unity.com/legal/licenses/unity-companion-license)
* MediaPipe Unity Plugin (homuler) is licensed under the [MIT](https://github.com/homuler/MediaPipeUnityPlugin/blob/master/LICENSE) license; the bundled MediaPipe models are Apache 2.0
* EyeMU is licensed under the [GPL 2.0](https://github.com/FIGLAB/EyeMU/blob/master/LICENSE) license
* HomulerMediaPipe is licensed under the [MIT](https://github.com/homuler/MediaPipeUnityPlugin/blob/master/LICENSE)

UnitEye itself therefore also needs the GPL License, however, we use version [3.0](/LICENSE).
