# Performance tuning

The gaze pipeline runs **synchronously** — the MediaPipe FaceMesh graph and the EyeMU inference block
the frame, so your project's frame rate is capped by how fast that pipeline runs (README covers this).
After the July 2026 audit (see [CODE-AUDIT.md](CODE-AUDIT.md)) the per-frame GC and CPU/GPU-readback
costs are largely gone; the remaining levers below are mostly about how hard MediaPipe works per frame.
Roughly in value order:

## 1. Lower the webcam capture resolution (biggest cheap win)

FaceMesh cost scales with input resolution, and gaze only needs the eye regions — 1080p is wasted. On
the **`Mediapipe`** GameObject, the **`WebCamSource`** component has **Preferable Default Width**
(`_preferableDefaultWidth`, default 1280). Set it to **960** or **640** and it picks the nearest
available capture resolution (options run down to 640×480 / 424×240). Big reduction in capture +
FaceMesh cost for little accuracy loss.

## 2. Turn off the face-mesh overlay

The 468-point overlay is a per-frame draw. Toggle it from the **Gaze UI → Show/Hide FaceMesh** button,
or the `showFaceMesh` field on `HomulerGaze`, or `FaceMeshSolution.Annotate`. (Added in code; the toggle
is honoured across calibration.)

## 3. Drop the iris/attention model — **only if you don't need camera-distance or drowsiness**

On the **`FaceMeshGraph`** component, uncheck **Refine Landmarks** (`refineLandmarks`). That loads the
lighter `face_landmark.bytes` instead of `face_landmark_with_attention.bytes`. Gaze and blink are
unaffected (the eye corners `263/362/33/133` and the eye-aspect-ratio landmarks are all in the base
468-point mesh). **Caveat:** it removes the iris landmarks (468–477), which `HomulerEyeHelper` uses for
the camera-distance estimate and iris size — those paths would need a guard (or will read empty iris
lists) with attention off, so test it if you rely on distance/drowsiness. For pure gaze it's a real
speedup.

## 4. Async model readback (advanced — not applied by default)

`HomulerEyeMURunner.PerformInference` reads the model outputs with `DownloadToArray()`, which forces a
GPU→CPU sync. You can defer this a frame with the Inference Engine's async readback to remove the stall,
at the cost of one frame of gaze latency. Sketch:

```csharp
_worker.Schedule();
var gaze = _worker.PeekOutput(OUTPUT_GAZE) as Tensor<float>;
gaze.ReadbackRequest();              // non-blocking
// ...next frame, before the next Schedule():
if (gaze.IsReadbackRequestDone())    // now DownloadToArray() is cheap
    var data = gaze.DownloadToArray();
```

Left out of the shipped code because a mistake here silently corrupts inference and it needs a live
webcam to verify — do it deliberately with a camera in front of you.

## 5. Decouple gaze rate from render rate (biggest structural change)

`FaceMeshSolution` (and its base `ImageSourceSolution`) has a **Running Mode** field. It defaults to a
**synchronous** mode, which is why the pipeline caps your frame rate. Switching it to **NonBlockingSync**
or **Async** runs the graph off the render loop and delivers landmarks via callback, so rendering can run
faster than inference. This is the biggest potential win but also the largest behavioural change (adds
latency; the eye crop reads whatever landmarks arrived last), so validate gaze accuracy after switching.

## 6. Ship with the debug UI off

`showEyes`, `showGazeUI`, `visualizeAOI` and the Gaze UI window all cost `OnGUI` time; leave them off in
a shipping build. If a `LandmarkVisualizer` is in your scene, disable it (it spawns a primitive every
frame until one exists).
