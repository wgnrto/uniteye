# Bug: `_asyncGpuReadback` throws every frame and stops gaze updating

**Affects:** UnitEye 1.1.0, Unity 6000.5.8f1, `com.unity.ai.inference` 2.6.1, Windows/D3D11
**Severity:** enabling `_asyncGpuReadback` previously threw every frame and halted gaze updates
**Status:** fixed in `IGazeBackbone`, `HomulerEyeMURunner`, `GazeEstimationRunner`, `CompositeGazeBackbone`, and `NativeGazeProvider`.

## Symptom

Setting `HomulerGaze._asyncGpuReadback = true` previously logged `InvalidOperationException` on every camera tick and halted gaze updates:

```
InvalidOperationException: Cannot access the data as it is not available
  at UnityEngine.Rendering.AsyncGPUReadbackRequest.GetData[T] (System.Int32 layer)
  at UnitEye.HomulerEyeMURunner.PerformInference (Mediapipe.Unity.WebCamSource webcam)
  at UnitEye.NativeGazeProvider.Tick ()
  at UnitEye.HomulerGaze.LateUpdate ()
```

## Root Cause

Four lifecycle and synchronization bugs combined to invalidate readback buffers and desynchronize outputs:

1. **Single-frame readback buffer lifetime vs camera gating**: Unity disposes the native staging buffer of an `AsyncGPUReadbackRequest` at the end of the single engine frame in which `done` turns `true`. Meanwhile, `NativeGazeProvider.Tick()` exited immediately whenever `!_webcam.didUpdateThisFrame`. Whenever display frame rate exceeded camera frame rate, the GPU readback completed during an intermediate render frame and expired before the next camera frame called `DownloadToArray()`.
2. **Staggered multi-tensor completion**: `HomulerEyeMURunner` and `GazeEstimationRunner` waited for all output tensors (`embedding` and `gaze`, or `yaw`, `pitch`, and `embedding`) to complete simultaneously before calling `DownloadToArray()`. Whenever two output tensors finished on consecutive engine frames, the earlier tensor expired before the later tensor completed.
3. **Short-circuiting in `CompositeGazeBackbone`**: `CompositeGazeBackbone.PerformInference` evaluated `_eyeMU.PerformInference(webcam) && _direction.PerformInference(webcam)`. On the first async frame, `_eyeMU` returned `false` while scheduling its readback, which skipped `_direction` and permanently desynchronized the two sub-backbones by one camera frame.
4. **Auxiliary signal skew**: `NativeGazeProvider.Tick()` paired frame `N - 1` async gaze predictions with live frame `N` blink, drowsiness, distance, and binocular iris disagreement values.

## Resolution

1. `IGazeBackbone` now exposes `PollAsyncReadback()` and `HasPendingReadback`. `NativeGazeProvider.Tick()` invokes `PollAsyncReadback()` at the start of every engine frame before checking `_webcam.didUpdateThisFrame`.
2. `HomulerEyeMURunner` and `GazeEstimationRunner` drain each completed tensor into a runner-owned `float[]` buffer on the exact engine frame its `IsReadbackRequestDone()` turns `true`, then publish once all tensors for that inference pass have been drained.
3. `CompositeGazeBackbone` polls and schedules both sub-backbones without short-circuiting so both models stay locked to the same camera frame.
4. `NativeGazeProvider` snapshots blink, drowsiness, and camera distance at schedule time and reads binocular iris disagreement from the published feature vector so all signals describe the same camera frame.
