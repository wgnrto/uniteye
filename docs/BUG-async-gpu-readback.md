# Bug: `_asyncGpuReadback` throws every frame and stops gaze updating

**Affects:** UnitEye 1.1.0, Unity 6000.5.8f1, `com.unity.ai.inference` 2.6.1, Windows/D3D11
**Severity:** the feature is unusable — enabling it silently disables gaze tracking
**Status:** not fixed. Reported with a repro; the fix needs someone who knows the
Inference Engine tensor-lifetime contract.

## Symptom

Set `HomulerGaze._asyncGpuReadback = true`. The player then logs this **every
frame**, and gaze never updates:

```
InvalidOperationException: Cannot access the data as it is not available
  at UnityEngine.Rendering.AsyncGPUReadbackRequest.GetData[T] (System.Int32 layer)
  at UnitEye.HomulerEyeMURunner.PerformInference (Mediapipe.Unity.WebCamSource webcam)
  at UnitEye.NativeGazeProvider.Tick ()
  at UnitEye.HomulerGaze.LateUpdate ()
```

It compiles and builds cleanly; the failure only appears at runtime. With
`_asyncGpuReadback = false` (the default) everything works, which is why this is
easy to miss.

## Where it goes wrong

`HomulerEyeMURunner.PerformInference`, async branch:

```csharp
_outEmbedding = _worker.PeekOutput(OUTPUT_EMBEDDING) as Tensor<float>;
_outGaze      = _worker.PeekOutput(OUTPUT_GAZE) as Tensor<float>;
_outEmbedding.ReadbackRequest();
_outGaze.ReadbackRequest();
_pendingReadback = true;
```

and the following frame:

```csharp
if (_asyncReadback && _pendingReadback)
{
    if (!_outEmbedding.IsReadbackRequestDone() || !_outGaze.IsReadbackRequestDone())
        return false;
    PublishPendingOutputs();   // -> DownloadToArray() -> throws
}
```

The ordering looks right: publishing happens before the next `Schedule()`, so the
pending tensors are not overwritten by new work. Yet `IsReadbackRequestDone()`
returns `true` while `DownloadToArray()` reports the data as unavailable.

That combination suggests the tensors returned by `PeekOutput` do not remain
valid readback targets across the frame boundary — i.e. the worker may recycle or
invalidate its output allocations between the `ReadbackRequest()` and the
download, so the completion flag refers to a request whose backing storage is
gone.

## Why this is worth fixing rather than removing

The setting exists for a real reason: in sync mode `DownloadToArray()` blocks the
CPU until the GPU finishes, which stalls the frame. That said — on the hardware
this was found on the shader/inference cost was not the bottleneck at all (the
camera was, at ~5fps), so the practical benefit may be smaller than expected.

## Suggested direction

Not verified, listed in order of confidence:

1. Take ownership of the output tensors instead of peeking them, e.g.
   `ReadbackAndClone()` or copying into runner-owned tensors before the request,
   so their lifetime is not tied to the worker's internal allocations.
2. Confirm the Inference Engine contract for `ReadbackRequest()` +
   `IsReadbackRequestDone()` + `DownloadToArray()` across frames in 2.6.1; the
   API changed between Barracuda, Sentis and Inference Engine and the previous
   pattern may no longer hold.
3. Failing both, mark the field `[Obsolete]` or remove it, so it cannot be
   enabled into a broken state.

## Workaround

Leave `_asyncGpuReadback = false`. In VIP-Sim it is pinned off in the scene with
a comment pointing here, because it looks like an obvious performance win and
will otherwise be re-enabled by the next person who goes looking for one.
