# Upgrading the homuler MediaPipe Unity Plugin

Scoping document — how to move UnitEye off the vendored **0.12.0** copy of the
[MediaPipe Unity Plugin](https://github.com/homuler/MediaPipeUnityPlugin), what it costs, and
which target to aim for. Written July 2026.

## TL;DR

| | Current | Quick bump | Latest |
|---|---|---|---|
| **Version** | 0.12.0 | 0.14.x | 0.16.3 (Nov 2024) |
| **API** | legacy Solution | legacy Solution | Task (`FaceLandmarker`) |
| **Effort** | — | low–medium | medium–large |
| **Gets you to latest?** | — | ❌ | ✅ |
| **Rewrite of the CV seam?** | — | no (re-vendor only) | yes |

**Recommendation:** don't take the 0.14.x half-step unless you need a *specific* upstream fix
right now. It re-vendors code onto an API that is already deleted upstream, so you'd pay the
Task-API rewrite later anyway. If an upgrade is warranted, go **straight to 0.16.3 on the Task
API** (Option B). If nothing concrete is broken on 0.12.0, **staying put is defensible** — the
native FaceMesh output we consume has not changed.

## Why you can't just bump the version

This integration is built entirely on the plugin's **legacy Solution API**. The vendored subset
in [`uniteye/Scripts/Runtime/Mediapipe/`](../uniteye/Scripts/Runtime/Mediapipe) is homuler's
`ImageSourceSolution` sample code, lightly customized:

- `FaceMeshSolution : ImageSourceSolution<FaceMeshGraph>` — runs a MediaPipe **CalculatorGraph**,
  feeds `TextureFrame`s into an `input_video` stream, and reads back the `multi_face_landmarks`,
  `face_rects_from_landmarks`, and `face_detections` output streams.
- `FaceMeshGraph : GraphRunner` — configures the graph, loads `.bytes` models from
  `StreamingAssets` (`face_detection_short_range.bytes` + `face_landmark_with_attention.bytes`),
  patches detection/tracking-confidence calculator options via protobuf reflection.
- Supporting plumbing: `ImageSource`/`WebCamSource`/`TextureFrame`/`TextureFramePool`, `Screen`,
  `Bootstrap`, `GlobalConfigManager`, `AssetLoader`, `GraphRunner`, `WaitForResult`, etc.

The plugin's release history:

- **0.13.1** — new **Task API** (`FaceLandmarker`, `FaceDetector`, …) introduced alongside the
  Solution API.
- **0.15.0** (Aug 2023) — *"remove legacy solutions (#1233)"* — `ImageSourceSolution`,
  `FaceMeshSolution`, the sample `GraphRunner` base, etc. are **deleted**.
- **0.16.3** (Nov 2024) — latest.

So **0.14.x is the last release that still contains the API this code depends on.** Anything ≥
0.15.0 removes `FaceMeshSolution`/`ImageSourceSolution`/`GraphRunner` and forces the Task API.

## What consumes the FaceMesh output (the blast radius)

Only a handful of first-party files read the solution, all through a narrow surface. Any upgrade
has to preserve these exact values (they feed verified eye-crop geometry and calibration):

| Consumer | What it reads |
|---|---|
| `NativeGazeProvider` | constructs `FaceMeshSolution` + `WebCamSource` off the `Mediapipe` GameObject; drives `Tick()` |
| `HomulerEyeMURunner` | `FaceLandmarks`, `EyeCorners` (263/362/33/133), `HeadGeom` → EyeMU model inputs |
| `HomulerEyeHelper` | `FaceLandmarks[i]` for EAR (blink/drowsy), `Left/RightIrisLandmarks` (5 pts) for iris-size/distance, `HeadYaw`/`HeadPitch` |
| `HomulerFunctions` | eye-crop rect from `FaceLandmarks` corners |
| `FaceMeshSolution` (ours) | derives `HeadYaw/Pitch/Roll` (landmarks 50/280/10/168/6/151), `HeadArea` (`FaceRects[0]`), `EyeCorners`, iris partition |
| `LandmarkVisualizer` | single `FaceLandmarks[i]` for a debug sphere |

Everything downstream of `IGazeProvider` (calibration, filtering, AOI, CSV) is **platform-agnostic
and untouched** by a homuler upgrade — that's the whole point of the provider seam. The WebGL
browser path (`webgl/`) is also unaffected: it never used homuler native at all.

## Option A — Quick bump to 0.14.x (stay on the Solution API)

Keep the architecture; re-vendor the newer sample code.

**Steps**
1. Pull `com.github.homuler.mediapipe` 0.14.x (native binaries + managed scripts + `.bytes` models
   + graph `.txt` config).
2. Re-apply our local customizations on top of the newer `FaceMeshSolution`/`FaceMeshGraph`/
   `GraphRunner`: the head-pose/`EyeCorners`/`HeadGeom` exposers, the `HeadRoll` [151] fix, and the
   `GraphRunner.GetInstanceID` → local-counter change (6.5 CS0619 fix — see
   [BARRACUDA-MIGRATION.md](BARRACUDA-MIGRATION.md) / memory).
3. Refresh `StreamingAssets` `.bytes` models to the 0.14.x versions
   (`MediaPipeAssetInstaller`).
4. Re-run the 85-check smoke suite on 6.3 **and** 6.5; rebuild + re-verify both WebGL players
   (the `#if !UNITY_WEBGL` guards must still exclude the native path cleanly).

**Buys:** newer native MediaPipe binaries + upstream bug fixes; same code shape.
**Does not buy:** the latest version, blendshapes, transformation-matrix head pose.
**Risk:** low–medium — Solution-API drift between 0.12→0.14, native-binary/graph-asset changes,
re-merging our patches. **You are still on a deleted API afterwards.**

## Option B — Rewrite to the Task API for 0.16.3 (recommended if upgrading)

Replace the CalculatorGraph plumbing with `FaceLandmarker`.

**What gets deleted / shrunk:** most of `uniteye/Scripts/Runtime/Mediapipe/` — `FaceMeshGraph`,
the sample `GraphRunner`, `ImageSourceSolution`, the `OutputStream`/`SidePacket`/packet plumbing.
`WebCamSource`/`TextureFrame` may partly survive (still need a webcam texture) or be replaced by
the Task API's image helpers.

**What gets written:** a thin `FaceLandmarkerProvider` (or a rewritten `FaceMeshSolution` shim
that keeps the same public surface so `NativeGazeProvider` barely changes):
- `FaceLandmarker.CreateFromOptions(...)` with `outputFaceBlendshapes` +
  `outputFacialTransformationMatrixes` as desired; one **`.task`** bundle in `StreamingAssets`
  instead of the separate `.bytes` models + graph `.txt`.
- Per frame: build an `Image` from the webcam texture → `DetectForVideo(image, timestampMs)` →
  `FaceLandmarkerResult`.
- **Landmark remap:** result `faceLandmarks[0]` is the 478-point list *with iris already appended*
  (no `PartitionLandmarkList`): mesh 0–467, **left iris 468–472, right iris 473–477**. Re-point
  `Left/RightIrisLandmarks` and re-verify every hardcoded index (263/362/33/133/50/280/10/168/6/151,
  and the EAR indices in `HomulerEyeHelper`).
- **`HeadArea`:** the Task API returns no `face_rects_from_landmarks`; derive it from the landmark
  bounding box (this is what the WebGL path already does — note the documented native-vs-browser
  distribution difference, so calibration stays per-platform).
- **Optional accuracy wins:** use the `eyeBlink` **blendshapes** for blink/drowsy instead of the
  hand-rolled EAR, and the **facial transformation matrix** for head pose instead of the
  landmark-difference `HeadYaw/Pitch/Roll` math. Both are cleaner and better-conditioned — but they
  change the 12-feature vector, so they'd require **recalibration** and re-verifying the shipped
  contract. Treat as a follow-up, not part of the port.

**Buys:** latest + supported API, far less vendored code, optional blendshape/matrix upgrades.
**Risk:** medium–large — re-verify the entire numeric chain (eye crops, EAR, iris size, distance,
calibration) against the new landmark source; the 85-check smoke suite pins much of this geometry,
so lean on it. Native-only as before (Task API still ships native binaries; **no WebGL**).

## Decision guide

- **Nothing broken on 0.12.0, no need for new features** → stay. The consumed output is stable.
- **Need a specific 0.13/0.14 upstream fix now** → Option A as a stopgap, but log that the
  Task-API rewrite is still owed.
- **Want to be current / want blendshapes / matrix head pose / less vendored code** → Option B,
  straight to 0.16.3. Skip A.

Whichever path: the provider seam means only `NativeGazeProvider` + the vendored `Mediapipe/`
folder move. Downstream calibration/filter/AOI/CSV and the WebGL path do not.
