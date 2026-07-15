# UnitEye code audit — performance & maintainability

Record of the July 2026 full-codebase audit (multi-agent review, adversarially verified: 118 raw
findings → 115 verified + 32 cross-cutting). Everything below the "Fixed" line landed as commits and
was verified against the 85-check editor smoke suite on Unity 6000.5.3f1 after each wave. The
"Deferred" section lists what was intentionally not applied, with the reason and file:line, so nothing
is lost.

## Fixed (verified, smoke 85/85)

**Wave 1 — correctness bugs + resource leaks**
- `HomulerFunctions.FlipTexture` allocated + leaked a `Texture2D` every frame → reuse a static buffer.
- `HomulerEyeMURunner.Dispose` leaked 4 `RenderTexture`s → release them.
- `WebCamInput.LoadCamera` leaked the previous `WebCamTexture`+`RenderTexture` on each switch → release first.
- `AOIManager`/`AOICombined.RemoveAOI(string)` + `AOIPolygon.RemoveAllPoints` mutated a list inside a
  `foreach` (crash on match, reachable from `UnitEyeAPI.RemoveAOI`) → `RemoveAll`.
- `WebCamInput.Start` NRE on the `staticInput` path → guard; size `inputRT` from the requested resolution.
- `Gazeable.HasGazeFocus` operator precedence (`&&` vs `?:`) → parenthesized.
- `HomulerGaze` Gaze UI Previous/Next webcam buttons were swapped.
- `HomulerGazeEvaluation.CalculateRMSE` used the field instead of its params → use params, document metric.
- `FaceMeshGraph` wrong stream name to `AddPacketPresenceCalculator`.
- `OneEuroFilter` froze on `dt<=0` → guard.
- Inspector editors mirrored properties through cached bools (broke Undo/multi-edit) → bind directly.
- `MainMenu` loaded deleted scenes → repoint survivors, warn on the rest.

**Wave 2 — per-frame heap allocations (exact math preserved; `Predict` is smoke-validated)**
- `RidgeRegression.Predict` (List+ToArray+DenseVector ×2/frame) → direct dot product, zero alloc.
- `SimpleMLP.Predict` (6 `float[]`/call) → reused per-instance scratch.
- `HomulerEyeMURunner.Features` (List+growth/frame) → reused `float[12]`; pose `float[4]` reused;
  `IGazeProvider.GetFeatures()` returns the buffer (calibration capture clones it).
- `OneEuroFilter` boxed `Vector2` 3-4×/frame via `Convert.ChangeType` → allocation-free `FilterVector2`.
- `HomulerGaze` debug UI mutated the **shared** `GUI.skin` styles (leaked into all other IMGUI) and
  allocated a `GUIStyle` + two `Enum.GetValues` arrays per OnGUI pass → cached per resolution.
- `CSVLogger.WriteQueue` O(n²) drain → iterate once + `Clear()`.
- Tensor RenderTextures dropped their unused 24-bit depth buffer.

**Wave 3 — smoothing, dedup, logging, API hardening**
- `EaseSmoothing`/`KalmanFilter` were frame-rate dependent → time-aware, normalized to 30 fps (unchanged there).
- Removed dead duplicated helpers (`Functions.FlipTexture`/`PreprocessImage`, `HomulerFunctions.PixelsToMm`/`Quit`).
- Added `UnitEyeLog` façade; replaced the 6 silent empty `catch{}` calibration loads with a single
  `LoadCalibrationModels()` that surfaces corrupt/incompatible files.
- `UnitEyeAPI`: `s_gazeScript` made private; `FindObjectOfType`→`FindFirstObjectByType`; cache
  `Camera.main`; **`GetHeadPose` fed radians into `Quaternion.Euler` (wants degrees)** → `Rad2Deg`.
- `NativeGazeProvider` computes `EyeFeature()` once/frame (was 3×).

**Wave 4 — misc perf/cleanup**
- `AOITagList` xray raycast reused `RaycastHit[]`+`List` buffers; dead `hit.Equals(null)` branch removed;
  `CheckValidWebCam` returns on first match.

**Wave 5 — GPU eye crop** (needs a webcam to fully validate the sampling orientation)
- Replaced the per-frame CPU eye-crop path (`GetPixels` readback + `Texture2D` churn + a second
  `GetPixels32` round-trip to flip the left eye) with a direct `Graphics.Blit` crop from the webcam
  texture into the eye RenderTextures. Removed `GetEyeTexture`/`FlipTexture` (now dead). `GetEyeCropRect`
  (the smoke-pinned geometry) is unchanged. Blocking `DownloadToArray` on the outputs left as-is.

**Wave 6 — namespace unification**
- Wrapped all ~33 global-namespace first-party runtime types in `namespace UnitEye`. Editor tooling
  invoked by `-executeMethod` kept its global names; `EyeMUAssetRebinder` gained `using UnitEye;`.

**Wave 7 — god-class split (part 1)**
- Extracted `CalibrationModelStore` (model fields + `Load` + `Refine`) out of `HomulerGaze`.

Also (earlier this session): removed the orphaned `HolisticBarracuda/` package + dead Visualizer
shaders/README; added [HOMULER-UPGRADE.md](HOMULER-UPGRADE.md).

## Deferred (with rationale)

### Needs a webcam to confirm (change applied; verify behaviour)
- **Wave 5's GPU eye crop** — verify the `Show Eyecrops` thumbnails look identical to before and that
  gaze accuracy is unchanged. If the crop is mirrored/upside-down, the fix is the `flipX`/y-scale in
  `HomulerEyeMURunner.BlitEyeCrop`. Optionally follow up with `AsyncGPUReadback` for the two
  `DownloadToArray` output readbacks (adds one frame of latency; removes two GPU→CPU stalls).

### Deferred with rationale
- **Extract the ~450-line debug IMGUI out of `HomulerGaze`** (the remaining god-class item; the model
  store landed in Wave 7). Evaluated and left in place: `GazeUI`/`OnGUI` bind sliders directly to
  `HomulerGaze`'s private filter/provider state (`kalmanFilter.Q = Q = GUI.HorizontalSlider(...)`), so a
  separate overlay component would have to expose those internals — little real decoupling for
  non-trivial regression risk in a GUI that has a documented history of subtle high-DPI / hit-testing /
  every-frame-revert bugs. Best done alongside a broader UI rework, with the editor open.
- **Route the remaining ~56 `Debug.*` calls through `UnitEyeLog`** so a host game can fully silence the
  package. The façade exists and the high-value sites (calibration load failures) are converted.

### Lower-value micro-optimizations / cleanups (safe, not yet applied)
- `AOICircle.CheckCircle` uses `Mathf.Sqrt` where a squared-distance compare suffices.
- `AOICombined.CheckAOI` doesn't early-out (careful: sub-AOI `CheckAOI` has focus/hit side effects).
- `FaceMeshSolution.EyeCorners` allocates 3 arrays/frame (LINQ `Concat`/`ToArray`).
- Duplicated Fisher-Yates (`ListExtension.Shuffle` vs `RidgeCalibrationTrainer.RandomPermutation`) and
  holdout/standardization/RMSE logic (`RidgeCalibrationTrainer` vs `SimpleMLP`).
- `AOICapsule`/`AOICapsuleBox` duplicate the corner-offset geometry.
- `CSVData.SerializeCSVData` wraps every value in an interpolated string, defeating its StringBuilder.
- Unused `using`s (`Unity.Mathematics` across AOI shapes, `System.Collections` in preset files);
  `CirclePreset` appears never instantiated (left in place — deleting a type risks scene GUID refs).
- WebGL JS per-frame allocations (`uniteye-cv.js`: a fresh 192 KB `Float32Array` per eye crop, two
  `ImageData` readbacks/frame, per-frame gaze string across the bridge) — the browser hot path;
  not applied here because it can't be validated by the Unity smoke suite.
- **Cross-platform drift guard**: add a golden-vector fixture (feature→predict, landmarks→crop,
  raw→filtered) asserted by *both* `UnitEyeSmokeTests` and `webgl/test.html`, since the native C# and
  browser JS math are hand-ported.
- Smoke suite covers only the OneEuro filter — add Kalman/Easing/CSV-formatting checks.
