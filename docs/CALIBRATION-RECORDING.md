# Recording calibration sessions as a dataset

Lets a consenting participant have their calibration session written to disk, so gaze accuracy work can be
measured against recorded sessions instead of re-run by hand each time. It closes the loop with the existing
evaluation: record → retrain or re-tune offline → re-benchmark.

**Nothing here uploads anything.** The feature records consent to publish; it never publishes. Moving a folder
into a repository is a separate, deliberate, human act. See [Publishing](#publishing) for why that separation
is load-bearing rather than squeamish.

## Setup

Add a `CalibrationRecordingConsent` component to the same GameObject as `HomulerGazeCalibration`, then:

| Field | Meaning |
|---|---|
| `askBeforeCalibration` | Off by default. On = every calibration starts with the consent screens. |
| `maxTierOffered` | Highest tier a participant may pick. Lower it for studies that must not collect imagery. |
| `withdrawalContact` | Shown verbatim to the participant and stored in `consent.json`. Set this. |
| `studyLabel` | Label for the session set. Never a participant name. |

With `askBeforeCalibration` off, nothing changes and nothing is recorded.

## Tiers

Each tier includes the ones above it. The ladder exists because "record the calibration" spans a huge privacy
range, and consent is only meaningful against a specific point on it.

| Tier | Adds | Good for |
|---|---|---|
| `Features` | Feature vector + on-screen label + quality flags | Retraining/re-benchmarking the **calibration head** (ridge, MLP) — what the in-app evaluation measures |
| `Landmarks` | 478 face landmarks, eye blendshapes, head pose | Head-pose and geometry features. ~40× the size of `Features` |
| `EyeCrops` | The two 128×128 eye crops the model consumes | **Retraining the gaze backbone.** Best value per byte |
| `FaceVideo` | Frames cropped to the face box | Detection/landmark work, without recording the room |
| `FullFrames` | The entire camera frame | Only when the background genuinely matters |

`FullFrames` requires a second confirmation against a live camera preview, because people reliably forget
what is behind them.

## On-disk layout

One folder per session, under `Application.persistentDataPath/UnitEyeRecordings/`, split by consent:

```
UnitEyeRecordings/
  publication-consented/     # participant agreed it may be published
  local-only/                # participant did not
    K7QM-3XPV-8ZR2/          # folder name IS the withdrawal code
      consent.json           # written first; if it is missing, treat the folder as unconsented and delete it
      session.json           # backbone, geometry, flips, format version
      samples.jsonl          # one JSON object per training sample
      features.f32           # raw float32 feature vectors
      landmarks.f32          # raw float32 landmark triplets (tier >= Landmarks)
      eyes/000123_L.png      # lossless (tier >= EyeCrops)
      frames/000123.jpg      # JPEG q90 (tier >= FaceVideo)
      summary.json           # outcome, counts, dropped images, session RMSE
```

Publication-consented and local-only are **separate roots** so excluding a set is a directory operation
rather than a flag a script can forget to check.

### `samples.jsonl`

One object per accepted training sample. `i` is the sample's index in the calibration's own training arrays,
so rows join to trained models directly — and images are 1:1 with rows *by construction*, because the
recorder is hooked at the tail of the capture path, past every rejection.

```json
{"i":0,"t":3.5171,"labelX":1712.5,"labelY":221,"targetX":1712.5,"targetY":221,
 "dwell":true,"headRotation":false,"preset":0,"round":0,"blinking":false,
 "irisDisagreement":0.0121,"featureOffset":0,"featureCount":36,
 "distanceMm":512.5,"headPitch":0.031,"headYaw":-0.08,"headRoll":0.004,
 "landmarkOffset":0,"landmarkCount":478,"eyeBlendshapes":[...]}
```

Fields are **omitted, never zero-filled**, when a platform cannot supply them. A sentinel written as a
measurement is indistinguishable from a real one once it is in a dataset.

Read a feature vector as `featureCount` little-endian float32s at byte `featureOffset` in `features.f32`;
landmarks likewise as `landmarkCount × 3` at `landmarkOffset`.

`session.json` deliberately carries **no** feature-vector length: the direction backbones append a
runtime-sized embedding tail only after their first inference, so a session-level field would be a confident
lie. Each row's `featureCount` is authoritative — and rows within one session can legitimately differ.

All numbers are written with `InvariantCulture`. This matters more than it sounds: under a comma-decimal
locale the ambient formatter renders `0.4193` as `0,4193`, which silently turns one JSON number into
something no other machine can parse. There is a smoke test for it.

### Physical scale: check `physicalScaleTrustworthy` before using centimetres

`session.json` records `physicalScaleTrustworthy`, and it is the field to gate on before trusting any
centimetre or degree figure from a session.

The pipeline works in `Screen.width`/`Screen.height` — the **render surface**, which in the Editor is the
Game view, not the monitor. Everything stays internally consistent there, so calibration, gaze and the
normalized labels are fine. But the conversion to physical units is `pixels × 25.4 / Screen.dpi`, which
assumes one render pixel covers one physical pixel. Run a 1920×1080 Game view inside a 900px-wide panel and
the reported RMSE, the accuracy verdict, `screenWidthCm`/`screenHeightCm` and any degrees-of-visual-angle
derived from them are all wrong by that ratio.

So the recorder writes the provenance instead of pretending: `displayWidthPx`, `displayHeightPx`,
`screenDpi`, `renderMatchesDisplay`, `recordedInEditor`, and the single `physicalScaleTrustworthy` verdict.
The calibration also logs a warning the moment a run starts under those conditions — the last point at which
maximising the Game view is free.

Consequences:

* **% of screen diagonal is always valid**, trustworthy scale or not: it is a ratio of the same units.
* **Degrees of visual angle are withheld** by the benchmark when the flag is false. A confidently wrong
  degree figure pooled into a median is worse than a missing one.
* Sessions recorded before this field existed read as **not** trustworthy — unknown scale is treated as
  untrusted, which is the safe direction.

For data you intend to donate or compare, **run a fullscreen build**, or a Game view at scale 1× with
Maximize On Play. Note `Screen.dpi` is independently unreliable (many Windows setups report a flat 96),
which is why the calibration screen has always carried its own DPI caveat.

### Caveats that belong with the data, not in a footnote

- **Landmark coordinates** are MediaPipe-normalized (0–1, y-down, top-left) against the **camera frame**, not
  the screen. Denormalize with `cameraFrameWidth`/`cameraFrameHeight` from `session.json`.
- **Frame flips** are recorded in `session.json`, not corrected. Landmark-to-pixel registration depends on
  them, and silently "fixing" orientation is how a dataset ends up with points that miss the face.
- **Six landmarks are smoothed** (4 eye corners + 2 iris centres) while the other 472 are raw, when
  `gazeLandmarksSmoothed` is true.
- **Eye crops are not centred on the eye.** The crop expands the corner span by 1.4× and places the box with a
  width-normalized offset applied to a height-normalized coordinate — a deliberate quirk retained for
  model compatibility. At 16:9 the eye sits roughly a quarter down from the crop top, so the crop reaches
  onto the cheek. The left crop is additionally mirrored, and images use a bottom-left (GPU) origin.
- **Images are dropped, not blocked, under load.** `summary.json` reports `imagesDropped`. Sample rows are
  never affected. A dataset that quietly lost frames looks complete and is not, so check this field.
- **Imagery is refused under async GPU readback.** In that mode the backbone publishes frame N−1's gaze while
  the crops already hold frame N, so pixels and labels would disagree with nothing downstream able to notice.
  The tier is clamped to `Landmarks` and a warning is logged.

## Turning a session into video

There is no runtime video encoder in Unity (`MediaEncoder` is Editor-only), and adding a third-party one for
this would be the wrong trade. Frame sequences are also the better artifact for training — no inter-frame
compression smearing the eye region, and exact frame-to-label pairing. To eyeball a session:

```bash
ffmpeg -framerate 30 -pattern_type glob -i 'frames/*.jpg' -c:v libx264 -pix_fmt yuv420p session.mp4
```

## Packaging and sharing

**`UnitEye ▸ Recorded Sessions`** ([`RecordedSessionBrowser`](../uniteye/Scripts/Editor/RecordedSessionBrowser.cs))
lists every recording with its tier, sample count, size, measured accuracy and dropped-image count, and does
the packaging:

* **Package … into a zip** — one archive, one folder per session named by its withdrawal code. Entry paths are
  built explicitly rather than from the absolute source path, so the archive carries no account name.
* **Package + open upload page…** — the same zip, then reveals it and opens the upload page in a browser. No
  credential needed.
* **Post to GitHub…** — uploads directly, as a release asset. See below.
* **Delete** — how a withdrawal request is honoured.

Every path shows a summary before anything leaves the machine: session count, size, tiers, and a distinct
warning when the selection contains imagery of people.

### Posting directly to GitHub

Set a token in the environment and restart Unity (it reads the environment at launch):

```bash
setx UNITEYE_GITHUB_TOKEN "github_pat_..."
```

Use a **fine-grained** token scoped to the one target repository with **Contents: Read and write**, and set an
expiry. A classic `repo` token grants read/write to every private repository the account can reach — a poor
blast radius for a credential sitting in an environment variable on a lab machine. Until the variable is set,
the button stays disabled with an explanatory line, so it never looks available when it is not.

Set owner / repo / release tag in the window. Then **Post to GitHub…**
([`GitHubReleaseUploader`](../uniteye/Scripts/Editor/GitHubReleaseUploader.cs)):

1. Confirms which account the token belongs to and shows it. Publishing biometric data from the wrong account
   is unrecoverable and easy to do.
2. Uploads **one asset per session**, not one combined zip — so honouring a withdrawal costs a single
   `DELETE`, instead of repackaging and re-uploading everyone else's data to remove one person.
3. Lands on a **draft** release. Draft assets are visible only to accounts with push access, so the gap
   between "uploaded" and "world-readable" stays under your control. Draft is *not* private and *not*
   encrypted — it is on GitHub's servers and visible to every collaborator.
4. Writes **`publication-receipt.json`** into each session folder (owner, repo, release id, asset id,
   download URL). This is the map from withdrawal code to published asset; without it you would know someone
   wants out but not which asset is theirs. It contains no credential.
5. Publishing the draft is a **separate, second confirmation**. That is the irreversible step.

The token is read from the environment at point of use, never cached, never serialized, and scrubbed from
every error string. It is Editor-only, so the smoke test asserting the runtime contains no network code still
passes and no credential can reach a build. The server-supplied `upload_url` is host-validated before use —
it decides where a token plus facial imagery gets sent.

Sessions that cannot be shared are listed but not selectable, with the reason: *participant said local-only*,
*hold until `<date>`*, *no consent.json*, or *no samples captured*. Those gates are the enforcement point for
the promise the consent screen made — `GazeConsentRecord.PublishableOn` encodes the consent and hold checks
together, so a folder cannot be packaged past them.

All of this is **Editor-only**, which is the design rather than a limitation: the runtime a participant runs
has no network code at all and a smoke test enforces that, so the consent screen's "we never send anything
over the internet" stays literally true, and no credential can reach a shipped build.

Publication is irreversible in a way that deserves the friction. Once a folder is public, copies exist beyond
anyone's control and git history keeps it recoverable even after deletion — the consent text says exactly
that, in those words.

Change the browser-upload target by editing `RecordedSessionBrowser.UploadPageUrl`.

Consider `.gitattributes` marking `*.f32`, `*.png` and `*.jpg` as binary, and Git LFS for the imagery tiers —
a few minutes of `EyeCrops` is hundreds of megabytes.

## Benchmarking the donated data

**`UnitEye ▸ Run Gaze Benchmark`** ([`GazeBenchmark`](../uniteye/Scripts/Editor/Benchmark/GazeBenchmark.cs)),
or headless:

```bash
Unity.exe -batchmode -projectPath <host project> -executeMethod UnitEye.Benchmark.GazeBenchmark.Run -logFile bench.log
```

This is what closes the loop: change something, re-run, and see whether accuracy moved **across every donated
session** rather than across one calibration you happened to run by hand. Results go to
`UnitEyeRecordings/benchmark.tsv`, one row per session per config, plus per-group summaries.

Out of the box it compares the four shipped combinations: `ridge-aug`, `ridge-noaug`, `mlp-aug`, `mlp-noaug`.
MLP folds are ~200 epochs each and dominate runtime — trim `DefaultConfigs` for a quick ridge-only A/B.

Three things make the number trustworthy, and all three are easy to get wrong:

* **The split is by screen location, not by sample.** Consecutive dwell rows are the same target, same head
  pose, milliseconds apart — near-duplicates. A per-sample split puts near-copies on both sides and reports a
  flatteringly low error. One whole dwell location is held out per fold, and a spatial **buffer**
  (6% of the diagonal) additionally removes *sweep* rows passing through it. Without the buffer the held-out
  location is not actually held out, because sweeps carry the correct label straight through it.
* **It trains through the shipped code.** `CalibrationSampleBalancer` and `RidgeCalibrationTrainer` /
  `SimpleMLP` are the same types the calibration uses, with the same augmentation and head-pose feature
  indices. A reimplementation would benchmark a pipeline nobody runs.
* **Both heads are scored in pixels.** The ridge pair predicts normalized units and `SimpleMLP` predicts
  pixels; each is converted at the point of fitting so the two are comparable.

**Expect worse numbers than the calibration reports.** `summary.json`'s `holdoutRmseCm` comes from the
shipped trainer's per-sample split, which has exactly the leak described above. Both are printed side by
side, and if a majority of sessions score *better* than the app's self-report the report emits a `# WARNING`:
that means the split has broken, not that the config improved.

Aggregation is **median and IQR per feature-layout group** (`backbone/featureCount`), never pooled across
groups — a 36-value EyeMU vector and a 32-value direction vector describe different systems. Accuracy is
reported as % of screen diagonal always, and in **degrees of visual angle** where rows carried `distanceMm`
(converted per row, then RMS-ed — viewing distance varies within a session).

Sessions that cannot be benchmarked are reported with a reason (`excluded:too-few-locations`,
`excluded:jagged-features`, `excluded:no-consent`, …) rather than skipped silently.

Not yet covered: the **thin-plate-spline warp** as a config. Its anchors are built from the raw captured
arrays, so benchmarking it needs the same leave-one-anchor-out treatment the shipped gate now uses — wired
naively it would train on the labels it is scored against and always appear to win.

## Withdrawal

A participant's only identifier is the random token that names their folder. There is no lookup table and
nothing linking it to a person, which is the point: it supports deletion without ever having collected a
name. Withdrawal is therefore `rm -rf` on the folder with that name, plus removal from anywhere it was
published. Participants who change their mind immediately can use **Delete my recording now** on the closing
screen without contacting anyone.

## Before you record anyone but yourself

Facial imagery and face geometry are biometric data, and in the EU that is special-category data under GDPR.
This is a flag, not legal advice: talk to your institution's ethics board about approval, retention and
lawful basis before collecting from participants. The consent screens here are designed to support that
conversation — the exact wording shown is SHA-256-pinned in `consent.json` and in a smoke test, so you can
demonstrate afterwards precisely what each participant agreed to.
