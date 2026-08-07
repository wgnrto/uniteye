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
* **Package and upload…** — the same zip, then reveals it and opens the upload page with a summary of exactly
  what is about to be shared (session count, size, tiers, and a distinct warning when the selection contains
  imagery of people).
* **Delete** — how a withdrawal request is honoured.

Sessions that cannot be shared are listed but not selectable, with the reason: *participant said local-only*,
*hold until `<date>`*, *no consent.json*, or *no samples captured*. Those gates are the enforcement point for
the promise the consent screen made — `GazeConsentRecord.PublishableOn` encodes the consent and hold checks
together, so a folder cannot be packaged past them.

The window is **Editor-only**, which is the design rather than a limitation:

* The runtime that a participant runs has no network code at all, and a smoke test enforces that, so the
  consent screen's "we never send anything over the internet" stays literally true.
* No GitHub credential can end up in a shipped build, because there is no credential.
* The tool stops at "here is the file, here is the page". Attaching and submitting is a human act, on a file
  that can be inspected first — appropriate for the one step that cannot be undone. Once a folder is pushed
  to a public repository, copies exist beyond anyone's control, and git history keeps it recoverable even
  after deletion. The consent text says exactly that, in those words.

If you automate publication later, keep a map from withdrawal code to published path — otherwise a
withdrawal request cannot actually be honoured.

Change the upload target by editing `RecordedSessionBrowser.UploadPageUrl`.

Consider `.gitattributes` marking `*.f32`, `*.png` and `*.jpg` as binary, and Git LFS for the imagery tiers —
a few minutes of `EyeCrops` is hundreds of megabytes.

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
