/*
 * uniteye-webgl-boot.js — glue between a Unity WebGL build and the browser gaze pipeline.
 *
 * Include this in your Unity WebGL template (after uniteye-core.js), alongside uniteye-cv.js, and serve
 * the EyeMU model at models/eyemu.onnx (adjust modelUrl below). It defines window.UnitEyeStartPipeline,
 * which UnitEyeWebGL.jslib calls once Unity starts. It streams RAW gaze + the 12-feature vector into
 * Unity, so Unity's own C# calibration + One-Euro filter + AOI logging run unchanged (the shared layer).
 */
window.UnitEyeStartPipeline = async function (send) {
  // Idempotent: a second receiver (e.g. after a scene reload) reuses the running pipeline instead of
  // opening a second webcam capture + inference loop.
  if (window.__unitEyeEngine) {
    console.log('UNITEYE_PIPELINE_REUSED (already running)');
    window.__unitEyeEngine.onGaze = function (finalGaze, raw, blink, facePresent, features) {
      send(raw.x, raw.y, blink, facePresent, features || []);
    };
    return;
  }
  const { UnitEyeWeb } = await import('./uniteye-cv.js');
  // ?mock=1 runs the mouse-driven mock pipeline (no webcam) — used for end-to-end testing of the
  // JS -> jslib -> C# bridge without camera hardware.
  const mock = new URLSearchParams(window.location.search).get('mock') === '1';
  const engine = new UnitEyeWeb({
    mock: mock,
    modelUrl: 'models/eyemu.onnx',
    // Feed RAW gaze + features to Unity; Unity applies calibration/filtering. (Do not calibrate here.)
    onGaze: function (finalGaze, raw, blink, facePresent, features) {
      send(raw.x, raw.y, blink, facePresent, features || []);
    },
  });
  await engine.init();
  engine.start();
  console.log('UNITEYE_PIPELINE_STARTED mock=' + mock);
  window.__unitEyeEngine = engine; // handy for debugging in the browser console
};
