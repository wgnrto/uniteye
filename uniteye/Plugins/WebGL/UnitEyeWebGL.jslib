// UnitEye WebGL bridge: starts the browser gaze pipeline and streams gaze into Unity.
// Called from WebGLGazeReceiver.cs via [DllImport("__Internal")] UnitEyeWebGL_Start(gameObjectName).
//
// Requires the WebGL page/template to define window.UnitEyeStartPipeline(send) — see
// webgl/uniteye-webgl-boot.js (include it, uniteye-core.js and uniteye-cv.js in your WebGL template,
// and serve models/eyemu.onnx alongside the build). The pipeline calls send(x, y, blink, face, features)
// each frame; this forwards it to the named GameObject's OnWebGaze method.
mergeInto(LibraryManager.library, {
  UnitEyeWebGL_Start: function (objNamePtr, sharedBufPtr, sharedBufCap) {
    var objName = UTF8ToString(objNamePtr);
    var seq = 0;
    function resolveSendMessage() {
      if (typeof SendMessage === 'function') return SendMessage;
      if (typeof Module !== 'undefined' && typeof Module.SendMessage === 'function') return Module.SendMessage;
      if (typeof window !== 'undefined' && window.unityInstance && typeof window.unityInstance.SendMessage === 'function')
        return function (o, m, v) { window.unityInstance.SendMessage(o, m, v); };
      return null;
    }
    function resolveHeapF32() {
      if (typeof HEAPF32 !== 'undefined' && HEAPF32.buffer && HEAPF32.buffer.byteLength > 0) return HEAPF32;
      if (typeof Module !== 'undefined' && Module.HEAPF32 && Module.HEAPF32.buffer && Module.HEAPF32.buffer.byteLength > 0) return Module.HEAPF32;
      return null;
    }
    var sendMessage = resolveSendMessage();
    console.log('UNITEYE_JSLIB_STARTED for ' + objName);
    function send(x, y, blink, facePresent, features) {
      var heap = sharedBufPtr ? resolveHeapF32() : null;
      if (heap && sharedBufCap >= 6) {
        var base = sharedBufPtr >> 2;
        var featLen = (facePresent && features && features.length) ? Math.min(features.length, sharedBufCap - 6) : 0;
        heap[base + 1] = x;
        heap[base + 2] = y;
        heap[base + 3] = facePresent ? 1 : 0;
        heap[base + 4] = blink ? 1 : 0;
        heap[base + 5] = featLen;
        for (var i = 0; i < featLen; i++) heap[base + 6 + i] = features[i];
        seq = (seq + 1) | 0;
        if (seq === 0) seq = 1;
        heap[base] = seq;
        return;
      }
      if (!sendMessage) return;
      var msg = x.toFixed(2) + ',' + y.toFixed(2) + ',' + (facePresent ? 1 : 0) + ',' + (blink ? 1 : 0) +
                (facePresent && features && features.length ? ',' + features.join(',') : '');
      sendMessage(objName, 'OnWebGaze', msg);
    }
    if (typeof window.UnitEyeStartPipeline === 'function') {
      Promise.resolve(window.UnitEyeStartPipeline(send)).catch(function (e) {
        var msg = (e && e.message) ? e.message : String(e);
        console.error('UNITEYE_PIPELINE_FAILED: ' + msg);
        if (sendMessage) sendMessage(objName, 'OnWebGazeError', msg);
      });
    } else {
      console.error('UnitEyeWebGL: window.UnitEyeStartPipeline not found. Include uniteye-webgl-boot.js (and uniteye-core.js / uniteye-cv.js) in your WebGL template.');
      if (sendMessage) sendMessage(objName, 'OnWebGazeError', 'UnitEyeStartPipeline not found (missing uniteye-webgl-boot.js in the template)');
    }
  },

  UnitEyeWebGL_SetPaused: function (paused) {
    if (typeof window !== 'undefined' && window.__unitEyeEngine && typeof window.__unitEyeEngine.setPaused === 'function') {
      window.__unitEyeEngine.setPaused(!!paused);
    }
  }
});
