// UnitEye WebGL bridge: starts the browser gaze pipeline and streams gaze into Unity.
// Called from WebGLGazeReceiver.cs via [DllImport("__Internal")] UnitEyeWebGL_Start(gameObjectName).
//
// Requires the WebGL page/template to define window.UnitEyeStartPipeline(send) — see
// webgl/uniteye-webgl-boot.js (include it, uniteye-core.js and uniteye-cv.js in your WebGL template,
// and serve models/eyemu.onnx alongside the build). The pipeline calls send(x, y, blink, face, features)
// each frame; this forwards it to the named GameObject's OnWebGaze method.
mergeInto(LibraryManager.library, {
  UnitEyeWebGL_Start: function (objNamePtr) {
    var objName = UTF8ToString(objNamePtr);
    // Resolve SendMessage across Unity WebGL runtime variants: jslib-scope global (older),
    // Module.SendMessage (framework export), or a page-level unityInstance.
    function resolveSendMessage() {
      if (typeof SendMessage === 'function') return SendMessage;
      if (typeof Module !== 'undefined' && typeof Module.SendMessage === 'function') return Module.SendMessage;
      if (typeof window !== 'undefined' && window.unityInstance && typeof window.unityInstance.SendMessage === 'function')
        return function (o, m, v) { window.unityInstance.SendMessage(o, m, v); };
      return null;
    }
    var sendMessage = resolveSendMessage();
    if (!sendMessage) {
      console.error('UnitEyeWebGL: no SendMessage available (loader variant not recognized).');
      return;
    }
    console.log('UNITEYE_JSLIB_STARTED for ' + objName);
    function send(x, y, blink, facePresent, features) {
      var msg = x.toFixed(2) + ',' + y.toFixed(2) + ',' + (facePresent ? 1 : 0) + ',' + (blink ? 1 : 0) +
                (features && features.length ? ',' + features.join(',') : '');
      sendMessage(objName, 'OnWebGaze', msg);
    }
    if (typeof window.UnitEyeStartPipeline === 'function') {
      // The pipeline start is async (CDN imports, getUserMedia, model load) — surface failures both to
      // the console and into Unity so a denied camera / 404 model is diagnosable instead of silent.
      Promise.resolve(window.UnitEyeStartPipeline(send)).catch(function (e) {
        var msg = (e && e.message) ? e.message : String(e);
        console.error('UNITEYE_PIPELINE_FAILED: ' + msg);
        sendMessage(objName, 'OnWebGazeError', msg);
      });
    } else {
      console.error('UnitEyeWebGL: window.UnitEyeStartPipeline not found. Include uniteye-webgl-boot.js (and uniteye-core.js / uniteye-cv.js) in your WebGL template.');
      sendMessage(objName, 'OnWebGazeError', 'UnitEyeStartPipeline not found (missing uniteye-webgl-boot.js in the template)');
    }
  }
});
