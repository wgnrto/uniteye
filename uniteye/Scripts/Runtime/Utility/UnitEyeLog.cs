using System;
using UnityEngine;

namespace UnitEye
{
    /// <summary>
    /// Tiny logging façade for the package. Lets a host game silence UnitEye's console output at runtime
    /// (set <see cref="Enabled"/> = false) and gives pure-compute classes a single seam instead of being
    /// hard-bound to UnityEngine.Debug scattered call sites. Errors and exceptions always log so real
    /// failures are never hidden; Info/Warn respect the Enabled gate.
    /// </summary>
    public static class UnitEyeLog
    {
        /// <summary>When false, Info/Warn are suppressed. Errors and exceptions always log.</summary>
        public static bool Enabled = true;

        public static void Info(string message)
        {
            if (Enabled) Debug.Log(message);
        }

        public static void Warn(string message)
        {
            if (Enabled) Debug.LogWarning(message);
        }

        public static void Error(string message) => Debug.LogError(message);

        public static void Exception(Exception e) => Debug.LogException(e);
    }
}
