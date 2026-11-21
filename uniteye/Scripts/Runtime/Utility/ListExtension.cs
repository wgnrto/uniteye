using System.Collections.Generic;
namespace UnitEye
{

    public static class ListExtension
    {
        [System.ThreadStatic]
        private static System.Random _rng;

        /// <summary>
        /// Randomly shuffles the list without mutating global UnityEngine.Random state.
        /// </summary>
        public static void Shuffle<T>(this IList<T> list)
        {
            if (list == null || list.Count <= 1) return;
            var rng = _rng ?? (_rng = new System.Random());
            var count = list.Count;
            var last = count - 1;

            for (int i = 0; i < last; i++)
            {
                var rand = rng.Next(i, count);
                var tmp = list[i];
                list[i] = list[rand];
                list[rand] = tmp;
            }
        }
    }
}
