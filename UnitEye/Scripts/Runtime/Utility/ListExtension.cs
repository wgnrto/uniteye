using System.Collections.Generic;

public static class ListExtension
{
    /// <summary>
    /// Randomly shuffles the list.
    /// </summary>
    public static void Shuffle<T>(this IList<T> list)
    {
        var count = list.Count;
        var last = count - 1;

        for (int i = 0; i < last; i++)
        {
            var rand = UnityEngine.Random.Range(i, count);
            var tmp = list[i];
            list[i] = list[rand];
            list[rand] = tmp;
        }
    }
}
