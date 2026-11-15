using System.Collections;
using System.Linq;
using HarmonyLib;
using UnityEngine;

namespace SoccerRL
{
    /// <summary>
    /// Small gameplay patches that make the game RL-friendly. They are only applied when the
    /// bridge is active (Plugin.Awake calls PatchAll only with -rl-port), and each is gated on a Setting.
    /// </summary>
    [HarmonyPatch]
    internal static class Patches
    {
        // ---- Fast restart: the coroutines that wait before reloading the pitch take a waitTime arg. ----

        [HarmonyPatch(typeof(GoalParent), "resetScene"), HarmonyPrefix]
        private static bool GoalResetScene(ref float waitTime, ref IEnumerator __result) => Reload(ref waitTime, ref __result);

        [HarmonyPatch(typeof(GoalParent), "asyncReset"), HarmonyPrefix]
        private static bool GoalAsyncReset(ref float waitTime, ref IEnumerator __result) => Reload(ref waitTime, ref __result);

        [HarmonyPatch(typeof(GoalParent), "gameOver"), HarmonyPrefix]
        private static void GoalGameOver(ref float waitTime) { if (Settings.FastRestart) waitTime = 0f; }

        [HarmonyPatch(typeof(Ball), nameof(Ball.resetScene)), HarmonyPrefix]
        private static bool BallResetScene(ref float waitTime, ref IEnumerator __result) => Reload(ref waitTime, ref __result);

        [HarmonyPatch(typeof(Ball), nameof(Ball.asyncReset)), HarmonyPrefix]
        private static bool BallAsyncReset(ref float waitTime, ref IEnumerator __result) => Reload(ref waitTime, ref __result);

        /// <summary>Shared prefix of the reload coroutines: skip the wait, or skip the reload entirely.</summary>
        private static bool Reload(ref float waitTime, ref IEnumerator __result)
        {
            if (!Settings.AutoReload)
            {
                __result = Enumerable.Empty<object>().GetEnumerator();
                return false;
            }
            if (Settings.FastRestart) waitTime = 0f;
            return true;
        }

        // ---- Fast spawn: Ball.Start starts a coroutine that blinks the ball for ~0.7s before releasing it.
        // Its synchronous prefix (positioning the ball) has already run when Start returns; we stop the rest
        // and release the ball right away.

        private static readonly AccessTools.FieldRef<Ball, GameObject> BallGraphicGO =
            AccessTools.FieldRefAccess<Ball, GameObject>("graphicGO");

        [HarmonyPatch(typeof(Ball), "Start"), HarmonyPostfix]
        private static void BallStart(Ball __instance)
        {
            if (!Settings.FastSpawn || PlayerPrefs.GetInt("JustStarted") != 0) return;
            __instance.StopAllCoroutines();
            if (__instance.r != null) __instance.r.isKinematic = false;
            var g = BallGraphicGO(__instance);
            if (g != null) g.SetActive(true);
        }

        // ---- Modifier overrides: Game.Start re-rolls match modifiers into PlayerPrefs; force ours after it.

        [HarmonyPatch(typeof(Game), "Start"), HarmonyPostfix]
        private static void GameStart() => Settings.ApplyOverrides();

        // ---- Mute: route every sound through one check.

        [HarmonyPatch(typeof(AudioManager), nameof(AudioManager.PlaySound)), HarmonyPrefix]
        private static bool PlaySound() => !Settings.Mute;

        [HarmonyPatch(typeof(AudioManager), nameof(AudioManager.PlayBallSound)), HarmonyPrefix]
        private static bool PlayBallSound() => !Settings.Mute;

        [HarmonyPatch(typeof(AudioManager), nameof(AudioManager.PlayJumpSound)), HarmonyPrefix]
        private static bool PlayJumpSound() => !Settings.Mute;

        [HarmonyPatch(typeof(AudioManager), nameof(AudioManager.PlayWooshSound)), HarmonyPrefix]
        private static bool PlayWooshSound() => !Settings.Mute;

        [HarmonyPatch(typeof(AudioManager), nameof(AudioManager.PlayHop)), HarmonyPrefix]
        private static bool PlayHop() => !Settings.Mute;

        [HarmonyPatch(typeof(AudioManager), nameof(AudioManager.PlaySoundDelayed)), HarmonyPrefix]
        private static bool PlaySoundDelayed() => !Settings.Mute;

        [HarmonyPatch(typeof(AudioManager), nameof(AudioManager.PlayMusic)), HarmonyPrefix]
        private static bool PlayMusic() => !Settings.Mute;
    }
}
