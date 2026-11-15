using System;
using System.Collections.Generic;
using System.Globalization;
using BepInEx;
using BepInEx.Logging;
using HarmonyLib;

namespace SoccerRL
{
    /// <summary>
    /// SoccerRL: turns Soccer Physics into a lockstep RL environment.
    ///
    /// Launch the game with  -rl-port 5555  to enable the TCP bridge. Without that
    /// argument the plugin does nothing and the game plays normally.
    ///
    /// Other command-line options:
    ///   -rl-host 127.0.0.1         interface to listen on
    ///   -rl-no-wait                don't block the game until a client connects
    ///   -rl-exit-on-disconnect     quit the game when the client goes away
    ///   -rl-ticks-per-frame N      physics ticks simulated per rendered frame
    ///   -rl-fps N                  targetFrameRate while a client is connected (-1 = unlimited)
    /// </summary>
    [BepInPlugin(Guid, Name, Version)]
    public class Plugin : BaseUnityPlugin
    {
        public const string Guid = "zinedine.soccerrl";
        public const string Name = "SoccerRL";
        public const string Version = "0.1.0";

        internal static ManualLogSource Log;
        internal static Bridge Bridge;

        private void Awake()
        {
            Log = Logger;
            var args = CliArgs.Parse(Environment.GetCommandLineArgs());
            if (args.Port <= 0)
            {
                Log.LogInfo("SoccerRL loaded but inactive (pass -rl-port <port> to enable the RL bridge).");
                return;
            }

            Settings.TicksPerFrame = args.TicksPerFrame;
            Settings.Fps = args.Fps;
            // Silence the game from the first frame: AudioManager.Start plays the music at boot,
            // before any client connects and sets the mute preferences.
            UnityEngine.AudioListener.volume = 0f;
            UnityEngine.PlayerPrefs.SetInt("musicOff", 1);
            UnityEngine.PlayerPrefs.SetInt("sfxOff", 1);

            new Harmony(Guid).PatchAll(typeof(Plugin).Assembly);
            Bridge = new Bridge(args);
            Bridge.Install();
            Log.LogInfo($"SoccerRL bridge listening on {args.Host}:{args.Port} (wait={args.WaitForClient}, exitOnDisconnect={args.ExitOnDisconnect})");
        }
    }

    internal class CliArgs
    {
        public string Host = "127.0.0.1";
        public int Port;
        public bool WaitForClient = true;
        public bool ExitOnDisconnect;
        public int TicksPerFrame = 1;
        public int Fps = -1;

        public static CliArgs Parse(string[] argv)
        {
            var a = new CliArgs();
            var env = Environment.GetEnvironmentVariable("SOCCER_RL_PORT");
            if (!string.IsNullOrEmpty(env)) int.TryParse(env, out a.Port);
            for (int i = 0; i < argv.Length; i++)
            {
                string next = i + 1 < argv.Length ? argv[i + 1] : null;
                switch (argv[i])
                {
                    case "-rl-port": a.Port = int.Parse(next, CultureInfo.InvariantCulture); i++; break;
                    case "-rl-host": a.Host = next; i++; break;
                    case "-rl-no-wait": a.WaitForClient = false; break;
                    case "-rl-exit-on-disconnect": a.ExitOnDisconnect = true; break;
                    case "-rl-ticks-per-frame": a.TicksPerFrame = Math.Max(1, int.Parse(next, CultureInfo.InvariantCulture)); i++; break;
                    case "-rl-fps": a.Fps = int.Parse(next, CultureInfo.InvariantCulture); i++; break;
                }
            }
            return a;
        }
    }

    /// <summary>Runtime-tunable knobs (shared by the bridge and the Harmony patches).</summary>
    internal static class Settings
    {
        /// <summary>Skip the slow-motion goal replay / out-of-bounds pause and reload the pitch immediately.</summary>
        public static bool FastRestart = true;
        /// <summary>Reload the pitch after a goal / out-of-bounds. Off for single-goal episodes, where the
        /// client's reset reloads it anyway (saves one scene load per episode).</summary>
        public static bool AutoReload = true;
        /// <summary>Skip the ~0.7s blinking-ball intro at every kickoff.</summary>
        public static bool FastSpawn = true;
        /// <summary>Silence all audio.</summary>
        public static bool Mute = true;
        /// <summary>Physics ticks per rendered frame (1 = render every tick).</summary>
        public static int TicksPerFrame = 1;
        /// <summary>Application.targetFrameRate while a client is connected. -1 = as fast as possible.</summary>
        public static int Fps = -1;

        /// <summary>PlayerPrefs that are forced after the game's own (random) choices, e.g. match modifiers.</summary>
        public static readonly Dictionary<string, int> PrefOverrides = new Dictionary<string, int>();

        /// <summary>Base fixed timestep of the game (it uses timeScale * 0.02).</summary>
        public const float BaseStep = 0.02f;

        public static void ApplyOverrides()
        {
            foreach (var kv in PrefOverrides) UnityEngine.PlayerPrefs.SetInt(kv.Key, kv.Value);
        }

        public static string Describe()
        {
            return $"fast_restart={(FastRestart ? 1 : 0)};auto_reload={(AutoReload ? 1 : 0)};fast_spawn={(FastSpawn ? 1 : 0)};mute={(Mute ? 1 : 0)};" +
                   $"ticks_per_frame={TicksPerFrame};fps={Fps}";
        }
    }
}
