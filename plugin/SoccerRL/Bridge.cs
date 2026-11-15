using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using UnityEngine.LowLevel;
using UnityEngine.SceneManagement;
using Object = UnityEngine.Object;
using Random = UnityEngine.Random;

namespace SoccerRL
{
    /// <summary>Marker type for our PlayerLoop entry.</summary>
    public struct SoccerRLTick { }

    /// <summary>
    /// Lockstep bridge between the game and an external controller.
    ///
    /// We inject a callback at the very start of Unity's FixedUpdate phase. On every physics tick that
    /// callback runs before any game script's FixedUpdate and before the physics step. When a step has
    /// finished, it sends an observation and then blocks the main thread until the client sends the next
    /// command. The game therefore advances exactly as fast as the controller asks it to, and wall-clock
    /// time never leaks into the simulation (Time.captureDeltaTime pins the frame delta).
    ///
    /// Wire protocol (little-endian). Client → server, 1 opcode byte followed by:
    ///   'S' u8 buttons, u8 mask, u16 repeat   step: hold `buttons` for `repeat` ticks on players in `mask`
    ///   'R' str options                       reset to a fresh kickoff (see BeginReset), replies State
    ///   'O'                                   observe: reply State without stepping
    ///   'I'                                   info: reply str (schema, settings, prefs)
    ///   'G' str queries                       get values by path (newline-separated), reply str
    ///   'X' str assignments                   set values by path ("path=value" lines), reply str
    ///   'C' str settings                      configure runtime settings, reply str
    ///   'P' u16 w, u16 h                      render cameras to an RGB24 image, reply u32 len, u16 w, u16 h, bytes
    ///   'T'                                   dump the scene hierarchy, reply str
    ///   'Q'                                   quit the game
    /// where str = u32 byte-length + UTF-8. Player bit order in buttons/mask: 0=P1, 1=P1G, 2=P2, 3=P2G.
    /// State = u32 byte-length + payload, see WriteState.
    /// </summary>
    internal class Bridge
    {
        // Event bits (accumulated between two State messages).
        private const int EvGoalP1 = 1 << 0;       // P1 (blue, right side) scored
        private const int EvGoalP2 = 1 << 1;       // P2 (red, left side) scored
        private const int EvOutOfBounds = 1 << 2;  // ball left the arena
        private const int EvSceneLoaded = 1 << 3;  // pitch was reloaded (after a goal / OOB / reset)
        private const int EvGameOver = 1 << 4;     // match just ended (someone reached 5)

        // State flag bits (instantaneous).
        private const int FlInMatch = 1 << 0;
        private const int FlBallLive = 1 << 1;
        private const int FlGameOver = 1 << 2;
        private const int FlBallGrounded = 1 << 3;
        private const int FlBallTouchingP1 = 1 << 4;
        private const int FlBallTouchingP2 = 1 << 5;

        public static readonly string[] PlayerNames = { "P1", "P1G", "P2", "P2G" };

        private readonly CliArgs args;
        private TcpListener listener;
        private TcpClient client;
        private NetworkStream stream;
        private BinaryReader reader;
        private BufferedStream outStream;
        private BinaryWriter writer;
        private readonly MemoryStream payload = new MemoryStream(4096);
        private BinaryWriter payloadWriter;
        private bool everConnected;

        // Simulation bookkeeping.
        private long globalTick;
        private int episodeTick;
        private int stepTicksLeft;
        private bool responsePending;
        private bool resetPending;
        private int? pendingSeed;
        private bool sceneDirty;
        private int events;
        private int lastP1Score, lastP2Score;
        private bool lastOob, lastGameOver;

        // Control.
        private byte buttons, mask, applied;

        // Bound scene objects.
        private readonly Player[] players = new Player[4];
        private Ball ball;
        private readonly List<Rigidbody2D> bodies = new List<Rigidbody2D>();
        private readonly List<string> bodyNames = new List<string>();

        private static readonly HarmonyLib.AccessTools.FieldRef<Player, bool> PlayerJumped =
            HarmonyLib.AccessTools.FieldRefAccess<Player, bool>("jumped");
        // Hidden jump timers: a new jump needs jumpLimiter == 0 (counts down 10 ticks after each jump);
        // the second, weaker boost fires once jump2delay (3) has counted down while airborne.
        private static readonly HarmonyLib.AccessTools.FieldRef<Player, int> PlayerJumpLimiter =
            HarmonyLib.AccessTools.FieldRefAccess<Player, int>("jumpLimiter");
        private static readonly HarmonyLib.AccessTools.FieldRef<Player, float> PlayerJump2Delay =
            HarmonyLib.AccessTools.FieldRefAccess<Player, float>("jump2delay");

        private Texture2D captureTex;

        public Bridge(CliArgs args)
        {
            this.args = args;
            payloadWriter = new BinaryWriter(payload);
        }

        public void Install()
        {
            // A standalone player pauses whenever its window is unfocused (always the case under Xvfb
            // or when launched from a script). Lockstep control must never stall on that.
            Application.runInBackground = true;

            var loop = PlayerLoop.GetCurrentPlayerLoop();
            if (!InsertAtStartOf(ref loop, typeof(UnityEngine.PlayerLoop.FixedUpdate),
                    new PlayerLoopSystem { type = typeof(SoccerRLTick), updateDelegate = Tick }))
                throw new Exception("SoccerRL: could not find the FixedUpdate player-loop phase");
            PlayerLoop.SetPlayerLoop(loop);

            SceneManager.sceneLoaded += OnSceneLoaded;
            listener = new TcpListener(IPAddress.Parse(args.Host), args.Port);
            listener.Start();
        }

        private static bool InsertAtStartOf(ref PlayerLoopSystem root, Type phase, PlayerLoopSystem sys)
        {
            var subs = root.subSystemList;
            if (subs == null) return false;
            for (int i = 0; i < subs.Length; i++)
            {
                if (subs[i].type == phase)
                {
                    var list = new List<PlayerLoopSystem>(subs[i].subSystemList ?? new PlayerLoopSystem[0]);
                    list.Insert(0, sys);
                    subs[i].subSystemList = list.ToArray();
                    root.subSystemList = subs;
                    return true;
                }
            }
            return false;
        }

        public static bool DebugLog = Environment.GetEnvironmentVariable("SOCCER_RL_DEBUG") == "1";

        private void OnSceneLoaded(Scene scene, LoadSceneMode mode)
        {
            if (DebugLog) Plugin.Log.LogInfo($"DBG sceneLoaded {scene.name} resetPending={resetPending}");
            sceneDirty = true;
            if (resetPending)
            {
                if (pendingSeed.HasValue) Random.InitState(pendingSeed.Value);
                Settings.ApplyOverrides();
            }
        }

        // ------------------------------------------------------------------------------------------
        // Tick: called once per physics step, before game FixedUpdates and before the physics step.
        // ------------------------------------------------------------------------------------------

        private void Tick()
        {
            globalTick++;
            if (DebugLog && (globalTick < 20 || globalTick % 200 == 0)) Plugin.Log.LogInfo($"DBG tick {globalTick} client={(client != null)} dirty={sceneDirty} reset={resetPending} ts={Time.timeScale} fdt={Time.fixedDeltaTime} scene={SceneManager.GetActiveScene().name}");
            try
            {
                if (client == null && !TryAccept()) return; // free-running, nobody connected

                if (sceneDirty) Bind();
                DetectEvents();

                if (resetPending) return; // let the old scene run until our LoadScene completes

                if (stepTicksLeft > 0)
                {
                    stepTicksLeft--;
                    if (stepTicksLeft > 0)
                    {
                        ApplyButtons();
                        episodeTick++;
                        return;
                    }
                }

                if (responsePending)
                {
                    SendState();
                    responsePending = false;
                }

                CommandLoop();
            }
            catch (Exception e) when (e is IOException || e is SocketException || e is ObjectDisposedException)
            {
                Plugin.Log.LogWarning("SoccerRL client disconnected: " + e.Message);
                DropClient();
                if (args.ExitOnDisconnect) Application.Quit();
            }
            catch (Exception e)
            {
                Plugin.Log.LogError("SoccerRL tick error: " + e);
            }
        }

        private void CommandLoop()
        {
            while (true)
            {
                byte op = reader.ReadByte();
                switch ((char)op)
                {
                    case 'S':
                    {
                        buttons = reader.ReadByte();
                        mask = reader.ReadByte();
                        int repeat = reader.ReadUInt16();
                        stepTicksLeft = Math.Max(1, repeat);
                        ApplyButtons();
                        episodeTick++;
                        responsePending = true;
                        return; // let this tick simulate
                    }
                    case 'R':
                        BeginReset(ReadStr());
                        responsePending = true;
                        return;
                    case 'O':
                        SendState();
                        break;
                    case 'I':
                        WriteStr(Info());
                        break;
                    case 'G':
                        WriteStr(Introspect.GetMany(ReadStr()));
                        break;
                    case 'X':
                        WriteStr(Introspect.SetMany(ReadStr()));
                        break;
                    case 'C':
                        WriteStr(Configure(ReadStr()));
                        break;
                    case 'P':
                    {
                        int w = reader.ReadUInt16(), h = reader.ReadUInt16();
                        SendPixels(w, h);
                        break;
                    }
                    case 'T':
                        WriteStr(Introspect.Tree());
                        break;
                    case 'Q':
                        DropClient();
                        Application.Quit();
                        return;
                    default:
                        throw new IOException($"unknown opcode 0x{op:X2}");
                }
            }
        }

        // ------------------------------------------------------------------------------------------
        // Connection handling
        // ------------------------------------------------------------------------------------------

        private bool TryAccept()
        {
            if (listener == null) return false;
            bool block = args.WaitForClient && !everConnected;
            if (!block && !listener.Pending()) return false;

            client = listener.AcceptTcpClient();
            client.NoDelay = true;
            stream = client.GetStream();
            reader = new BinaryReader(stream);
            outStream = new BufferedStream(stream, 1 << 16);
            writer = new BinaryWriter(outStream);
            everConnected = true;
            responsePending = false;
            stepTicksLeft = 0;
            ApplyTimeSettings();
            sceneDirty = true;
            Plugin.Log.LogInfo("SoccerRL client connected from " + client.Client.RemoteEndPoint);
            return true;
        }

        private void DropClient()
        {
            try { client?.Close(); } catch { /* ignore */ }
            client = null; stream = null; reader = null; writer = null; outStream = null;
            responsePending = false;
            resetPending = false;
            stepTicksLeft = 0;
            // Back to normal real-time play.
            Time.captureDeltaTime = 0f;
            Application.targetFrameRate = 60;
        }

        private void ApplyTimeSettings()
        {
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = Settings.Fps;
            // One rendered frame advances Settings.TicksPerFrame fixed steps; the game always uses
            // fixedDeltaTime = timeScale * 0.02, so this holds regardless of the current timeScale.
            Time.captureDeltaTime = Settings.BaseStep * Settings.TicksPerFrame;
            Time.maximumDeltaTime = Mathf.Max(Time.maximumDeltaTime, Time.captureDeltaTime * 2f);
            AudioListener.volume = Settings.Mute ? 0f : 1f;
        }

        // ------------------------------------------------------------------------------------------
        // Scene binding & events
        // ------------------------------------------------------------------------------------------

        private void Bind()
        {
            sceneDirty = false;
            Array.Clear(players, 0, players.Length);
            ball = null;
            bodies.Clear();
            bodyNames.Clear();
            applied = 0;

            foreach (var p in Object.FindObjectsOfType<Player>())
            {
                int idx = Array.IndexOf(PlayerNames, p.name);
                if (idx >= 0) players[idx] = p;
            }
            for (int i = 0; i < players.Length; i++)
            {
                if (players[i] == null) continue;
                var root = players[i].transform;
                foreach (var rb in players[i].GetComponentsInChildren<Rigidbody2D>(true))
                {
                    bodies.Add(rb);
                    bodyNames.Add(PlayerNames[i] + "/" + RelativePath(root, rb.transform));
                }
            }
            ball = Object.FindObjectOfType<Ball>();
            if (ball != null && ball.r != null)
            {
                bodies.Add(ball.r);
                bodyNames.Add("Ball");
            }

            // Undo the game's 60 fps cap etc. on every load.
            if (client != null) ApplyTimeSettings();

            if (resetPending)
            {
                resetPending = false;
                episodeTick = 0;
                events = 0;
                SnapshotPrefs();
            }
            else
            {
                events |= EvSceneLoaded;
            }
        }

        private static string RelativePath(Transform root, Transform t)
        {
            var parts = new List<string>();
            while (t != null && t != root) { parts.Add(t.name); t = t.parent; }
            parts.Reverse();
            return string.Join("/", parts);
        }

        private void SnapshotPrefs()
        {
            lastP1Score = PlayerPrefs.GetInt("P1Score");
            lastP2Score = PlayerPrefs.GetInt("P2Score");
            lastOob = PlayerPrefs.GetInt("OutOfBounds") == 1;
            lastGameOver = PlayerPrefs.GetInt("GameOver") == 1;
        }

        private void DetectEvents()
        {
            int s1 = PlayerPrefs.GetInt("P1Score");
            int s2 = PlayerPrefs.GetInt("P2Score");
            bool oob = PlayerPrefs.GetInt("OutOfBounds") == 1;
            bool over = PlayerPrefs.GetInt("GameOver") == 1;
            if (s1 > lastP1Score) events |= EvGoalP1;
            if (s2 > lastP2Score) events |= EvGoalP2;
            if (oob && !lastOob) events |= EvOutOfBounds;
            if (over && !lastGameOver) events |= EvGameOver;
            lastP1Score = s1; lastP2Score = s2; lastOob = oob; lastGameOver = over;
        }

        // ------------------------------------------------------------------------------------------
        // Actions
        // ------------------------------------------------------------------------------------------

        private void ApplyButtons()
        {
            for (int i = 0; i < 4; i++)
            {
                int bit = 1 << i;
                if ((mask & bit) == 0) continue;
                var p = players[i];
                if (p == null) continue;
                bool down = (buttons & bit) != 0;
                bool was = (applied & bit) != 0;
                if (down) p.pressKey();         // the game calls pressKey every frame while held
                else if (was) p.releaseKey();   // and releaseKey once on key-up
                if (down) applied |= (byte)bit; else applied &= (byte)~bit;
            }
        }

        // ------------------------------------------------------------------------------------------
        // Reset
        // ------------------------------------------------------------------------------------------

        /// <summary>
        /// Options ("key=value" separated by ';' or newlines), all optional:
        ///   seed=N            seed UnityEngine.Random right after the scene loads (deterministic kickoffs)
        ///   ai=0|1            built-in AI drives P2 (default 1)
        ///   two_buttons=0|1   separate buttons for outfield player and goalie (default 0)
        ///   endless=0|1       no first-to-5 game over (default 0)
        ///   p1_starts=-1|0|1  which side kicks off (-1 = random)
        ///   p1_score=N, p2_score=N   starting score
        ///   ground, splits, mini_goals, big_goals, ball, headless   match modifiers; -1 = let the game roll them
        ///   + any runtime setting accepted by Configure (fast_restart, fast_spawn, mute, ...)
        /// </summary>
        private void BeginReset(string options)
        {
            var o = KV.Parse(options);
            Configure(options); // settings keys may be mixed into the reset options

            int? seed = o.ContainsKey("seed") ? (int?)KV.Int(o, "seed", 0) : null;
            var rng = seed.HasValue ? new System.Random(seed.Value) : new System.Random();

            int p1s = KV.Int(o, "p1_starts", -1);
            if (p1s < 0) p1s = rng.Next(2);

            PlayerPrefs.SetInt("GameOver", 0);
            PlayerPrefs.SetInt("Continued", 0);
            PlayerPrefs.SetInt("JustStarted", 0);
            PlayerPrefs.SetInt("OutOfBounds", 0);
            PlayerPrefs.SetInt("P1Score", KV.Int(o, "p1_score", 0));
            PlayerPrefs.SetInt("P2Score", KV.Int(o, "p2_score", 0));
            PlayerPrefs.SetInt("P1Starts", p1s);
            PlayerPrefs.SetInt("AI", KV.Int(o, "ai", 1));
            PlayerPrefs.SetInt("TwoButtons", KV.Int(o, "two_buttons", 0));
            PlayerPrefs.SetInt("Endless", KV.Int(o, "endless", 0));
            PlayerPrefs.SetInt("ReplaysOff", 1);
            PlayerPrefs.SetInt("isIOS", 0);
            PlayerPrefs.SetInt("bbOn", 0);
            PlayerPrefs.SetInt("LowG", 0);
            foreach (var k in new[] { "P1Ready", "P2Ready", "P1PReady", "P1GReady", "P2PReady", "P2GReady" })
                PlayerPrefs.SetInt(k, 1);
            if (Settings.Mute) { PlayerPrefs.SetInt("sfxOff", 1); PlayerPrefs.SetInt("musicOff", 1); }

            Settings.PrefOverrides.Clear();
            foreach (var m in Modifiers)
            {
                int v = KV.Int(o, m.Key, -1);
                if (v >= 0) Settings.PrefOverrides[m.Value] = v;
            }
            Settings.ApplyOverrides();

            pendingSeed = seed;
            applied = 0;
            buttons = 0;
            stepTicksLeft = 0;
            sceneDirty = false;
            resetPending = true;
            Time.timeScale = 0.85f; // in case we were frozen in a game-over screen
            Time.fixedDeltaTime = Time.timeScale * Settings.BaseStep;
            SceneManager.LoadScene("SceneMain");
        }

        /// <summary>Reset option name → PlayerPrefs key for match modifiers.</summary>
        public static readonly Dictionary<string, string> Modifiers = new Dictionary<string, string>
        {
            { "ground", "Ground" },         // 1 = normal, 2 = slippery (rain)
            { "splits", "Splits" },         // 1 = acrobatic
            { "mini_goals", "MiniGoals" },  // 1 = small goals
            { "big_goals", "BigGoals" },    // 1 = huge goals
            { "ball", "Ball" },             // 0 = soccer, 1 = american football, 2 = beach ball
            { "headless", "Headless" },     // 1 = no heads
        };

        private string Configure(string options)
        {
            var o = KV.Parse(options);
            if (o.Count == 0) return Settings.Describe();
            if (o.ContainsKey("fast_restart")) Settings.FastRestart = KV.Int(o, "fast_restart", 1) != 0;
            if (o.ContainsKey("auto_reload")) Settings.AutoReload = KV.Int(o, "auto_reload", 1) != 0;
            if (o.ContainsKey("fast_spawn")) Settings.FastSpawn = KV.Int(o, "fast_spawn", 1) != 0;
            if (o.ContainsKey("mute")) Settings.Mute = KV.Int(o, "mute", 1) != 0;
            if (o.ContainsKey("ticks_per_frame")) Settings.TicksPerFrame = Math.Max(1, KV.Int(o, "ticks_per_frame", 1));
            if (o.ContainsKey("fps")) Settings.Fps = KV.Int(o, "fps", -1);
            ApplyTimeSettings();
            return Settings.Describe();
        }

        // ------------------------------------------------------------------------------------------
        // Output
        // ------------------------------------------------------------------------------------------

        private string Info()
        {
            var sb = new StringBuilder();
            sb.Append("version=").Append(Plugin.Version).Append('\n');
            sb.Append("scene=").Append(SceneManager.GetActiveScene().name).Append('\n');
            sb.Append("bodies=").Append(string.Join(",", bodyNames)).Append('\n');
            sb.Append("players=").Append(string.Join(",", PlayerNames)).Append('\n');
            sb.Append("player_fields=grounded,jumped,ai_triggered,button,jump_limiter,jump2_delay\n");
            sb.Append("body_fields=x,y,angle,vx,vy,angular_velocity\n");
            sb.Append("settings=").Append(Settings.Describe()).Append('\n');
            sb.Append("prefs=");
            sb.Append(string.Join(",", new[] { "AI", "TwoButtons", "Endless", "P1Score", "P2Score", "P1Starts",
                    "Ground", "Splits", "MiniGoals", "BigGoals", "Ball", "Headless" }
                .Select(k => k + ":" + PlayerPrefs.GetInt(k).ToString(CultureInfo.InvariantCulture))));
            sb.Append('\n');
            var ov = Settings.PrefOverrides.Select(kv => kv.Key + ":" + kv.Value);
            sb.Append("overrides=").Append(string.Join(",", ov)).Append('\n');
            return sb.ToString();
        }

        /// <summary>
        /// State payload layout:
        ///   i64 global_tick, i32 episode_tick, i32 p1_score, i32 p2_score, i32 events, i32 flags,
        ///   f32 time_scale, f32 fixed_dt, i32 n_bodies,
        ///   n_bodies × (f32 x, y, angle_deg, vx, vy, angular_velocity_deg),
        ///   4 players × (f32 grounded, jumped, ai_triggered, button, jump_limiter / 10, jump2_delay / 3)
        /// </summary>
        private void SendState()
        {
            payload.SetLength(0);
            var w = payloadWriter;
            int flags = 0;
            bool inMatch = players[0] != null;
            if (inMatch) flags |= FlInMatch;
            if (ball != null && ball.r != null)
            {
                if (!ball.r.isKinematic) flags |= FlBallLive;
                if (ball.grounded) flags |= FlBallGrounded;
                if (ball.touchingP1) flags |= FlBallTouchingP1;
                if (ball.touchingP2) flags |= FlBallTouchingP2;
            }
            if (lastGameOver) flags |= FlGameOver;

            w.Write(globalTick);
            w.Write(episodeTick);
            w.Write(lastP1Score);
            w.Write(lastP2Score);
            w.Write(events);
            w.Write(flags);
            w.Write(Time.timeScale);
            w.Write(Time.fixedDeltaTime);
            w.Write(bodies.Count);
            foreach (var rb in bodies)
            {
                if (rb == null) { for (int k = 0; k < 6; k++) w.Write(0f); continue; }
                var pos = rb.position;
                var vel = rb.velocity;
                w.Write(pos.x); w.Write(pos.y); w.Write(rb.rotation);
                w.Write(vel.x); w.Write(vel.y); w.Write(rb.angularVelocity);
            }
            for (int i = 0; i < 4; i++)
            {
                var p = players[i];
                if (p == null) { for (int k = 0; k < 6; k++) w.Write(0f); continue; }
                w.Write(p.grounded ? 1f : 0f);
                w.Write(PlayerJumped(p) ? 1f : 0f);
                w.Write(p.aiTriggered ? 1f : 0f);
                w.Write((applied & (1 << i)) != 0 ? 1f : 0f);
                w.Write(PlayerJumpLimiter(p) / 10f);
                w.Write(PlayerJump2Delay(p) / 3f);
            }
            w.Flush();

            writer.Write((uint)payload.Length);
            writer.Write(payload.GetBuffer(), 0, (int)payload.Length);
            writer.Flush();
            events = 0;
        }

        private void SendPixels(int w, int h)
        {
            w = Mathf.Clamp(w, 1, 4096);
            h = Mathf.Clamp(h, 1, 4096);
            byte[] data = null;
            try
            {
                if (captureTex == null || captureTex.width != w || captureTex.height != h)
                    captureTex = new Texture2D(w, h, TextureFormat.RGB24, false);
                var rt = RenderTexture.GetTemporary(w, h, 24, RenderTextureFormat.ARGB32);
                var cams = Camera.allCameras.Where(c => c.enabled).OrderBy(c => c.depth).ToArray();
                foreach (var cam in cams)
                {
                    var prev = cam.targetTexture;
                    cam.targetTexture = rt;
                    cam.Render();
                    cam.targetTexture = prev;
                }
                var prevActive = RenderTexture.active;
                RenderTexture.active = rt;
                captureTex.ReadPixels(new Rect(0, 0, w, h), 0, 0, false);
                RenderTexture.active = prevActive;
                RenderTexture.ReleaseTemporary(rt);
                data = captureTex.GetRawTextureData();
            }
            catch (Exception e)
            {
                Plugin.Log.LogWarning("SoccerRL pixel capture failed: " + e.Message);
            }
            if (data == null) { writer.Write(0u); writer.Write((ushort)0); writer.Write((ushort)0); }
            else
            {
                writer.Write((uint)data.Length); writer.Write((ushort)w); writer.Write((ushort)h);
                writer.Write(data);
            }
            writer.Flush();
        }

        private string ReadStr()
        {
            int n = (int)reader.ReadUInt32();
            return n == 0 ? "" : Encoding.UTF8.GetString(reader.ReadBytes(n));
        }

        private void WriteStr(string s)
        {
            var b = Encoding.UTF8.GetBytes(s ?? "");
            writer.Write((uint)b.Length);
            writer.Write(b);
            writer.Flush();
        }
    }

    /// <summary>Tiny "k=v;k=v" parser.</summary>
    internal static class KV
    {
        public static Dictionary<string, string> Parse(string s)
        {
            var d = new Dictionary<string, string>();
            if (string.IsNullOrEmpty(s)) return d;
            foreach (var part in s.Split(new[] { ';', '\n' }, StringSplitOptions.RemoveEmptyEntries))
            {
                int eq = part.IndexOf('=');
                if (eq <= 0) continue;
                d[part.Substring(0, eq).Trim()] = part.Substring(eq + 1).Trim();
            }
            return d;
        }

        public static int Int(Dictionary<string, string> d, string k, int def)
        {
            return d.TryGetValue(k, out var v) && int.TryParse(v, NumberStyles.Integer, CultureInfo.InvariantCulture, out var r) ? r : def;
        }
    }
}
