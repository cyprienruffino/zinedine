using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Reflection;
using System.Text;
using HarmonyLib;
using UnityEngine;
using UnityEngine.SceneManagement;
using Object = UnityEngine.Object;

namespace SoccerRL
{
    /// <summary>
    /// Read/write arbitrary game state by path. One query per line. Supported forms:
    ///
    ///   pref:P1Score                         PlayerPrefs int
    ///   static:UnityEngine.Time.timeScale    static field/property (type resolved by full or short name)
    ///   obj:P1/Torso:Rigidbody2D.velocity    member of a component on a GameObject (GameObject.Find path)
    ///   obj:P1:Player.jumpPower1             private members work too
    ///   obj:P1:Player.pressKey()             zero-argument method call (get only)
    ///
    /// Setting uses "query=value". Vectors are written "x,y" or "x,y,z"; bools as 0/1/true/false.
    /// Results come back one line per query, errors are prefixed with "!".
    /// </summary>
    internal static class Introspect
    {
        private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

        public static string GetMany(string queries)
        {
            var sb = new StringBuilder();
            foreach (var q in SplitLines(queries))
            {
                try { sb.Append(Format(Get(q))); }
                catch (Exception e) { sb.Append('!').Append(Flatten(e)); }
                sb.Append('\n');
            }
            return sb.ToString();
        }

        public static string SetMany(string assignments)
        {
            var sb = new StringBuilder();
            foreach (var line in SplitLines(assignments))
            {
                try
                {
                    int eq = line.LastIndexOf('=');
                    if (eq <= 0) throw new ArgumentException("expected path=value");
                    Set(line.Substring(0, eq).Trim(), line.Substring(eq + 1).Trim());
                    sb.Append("ok");
                }
                catch (Exception e) { sb.Append('!').Append(Flatten(e)); }
                sb.Append('\n');
            }
            return sb.ToString();
        }

        private static IEnumerable<string> SplitLines(string s) =>
            (s ?? "").Split('\n').Select(l => l.Trim()).Where(l => l.Length > 0);

        private static string Flatten(Exception e)
        {
            while (e is TargetInvocationException && e.InnerException != null) e = e.InnerException;
            return e.GetType().Name + ": " + e.Message.Replace('\n', ' ');
        }

        // ------------------------------------------------------------------------------------------

        private static object Get(string q)
        {
            if (q.StartsWith("pref:")) return PlayerPrefs.GetInt(q.Substring(5));
            Resolve(q, out var target, out var type, out var member);
            if (member.EndsWith("()"))
            {
                var m = AccessTools.Method(type, member.Substring(0, member.Length - 2), Type.EmptyTypes)
                        ?? throw new MissingMemberException(type.Name, member);
                return m.Invoke(target, null);
            }
            var f = AccessTools.Field(type, member);
            if (f != null) return f.GetValue(target);
            var p = AccessTools.Property(type, member) ?? throw new MissingMemberException(type.Name, member);
            return p.GetValue(target, null);
        }

        private static void Set(string q, string value)
        {
            if (q.StartsWith("pref:")) { PlayerPrefs.SetInt(q.Substring(5), int.Parse(value, Inv)); return; }
            Resolve(q, out var target, out var type, out var member);
            var f = AccessTools.Field(type, member);
            if (f != null) { f.SetValue(target, Parse(value, f.FieldType)); return; }
            var p = AccessTools.Property(type, member) ?? throw new MissingMemberException(type.Name, member);
            p.SetValue(target, Parse(value, p.PropertyType), null);
        }

        /// <summary>Resolve "static:Type.member" or "obj:Path:Component.member[.member...]".</summary>
        private static void Resolve(string q, out object target, out Type type, out string member)
        {
            string chain;
            if (q.StartsWith("static:"))
            {
                string rest = q.Substring(7);
                // Longest prefix that names a type.
                int dot = rest.LastIndexOf('.');
                type = null;
                while (dot > 0)
                {
                    type = FindType(rest.Substring(0, dot));
                    if (type != null) break;
                    dot = rest.LastIndexOf('.', dot - 1);
                }
                if (type == null) throw new TypeLoadException("no type in " + rest);
                target = null;
                chain = rest.Substring(dot + 1);
            }
            else if (q.StartsWith("obj:"))
            {
                string rest = q.Substring(4);
                int colon = rest.IndexOf(':');
                if (colon < 0) throw new ArgumentException("expected obj:Path:Component.member");
                string path = rest.Substring(0, colon);
                string compAndMember = rest.Substring(colon + 1);
                int dot = compAndMember.IndexOf('.');
                if (dot < 0) throw new ArgumentException("expected Component.member");
                string compName = compAndMember.Substring(0, dot);
                chain = compAndMember.Substring(dot + 1);

                var go = FindGameObject(path) ?? throw new ArgumentException("no GameObject " + path);
                object comp;
                if (compName == "GameObject") comp = go;
                else if (compName == "Transform") comp = go.transform;
                else
                {
                    var ct = FindType(compName) ?? throw new TypeLoadException("no type " + compName);
                    comp = go.GetComponent(ct);
                    if ((Object)comp == null) throw new ArgumentException($"{path} has no {compName}");
                }
                target = comp;
                type = comp.GetType();
            }
            else throw new ArgumentException("query must start with pref:, static: or obj:");

            // Walk intermediate members: a.b.c → target = value(a.b), member = c
            var parts = chain.Split('.');
            for (int i = 0; i < parts.Length - 1; i++)
            {
                var f = AccessTools.Field(type, parts[i]);
                if (f != null) target = f.GetValue(target);
                else
                {
                    var p = AccessTools.Property(type, parts[i]) ?? throw new MissingMemberException(type.Name, parts[i]);
                    target = p.GetValue(target, null);
                }
                if (target == null) throw new NullReferenceException(parts[i] + " is null");
                type = target.GetType();
            }
            member = parts[parts.Length - 1];
        }

        private static readonly Dictionary<string, Type> TypeCache = new Dictionary<string, Type>();

        private static Type FindType(string name)
        {
            if (TypeCache.TryGetValue(name, out var t)) return t;
            t = AccessTools.TypeByName(name);
            if (t == null)
            {
                foreach (var ns in new[] { "UnityEngine.", "System." })
                {
                    t = AccessTools.TypeByName(ns + name);
                    if (t != null) break;
                }
            }
            TypeCache[name] = t;
            return t;
        }

        /// <summary>GameObject.Find, falling back to a search that includes inactive objects.</summary>
        private static GameObject FindGameObject(string path)
        {
            var go = GameObject.Find(path);
            if (go != null) return go;
            var parts = path.Split('/');
            foreach (var root in SceneManager.GetActiveScene().GetRootGameObjects())
            {
                if (root.name != parts[0]) continue;
                var t = root.transform;
                for (int i = 1; i < parts.Length && t != null; i++) t = t.Find(parts[i]);
                if (t != null) return t.gameObject;
            }
            return null;
        }

        // ------------------------------------------------------------------------------------------

        private static object Parse(string s, Type t)
        {
            if (t == typeof(string)) return s;
            if (t == typeof(bool)) return s == "1" || s.Equals("true", StringComparison.OrdinalIgnoreCase);
            if (t.IsEnum) return Enum.Parse(t, s, true);
            float[] F() => s.Trim('(', ')').Split(',').Select(x => float.Parse(x.Trim(), Inv)).ToArray();
            if (t == typeof(Vector2)) { var v = F(); return new Vector2(v[0], v[1]); }
            if (t == typeof(Vector3)) { var v = F(); return new Vector3(v[0], v[1], v.Length > 2 ? v[2] : 0f); }
            if (t == typeof(Color)) { var v = F(); return new Color(v[0], v[1], v[2], v.Length > 3 ? v[3] : 1f); }
            return Convert.ChangeType(s, t, Inv);
        }

        private static string Format(object o)
        {
            switch (o)
            {
                case null: return "null";
                case float f: return f.ToString("R", Inv);
                case double d: return d.ToString("R", Inv);
                case bool b: return b ? "1" : "0";
                case Vector2 v: return $"{v.x.ToString("R", Inv)},{v.y.ToString("R", Inv)}";
                case Vector3 v: return $"{v.x.ToString("R", Inv)},{v.y.ToString("R", Inv)},{v.z.ToString("R", Inv)}";
                case Color c: return $"{c.r.ToString("R", Inv)},{c.g.ToString("R", Inv)},{c.b.ToString("R", Inv)},{c.a.ToString("R", Inv)}";
                case Object uo: return uo == null ? "null" : uo.name + " (" + uo.GetType().Name + ")";
                case IFormattable fm: return fm.ToString(null, Inv);
                default: return o.ToString().Replace('\n', ' ');
            }
        }

        // ------------------------------------------------------------------------------------------

        /// <summary>Indented dump of the active scene: name [tag] (active) @pos : components.</summary>
        public static string Tree()
        {
            var sb = new StringBuilder();
            var scene = SceneManager.GetActiveScene();
            sb.Append("# scene ").Append(scene.name).Append('\n');
            foreach (var root in scene.GetRootGameObjects()) Dump(root.transform, 0, sb);
            return sb.ToString();
        }

        private static void Dump(Transform t, int depth, StringBuilder sb)
        {
            sb.Append(' ', depth * 2).Append(t.name);
            if (t.tag != "Untagged") sb.Append(" [").Append(t.tag).Append(']');
            if (!t.gameObject.activeInHierarchy) sb.Append(" (inactive)");
            var p = t.position;
            sb.Append(" @").Append(p.x.ToString("0.###", Inv)).Append(',').Append(p.y.ToString("0.###", Inv));
            var comps = t.GetComponents<Component>()
                .Where(c => c != null && !(c is Transform))
                .Select(c => c.GetType().Name);
            sb.Append(" : ").Append(string.Join(",", comps)).Append('\n');
            for (int i = 0; i < t.childCount; i++) Dump(t.GetChild(i), depth + 1, sb);
        }
    }
}
