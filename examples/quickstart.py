"""Tour of the low-level API: launch a headless game, step it, read and write game variables."""

from soccer_rl import P1, P1G, P2, GameProcess

with GameProcess(headless=True) as game:
    c = game.client

    # Fresh kickoff. Every reset option is optional; see Bridge.BeginReset for the full list.
    s = c.reset(seed=123, ai=1, ground=2, ball=2)  # built-in AI on P2, slippery pitch, beach ball
    print("bodies:", c.body_names[:6], "...", len(c.body_names), "total")
    print("ball (x, y, angle, vx, vy, w):", s.ball)

    # Hold P1's button (outfield player + goalie share it in one-button mode) for 30 ticks,
    # then release for 10. mask says which players we drive; P2 is left to the game's AI.
    me = (1 << P1) | (1 << P1G)
    for _ in range(5):
        s = c.step(buttons=me, mask=me, repeat=30)
        s = c.step(buttons=0, mask=me, repeat=10)
    print(f"tick {s.episode_tick}: score {s.p1_score}-{s.p2_score}, P1 torso at {s.body('P1/Torso')[:2]}")

    # Any game variable by path: PlayerPrefs, statics, components (private members too).
    print(c.get(
        "pref:P1Score",
        "static:UnityEngine.Physics2D.gravity",
        "obj:P1:Player.grounded",
        "obj:P2:Player.isAI",
        "obj:P1/Torso:Rigidbody2D.mass",
    ))

    # ...and write them: moon gravity, super-jump for P1.
    print(c.set_values({
        "static:UnityEngine.Physics2D.gravity": (0, -5),
        "obj:P1:Player.jumpPower2": 900,
    }))
    s = c.step(buttons=me, mask=me, repeat=40)
    print("P1 torso after a moon jump:", s.body("P1/Torso")[:2])

    # Explore what's in the scene.
    print("\n".join(c.tree().splitlines()[:15]), "\n...")

    # Drive both teams yourself (self-play): put P2's bits in the mask and reset with ai=0.
    c.reset(seed=1, ai=0)
    s = c.step(buttons=(1 << P1) | (1 << P2), mask=0b1111, repeat=20)
    print("self-play step ok, tick", s.episode_tick)
