#!/usr/bin/env python3
"""simlab2: five more Reel/Short formats modelled on proven viral ball-sim types (original sims, own look).
Kinds: polyfit (polygon shrinks each bounce, one note per bounce, ends in a perfect fit),
escape (several balls in rotating rings with gaps, which escapes first),
squeeze (steel bars crush balls, each crushed ball becomes 4, ring shatters),
mirror (ellipse focus property: every ray leaves one focus and arrives at the other together),
mine (four miners, four attitudes, one reaches the diamond).
Usage: simlab2.py --kind polyfit --pal bb00 --title "A|B" --T 32 --seed 3 --out out/x.mp4 [--sides 3 --tune ode]
Imports helpers from simlab.py (same folder).
"""
import argparse, bisect, json, math, os, random, sys
import numpy as np, cv2
from scipy.io import wavfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simlab as S
from simlab import (W, H, FPS, SR, CX, CY, PALS, TUNES, hexbgr, grad, bell, text_layer, blit, blit_fade, fade,
                    title_block, make_bg, draw_ball, encode, OVERLAY, note_name)

# ---------------- audio helpers ----------------
def noise_burst(dur, decay, lo=0.0, seed=0):
    rng = np.random.RandomState(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    y = rng.randn(n) * np.exp(-t * decay)
    if lo:
        y = np.convolve(y, np.ones(int(lo)) / lo, mode="same")
    return y


def thunk(dur=0.18):
    t = np.arange(int(dur * SR)) / SR
    return (np.sin(2 * np.pi * 95 * t * np.exp(-t * 9)) * np.exp(-t * 22) + 0.35 * noise_burst(dur, 40, 3, 1))


def mix(events, T, path, gain=0.9):
    """events: (t, kind, a, b): kind 'bell' a=midi b=vel | 'noise' a=dur b=vel | 'thunk' a=_ b=vel | 'raw' a=array b=vel"""
    buf = np.zeros(int((T + 1.5) * SR))
    cache = {}
    for t, kind, a, b in events:
        if kind == "bell":
            s = bell(int(a))
        elif kind == "thunk":
            s = cache.setdefault("thunk", thunk())
        elif kind == "noise":
            key = ("n", a)
            s = cache.setdefault(key, noise_burst(a, 5.0 / a, 6, int(a * 100)))
        else:
            s = a
        i0 = int(t * SR)
        if i0 < 0 or i0 >= len(buf):
            continue
        s = s[: len(buf) - i0]
        buf[i0:i0 + len(s)] += s * b
    buf = np.tanh(buf * gain) * 0.85
    buf = buf[: int((T + 1.0) * SR)]
    n = int(0.05 * SR)
    buf[-n:] *= np.linspace(1, 0, n)
    wavfile.write(path, SR, (buf * 32767).astype(np.int16))


def finish(a, gen, T, wav):
    encode(a.out, wav, gen, T)


def tune_notes(name):
    return [m for (m, d) in TUNES[name]]


def hue_cols(n, sat=0.65, val=1.0, h0=0.0):
    out = []
    for i in range(n):
        h = (h0 + i / n) % 1.0
        c = cv2.cvtColor(np.uint8([[[int(h * 179), int(sat * 255), int(val * 255)]]]), cv2.COLOR_HSV2BGR)[0, 0]
        out.append((int(c[0]), int(c[1]), int(c[2])))
    return out


def glow_add(fr, layer, sigma, k=0.8):
    g = cv2.GaussianBlur(layer, (0, 0), sigma)
    fr[:] = cv2.add(fr, (g * k).astype(np.uint8))
    fr[:] = cv2.add(fr, layer)


# =====================================================================
# 1. polyfit: regular polygon shrinks on every bounce, one note per bounce, perfect fit at the end
# =====================================================================
def poly_geom(n, Rc, rot):
    apo = Rc * math.cos(math.pi / n)
    normals = [(math.cos(rot + 2 * math.pi * i / n), math.sin(rot + 2 * math.pi * i / n)) for i in range(n)]
    return apo, normals


def sim_poly(n, N, K, rb, a0, a_end, seed, rot):
    """Scale-invariant speed v = K*(a-rb): the bounce interval stays roughly constant while the polygon shrinks."""
    rng = random.Random(seed)
    ang = rng.uniform(0, 2 * math.pi)
    d = (math.cos(ang), math.sin(ang))
    normals = [(math.cos(rot + 2 * math.pi * i / n), math.sin(rot + 2 * math.pi * i / n)) for i in range(n)]
    x = y = 0.0
    a = a0
    bounces = []     # (t, a_after, x, y)
    t, dt = 0.0, 1.0 / 2400
    k = 0
    last_hit = -1.0
    path = []        # (t, x, y)
    while k < N and t < 80:
        v = K * max(a - rb, 6.0)
        x += d[0] * v * dt
        y += d[1] * v * dt
        t += dt
        hit = False
        for (nx, ny) in normals:
            pen = nx * x + ny * y + rb - a
            if pen > 0:
                dot = d[0] * nx + d[1] * ny
                if dot > 0:
                    d = (d[0] - 2 * dot * nx, d[1] - 2 * dot * ny)
                x -= nx * pen
                y -= ny * pen
                hit = True
        if hit and t - last_hit > 0.05:
            last_hit = t
            k += 1
            a = a0 * (a_end / a0) ** (k / N)
            for _ in range(8):
                moved = False
                for (nx, ny) in normals:
                    pen = nx * x + ny * y + rb - a
                    if pen > 0:
                        x -= nx * pen
                        y -= ny * pen
                        moved = True
                if not moved:
                    break
            else:
                x = y = 0.0
            bounces.append((t, a, x, y))
        if int(t / dt) % 80 == 0:
            path.append((t, x, y))
    return bounces, path


def render_polyfit(a, meta, wav, tcol):
    n = a.sides
    pal = PALS[a.pal]
    notes = [m for (m, d) in TUNES[a.tune]]
    N = 2 * len(notes)
    rb = 22.0
    Rc = 470.0
    rot = math.pi / 2 if n % 2 else math.pi / 2 + math.pi / n   # point up for odd n, flat top for even n
    a0 = Rc * math.cos(math.pi / n)
    a_end = rb + 3.0
    target = a.T - 3.6
    K = 4.0
    for _ in range(4):
        bounces, path = sim_poly(n, N, K, rb, a0, a_end, a.seed, rot)
        tN = bounces[-1][0] if len(bounces) == N else 80
        K *= tN / target
    bounces, path = sim_poly(n, N, K, rb, a0, a_end, a.seed, rot)
    assert len(bounces) == N, "polyfit: bounce count mismatch"
    tN = bounces[-1][0]
    bt = [b[0] for b in bounces]
    ev = [(t, "bell", notes[i % len(notes)], 0.38) for i, t in enumerate(bt)]
    ev.append((tN + 0.35, "bell", notes[0] - 12, 0.5))
    ev.append((tN + 0.35, "bell", notes[0], 0.4))
    mix(ev, a.T, wav)
    OVERLAY.update(hook_top=True)
    ptt = [p[0] for p in path]
    nfr = int(a.T * FPS)
    trail = np.zeros((H, W, 3), np.uint8)
    cy0 = CY + 40
    meta.update(sides=n, tune=a.tune, bounces=N, speed_K=round(K, 3), last_bounce_s=round(tN, 2), verified=True)

    def scale_at(t):
        j = bisect.bisect_right(bt, t) - 1
        a_prev = a0 if j < 0 else bounces[j][1]
        a_prev2 = a0 if j < 1 else bounces[j - 1][1]
        if j < 0:
            return a0
        k = min(1.0, (t - bt[j]) / 0.16)
        k = 1 - (1 - k) ** 3
        return a_prev2 + (a_prev - a_prev2) * k

    def ball_at(t):
        if t >= tN:
            u = min(1.0, (t - tN) / 0.5)
            u = u * u * (3 - 2 * u)
            return (bounces[-1][2] * (1 - u), bounces[-1][3] * (1 - u))
        j = bisect.bisect_right(ptt, t) - 1
        j = max(0, min(j, len(path) - 2))
        u = (t - path[j][0]) / max(1e-6, path[j + 1][0] - path[j][0])
        u = max(0.0, min(1.0, u))
        return (path[j][1] + (path[j + 1][1] - path[j][1]) * u, path[j][2] + (path[j + 1][2] - path[j][2]) * u)

    def gen():
        bg = make_bg(pal, "circle", 1)
        bg = make_bg(dict(pal, ring=pal["bg2"]), "circle", 1)
        hits_seen = 0
        prev = ball_at(0)
        lastname = "-"
        for i in range(nfr):
            t = i / FPS
            nh = bisect.bisect_right(bt, t)
            if nh > hits_seen:
                lastname = note_name(notes[(nh - 1) % len(notes)])
            hits_seen = nh
            sc = scale_at(t)
            col = grad(pal["stops"], (nh / max(1, N)) * 1.6)
            x, y = ball_at(t)
            trail[:] = (trail * 0.93).astype(np.uint8)
            cv2.line(trail, (int(CX + prev[0]), int(cy0 + prev[1])), (int(CX + x), int(cy0 + y)), col, 5, cv2.LINE_AA)
            prev = (x, y)
            fr = bg.copy()
            lay = np.zeros((H, W, 3), np.uint8)
            Rcur = sc / math.cos(math.pi / n)
            pts = np.array([[CX + Rcur * math.cos(rot + math.pi / n + 2 * math.pi * q / n),
                             cy0 + Rcur * math.sin(rot + math.pi / n + 2 * math.pi * q / n)] for q in range(n)], np.int32)
            # vertices sit between edge normals: normals at rot + 2*pi*i/n, vertices at +pi/n
            cv2.polylines(lay, [pts], True, hexbgr(pal["ring"]) if t < tN else (255, 255, 255), 9, cv2.LINE_AA)
            glow_add(fr, lay, 18, 0.9)
            glow_add(fr, trail, 6, 0.6)
            draw_ball(fr, CX + x, cy0 + y, rb, col)
            title_block(fr, a.title, tcol)
            blit(fr, text_layer("NOTES PLAYED", 40, tcol), CX, 1500)
            blit(fr, text_layer(f"{nh} / {N}", 96, tcol), CX, 1540)
            if t >= tN + 0.4:
                blit_fade(fr, text_layer("PERFECT FIT", 84, tcol), CX, 1730, min(1.0, (t - tN - 0.4) / 0.4))
            else:
                blit(fr, text_layer(lastname, 52, tcol), CX, 1740)
            yield fr
        for _ in range(FPS):
            yield fr
    finish(a, gen(), a.T, wav)


# =====================================================================
# 2. escape: balls in rotating rings with gaps; which one gets out first?
# =====================================================================
def wrap(d):
    return (d + math.pi) % (2 * math.pi) - math.pi


def sim_escape(n_balls, rings, rb, v, seed, T, sub=600):
    """rings: list of (radius, gap_rad, omega_rad_s), outermost last. Ball escapes ring k by crossing it inside the gap."""
    rng = random.Random(seed)
    balls = []
    r0 = rings[0][0] * 0.45
    for i in range(n_balls):
        ang = rng.uniform(0, 2 * math.pi)
        rad = rng.uniform(0, r0)
        d = rng.uniform(0, 2 * math.pi)
        balls.append(dict(x=rad * math.cos(ang), y=rad * math.sin(ang), vx=v * math.cos(d), vy=v * math.sin(d), lvl=0, out=None))
    hits, escapes, snaps = [], [], []
    dt = 1.0 / sub
    for step in range(int(T * sub)):
        t = step * dt
        for b in balls:
            if b["out"] is not None:
                b["x"] += b["vx"] * dt
                b["y"] += b["vy"] * dt
                b["vy"] += 1400 * dt
                continue
            b["x"] += b["vx"] * dt
            b["y"] += b["vy"] * dt
            k = b["lvl"]
            R, gap, om = rings[k]
            rho = math.hypot(b["x"], b["y"])
            if rho + rb >= R:
                ph = math.atan2(b["y"], b["x"])
                g = om * t
                inside_gap = abs(wrap(ph - g)) < gap / 2 - rb / R
                vr = (b["x"] * b["vx"] + b["y"] * b["vy"]) / max(rho, 1e-6)
                if vr > 0:
                    if inside_gap:
                        if rho - rb > R:
                            b["lvl"] += 1
                            if b["lvl"] == len(rings):
                                b["out"] = t
                                escapes.append((t, balls.index(b)))
                    else:
                        nx, ny = b["x"] / rho, b["y"] / rho
                        b["vx"] -= 2 * vr * nx
                        b["vy"] -= 2 * vr * ny
                        b["x"] = nx * (R - rb)
                        b["y"] = ny * (R - rb)
                        hits.append((t, balls.index(b), b["lvl"]))
            # inner rings: stay below ring k+? handled by lvl; also keep inside current ring only
        # ball-ball elastic
        live = [b for b in balls if b["out"] is None]
        for i in range(len(live)):
            for j in range(i + 1, len(live)):
                p, q = live[i], live[j]
                dx, dy = q["x"] - p["x"], q["y"] - p["y"]
                dd = math.hypot(dx, dy)
                if dd < 2 * rb and dd > 1e-6:
                    nx, ny = dx / dd, dy / dd
                    rv = (q["vx"] - p["vx"]) * nx + (q["vy"] - p["vy"]) * ny
                    if rv < 0:
                        p["vx"] += rv * nx; p["vy"] += rv * ny
                        q["vx"] -= rv * nx; q["vy"] -= rv * ny
                        ov = (2 * rb - dd) / 2
                        p["x"] -= nx * ov; p["y"] -= ny * ov
                        q["x"] += nx * ov; q["y"] += ny * ov
        if step % (sub // FPS) == 0:
            snaps.append([(b["x"], b["y"], b["lvl"], b["out"] is not None) for b in balls])
        if len(escapes) == n_balls and t > escapes[-1][0] + 3.0:
            break
    return hits, escapes, snaps


def render_escape(a, meta, wav, tcol):
    pal = PALS[a.pal]
    if a.variant == 1:
        rings = [(410.0, math.radians(30), math.radians(52))]
        n_balls, rb, v = 6, 30.0, 780.0
    else:
        rings = [(290.0, math.radians(46), math.radians(-70)), (420.0, math.radians(40), math.radians(48))]
        n_balls, rb, v = 5, 26.0, 820.0
    # pick a seed whose first escape is 3-9 s in and the last lands 4 s before the end
    best = None
    for s in range(a.seed, a.seed + 160):
        hits, esc, snaps = sim_escape(n_balls, rings, rb, v, s, a.T - 3.0)
        if len(esc) == n_balls:
            first, last = esc[0][0], esc[-1][0]
            score = abs(first - 5.0) + abs(last - (a.T - 6.0))
            if best is None or score < best[0]:
                best = (score, s, hits, esc, snaps)
            if score < 4:
                break
    assert best, "escape: no seed found"
    _, seed, hits, esc, snaps = best
    cols = hue_cols(n_balls, 0.62, 1.0, 0.02)
    pent = [60, 62, 64, 67, 69, 72, 74, 76, 79, 81]
    ev = [(t, "bell", pent[(bi * 2 + lv) % len(pent)] + 12, 0.3) for (t, bi, lv) in hits]
    for rank, (t, bi) in enumerate(esc):
        for q, m in enumerate([72, 76, 79, 84]):
            ev.append((t + q * 0.07, "bell", m + 6 * min(rank, 1), 0.55 if rank == 0 else 0.35))
    mix(ev, a.T, wav)
    OVERLAY.update(hook_top=True)
    meta.update(variant=a.variant, seed_used=seed, balls=n_balls, escapes=[(round(t, 2), b) for t, b in esc], verified=True)
    nfr = int(a.T * FPS)
    et = [e[0] for e in esc]

    def gen():
        bg = make_bg(dict(pal), "circle", 1)
        bg = make_bg(dict(pal, ring=pal["bg2"]), "circle", 1)
        last = None
        for i in range(nfr):
            t = i / FPS
            snap = snaps[min(i, len(snaps) - 1)]
            fr = bg.copy()
            lay = np.zeros((H, W, 3), np.uint8)
            for (R, gap, om) in rings:
                g = om * t
                st = math.degrees(g + gap / 2)
                en = math.degrees(g - gap / 2) + 360
                cv2.ellipse(lay, (CX, CY), (int(R), int(R)), 0, st, en, hexbgr(pal["ring"]), 10, cv2.LINE_AA)
            glow_add(fr, lay, 16, 0.8)
            for bi, (x, y, lvl, out) in enumerate(snap):
                px, py = CX + x, CY + y
                if -80 < px < W + 80 and -80 < py < H + 80:
                    draw_ball(fr, px, py, rb, cols[bi])
            title_block(fr, a.title, tcol)
            # leaderboard
            nesc = bisect.bisect_right(et, t)
            blit(fr, text_layer("ESCAPED", 40, tcol), CX, 1478)
            for rk in range(n_balls):
                x = CX - (n_balls - 1) * 62 + rk * 124
                if rk < nesc:
                    cv2.circle(fr, (x, 1610), 38, cols[esc[rk][1]], -1, cv2.LINE_AA)
                    blit(fr, text_layer(str(rk + 1), 44, "#05060b"), x, 1588)
                else:
                    cv2.circle(fr, (x, 1610), 38, (70, 70, 80), 3, cv2.LINE_AA)
            if nesc >= 1:
                blit_fade(fr, text_layer("FIRST OUT", 52, tcol), CX, 1700, min(1.0, (t - et[0]) / 0.3) * (1 if t < et[0] + 3 else 0))
            yield fr
        for _ in range(FPS):
            yield fr
    finish(a, gen(), a.T, wav)


# =====================================================================
# 3. squeeze: steel bars crush balls, each crushed ball becomes 4 new balls, the ring finally shatters
# =====================================================================
def render_squeeze(a, meta, wav, tcol):
    pal = PALS[a.pal]
    rng = np.random.RandomState(a.seed)
    R = 424.0
    S = 12 if a.variant == 1 else 11
    intro, D, shat = 2.0, 1.9, 5.0
    T = a.T
    BOX = 450.0 if a.variant == 1 else 470.0
    CAP = 5000
    pos = np.zeros((1, 2)); ang = rng.uniform(0, 2 * np.pi)
    pos[0] = (rng.uniform(-150, 150), rng.uniform(-150, 150))
    spd = 520.0
    vel = np.array([[spd * math.cos(ang), spd * math.sin(ang)]])
    gen_id = np.zeros(1, int)
    caught = np.zeros(1, bool)
    cols = [grad(pal["stops"], 0.9 * k / max(1, S)) for k in range(S + 3)]
    ev = []
    counts = []
    dt = 1.0 / 60
    nfr = int(T * FPS)
    sq_t0 = [intro + k * D for k in range(S)]
    t_shatter = intro + S * D
    # per squeeze plan: box centre, width schedule
    state = dict(k=-1, box_c=np.zeros(2), Wnow=0.0, popped=False, cracks=[])
    shards = []
    flashes = []     # (frame, x, y)
    ring_broken = False
    tr = np.zeros((H, W, 3), np.uint8)
    last_count = 1
    OVERLAY.update(hook_top=True)
    bg = make_bg(dict(pal, ring=pal["bg2"]), "circle", 1)

    def radius_for(k):
        return max(4.5, 17.0 - 1.45 * k)

    def frame_draw(t, fr_idx):
        nonlocal pos, vel, gen_id, caught, ring_broken, shards, flashes, last_count
        fr = bg.copy()
        k = int((t - intro) // D) if t >= intro else -1
        k = min(k, S - 1)
        rk = radius_for(max(k, 0))
        # --- physics, two substeps per frame
        for sub in range(2):
            tt = t + sub * dt
            kk = int((tt - intro) // D) if tt >= intro else -1
            if kk != state["k"] and kk >= 0 and kk < S and tt < t_shatter:
                state["k"] = kk
                state["popped"] = False
                state["armed"] = True
                caught[:] = False
                n = len(pos)
                if kk < 2:
                    # dodge: put the box away from the lone ball
                    side = -1.0 if pos[0, 0] > 0 else 1.0
                    state["box_c"] = np.array([side * 150.0, 0.0])
                    state["bw"] = 230.0
                elif n <= 13:
                    state["box_c"] = np.zeros(2)
                    state["bw"] = 340.0
                    j = int(np.argmin(np.hypot(pos[:, 0], pos[:, 1])))
                    dj = float(np.hypot(*pos[j])) + 1e-6
                    vel[j] = -pos[j] / dj * spd
                else:
                    state["box_c"] = np.zeros(2)
                    state["bw"] = BOX
                c = state["box_c"]
                lim = R - state["bw"] * 0.72
                nrm = np.hypot(*c)
                if nrm > lim:
                    state["box_c"] = c * lim / nrm
                ev.append((tt, "noise", 0.12, 0.5))
            if state["k"] >= 0 and tt < t_shatter and not ring_broken:
                u = (tt - intro - state["k"] * D) / D
                # 0..0.22 bars slide in, 0.22..0.78 squeeze, 0.78..1 release
                bw = state["bw"]
                state["u"] = u
                if u < 0.22:
                    Wn = bw
                elif u < 0.78:
                    q = (u - 0.22) / 0.56
                    Wn = bw * (1 - q * q * (3 - 2 * q)) * 0.97 + bw * 0.03 * (1 - q)
                else:
                    Wn = 0.0
                state["Wnow"] = Wn
                if u >= 0.22 and state.get("armed") and len(pos) > 0:
                    state["armed"] = False
                    c = state["box_c"]
                    inside = (np.abs(pos[:, 0] - c[0]) < bw / 2) & (np.abs(pos[:, 1] - c[1]) < bw / 2)
                    if state["k"] < 2:
                        inside[:] = False
                    caught[:] = inside
                    if os.environ.get("SQDEBUG"):
                        print("catch", state["k"], len(pos), int(inside.sum()), c, bw, pos[:3], file=sys.stderr)
                    state["cwid0"] = bw
                    ev.append((tt, "thunk", 0, 0.6))
                if u >= 0.22 and u < 0.78 and caught.any():
                    c = state["box_c"]
                    # caught balls stay inside the closing box
                    half = max(Wn / 2, 1.0)
                    idx = np.where(caught)[0]
                    pos[idx, 0] = np.clip(pos[idx, 0], c[0] - half + rk * 0.5, c[0] + half - rk * 0.5) if half > rk else c[0]
                    vel[idx] *= 0.9
                if u >= 0.74 and not state["popped"] and caught.any():
                    state["popped"] = True
                    idx = np.where(caught)[0]
                    allowed = max(0, (CAP - len(pos)) // 3)
                    if len(idx) > allowed:
                        caught[idx[allowed:]] = False
                        idx = idx[:allowed]
                    m = len(idx)
                    base = pos[idx]
                    newp, newv = [], []
                    for q in range(4):
                        ang = rng.uniform(0, 2 * np.pi, m)
                        sp = rng.uniform(380, 800, m)
                        newp.append(base + 0)
                        newv.append(np.stack([sp * np.cos(ang), sp * np.sin(ang)], 1))
                    keep = ~caught
                    pos = np.concatenate([pos[keep]] + newp)
                    vel = np.concatenate([vel[keep]] + newv)
                    gen_id = np.concatenate([gen_id[keep]] + [np.full(m, min(state["k"] + 1, S + 2), int) for _ in range(4)])
                    caught = np.zeros(len(pos), bool)
                    for (x, y) in base[: min(m, 30)]:
                        flashes.append((fr_idx, x, y))
                    ev.append((tt, "noise", 0.25, 0.9))
                    ev.append((tt, "bell", 55 + 2 * state["k"], 0.5))
                    ev.append((tt + 0.02, "bell", 67 + state["k"], 0.3))
            if len(pos) > 4200:
                pass
            if not ring_broken:
                pos = pos + vel * dt
                d = np.hypot(pos[:, 0], pos[:, 1])
                out = d > R - rk
                if out.any():
                    nx, ny = pos[out, 0] / d[out], pos[out, 1] / d[out]
                    vr = vel[out, 0] * nx + vel[out, 1] * ny
                    rv = np.where(vr > 0, vr, 0)
                    vel[out, 0] -= 2 * rv * nx
                    vel[out, 1] -= 2 * rv * ny
                    pos[out, 0] = nx * (R - rk)
                    pos[out, 1] = ny * (R - rk)
                    nh = int(out.sum())
                    if nh and rng.rand() < 0.9:
                        pent = [72, 74, 76, 79, 81, 84, 86, 88, 91]
                        ev.append((tt, "bell", pent[rng.randint(len(pent))], 0.07 + 0.2 * min(1.0, 8.0 / (1 + nh))))
                # solid bars/plates repel un-caught balls
                if state["k"] >= 0 and state["Wnow"] > 0 and state.get("u", 0) >= 0.22:
                    c = state["box_c"]
                    hw = state["Wnow"] / 2 + 56
                    hh = state["bw"] / 2 + 14
                    dx = pos[:, 0] - c[0]
                    dy = pos[:, 1] - c[1]
                    mask = (np.abs(dx) < hw) & (np.abs(dy) < hh) & (~caught)
                    # keep out of the bars (left/right slabs) only, balls may enter between them via top/bottom? no: whole block solid
                    if mask.any():
                        px = hw - np.abs(dx)
                        py = hh - np.abs(dy)
                        ux = mask & (px < py)
                        uy = mask & ~(px < py)
                        pos[ux, 0] = c[0] + np.sign(dx[ux]) * hw
                        vel[ux, 0] = np.sign(dx[ux]) * np.abs(vel[ux, 0])
                        pos[uy, 1] = c[1] + np.sign(dy[uy]) * hh
                        vel[uy, 1] = np.sign(dy[uy]) * np.abs(vel[uy, 1])
            else:
                vel[:, 1] += 900 * dt
                pos = pos + vel * dt
        n = len(pos)
        counts.append(n)
        # ---- draw balls on a glow layer
        lay = np.zeros((H, W, 3), np.uint8)
        px = (CX + pos[:, 0]).astype(int)
        py = (CY + pos[:, 1]).astype(int)
        rr = max(2, int(round(rk if not ring_broken else radius_for(S - 1))))
        if n > 300:
            lay_s = np.zeros((H // 2, W // 2, 3), np.uint8)
        for j in range(n):
            if 0 <= px[j] < W and 0 <= py[j] < H:
                col = cols[gen_id[j]]
                if caught[j] and state.get("Wnow", 1) > 0 and state.get("cwid0"):
                    sq = max(0.18, state["Wnow"] / state["cwid0"])
                    cv2.ellipse(fr, (px[j], py[j]), (max(2, int(rr * sq * 1.3)), int(rr * (1 + (1 - sq) * 0.9))), 0, 0, 360, col, -1, cv2.LINE_AA)
                else:
                    cv2.circle(lay, (px[j], py[j]), rr, col, -1, cv2.LINE_AA)
        glow_add(fr, lay, 5, 0.5)
        # ---- ring (heats up, cracks, shatters)
        heat = min(1.0, (max(0, k) / S) ** 1.5) if not ring_broken else 1.0
        ringc = tuple(int(c0 * (1 - heat) + c1 * heat) for c0, c1 in zip(hexbgr(pal["ring"]), (40, 70, 255)))
        if not ring_broken:
            rl = np.zeros((H, W, 3), np.uint8)
            jit = 0 if k < S - 3 else int(rng.randint(-2, 3))
            cv2.circle(rl, (CX, CY), int(R + 6 + jit), ringc, 10, cv2.LINE_AA)
            if k >= S - 3 and k >= 0:
                nc = (k - (S - 4)) * 5
                r2 = np.random.RandomState(7)
                for q in range(nc):
                    aq = r2.uniform(0, 2 * np.pi)
                    cv2.line(rl, (int(CX + (R - 8) * math.cos(aq)), int(CY + (R - 8) * math.sin(aq))),
                             (int(CX + (R + 24) * math.cos(aq + 0.05)), int(CY + (R + 24) * math.sin(aq + 0.05))), (255, 255, 255), 3, cv2.LINE_AA)
            glow_add(fr, rl, 20, 0.9)
        else:
            rl = np.zeros((H, W, 3), np.uint8)
            for sh in shards:
                sh["a"] += sh["w"] * dt * 2
                sh["x"] += sh["vx"] * dt * 2
                sh["y"] += sh["vy"] * dt * 2
                sh["vy"] += 900 * dt * 2
                a0_, a1_ = sh["a"] - 0.1, sh["a"] + 0.1
                cv2.ellipse(rl, (int(CX + sh["x"]), int(CY + sh["y"])), (int(R), int(R)), 0, math.degrees(a0_), math.degrees(a1_), ringc, 10, cv2.LINE_AA)
            glow_add(fr, rl, 14, 0.7)
        # ---- steel press
        if state["k"] >= 0 and t < t_shatter and state.get("Wnow", 0) > 0 and not ring_broken:
            c = state["box_c"]; bw = state["bw"]; Wn = state["Wnow"]
            u = (t - intro - state["k"] * D) / D
            slide = min(1.0, u / 0.22)
            ease = 1 - (1 - slide) ** 3
            off = (1 - ease) * 360
            cx0, cy0 = int(CX + c[0]), int(CY + c[1])
            for sgn in (-1, 1):
                x0 = int(cx0 + sgn * (Wn / 2 + 28 + off) - 28)
                y0, y1 = int(cy0 - bw / 2 - 14), int(cy0 + bw / 2 + 14)
                cv2.rectangle(fr, (x0, y0), (x0 + 56, y1), (150, 150, 160), -1)
                cv2.rectangle(fr, (x0 + 4, y0 + 4), (x0 + 18, y1 - 4), (215, 215, 225), -1)
                cv2.rectangle(fr, (x0, y0), (x0 + 56, y1), (60, 60, 70), 3)
            for sgn in (-1, 1):
                y0 = int(cy0 + sgn * (bw / 2 + 8) - 8)
                x0, x1 = int(cx0 - Wn / 2 - 56 - off), int(cx0 + Wn / 2 + 56 + off)
                cv2.rectangle(fr, (x0, y0), (x1, y0 + 16), (140, 140, 150), -1)
                cv2.rectangle(fr, (x0, y0), (x1, y0 + 16), (60, 60, 70), 2)
        # ---- pop flashes
        live = []
        for (f0, x, y) in flashes:
            kf = (fr_idx - f0) / 8.0
            if kf <= 1:
                live.append((f0, x, y))
                cv2.circle(fr, (int(CX + x), int(CY + y)), int(10 + 46 * kf), (255, 255, 255), 2, cv2.LINE_AA)
        flashes[:] = live
        title_block(fr, a.title, tcol)
        blit(fr, text_layer("SQUEEZES", 38, tcol), CX - 230, 1500)
        blit(fr, text_layer(str(max(0, min(S, k + (1 if t >= intro and ((t - intro) % D) / D > 0.78 else 0)))), 100, tcol), CX - 230, 1540)
        blit(fr, text_layer("BALLS", 38, tcol), CX + 230, 1500)
        blit(fr, text_layer(f"{n:,}", 100, tcol), CX + 230, 1540)
        return fr

    def gen():
        nonlocal ring_broken, shards
        for i in range(nfr):
            t = i / FPS
            if t >= t_shatter and not ring_broken:
                ring_broken = True
                nseg = 26
                for q in range(nseg):
                    aq = 2 * np.pi * q / nseg
                    sp = rng.uniform(250, 520)
                    shards.append(dict(a=aq, x=0.0, y=0.0, vx=math.cos(aq) * sp, vy=math.sin(aq) * sp - 120, w=rng.uniform(-3, 3)))
                ev.append((t, "noise", 0.9, 1.2))
                for q in range(6):
                    ev.append((t + 0.05 * q, "bell", 96 - 3 * q, 0.5))
                # burst: every ball flies outwards
                d = np.hypot(pos[:, 0], pos[:, 1]) + 1e-6
                sp = rng.uniform(350, 900, len(pos))
                vel[:] = np.stack([pos[:, 0] / d * sp, pos[:, 1] / d * sp - 200], 1)
            fr = frame_draw(t, i)
            if t >= t_shatter + 1.0:
                blit_fade(fr, text_layer(f"{len(pos):,} BALLS BURST OUT", 54, tcol), CX, 1730, min(1.0, (t - t_shatter - 1.0) / 0.5))
            yield fr
        for _ in range(FPS):
            yield fr
        mix(ev, T, wav)

    # audio is produced after frames (events collected during simulation); encode reads wav at the end, so pre-render frames first
    frames = []
    g = gen()
    import tempfile
    tmpdir = tempfile.mkdtemp()
    raw = os.path.join(tmpdir, "f.raw")
    with open(raw, "wb") as fh:
        for fr in g:
            fh.write(fr.tobytes())
    meta.update(squeezes=S, final_balls=int(len(pos)), counts_sample=[int(counts[i]) for i in range(0, len(counts), FPS * 2)], verified=True)

    def replay():
        with open(raw, "rb") as fh:
            sz = H * W * 3
            while True:
                b = fh.read(sz)
                if len(b) < sz:
                    break
                yield np.frombuffer(b, np.uint8).reshape(H, W, 3)
    encode(a.out, wav, replay(), T)
    os.remove(raw)


# =====================================================================
# 4. mirror: ellipse focus property
# =====================================================================
def ell_hit(px, py, dx, dy, A, B):
    """Forward intersection of ray with ellipse x^2/B^2 + y^2/A^2 = 1 (A vertical semi-axis)."""
    qa = dx * dx / (B * B) + dy * dy / (A * A)
    qb = 2 * (px * dx / (B * B) + py * dy / (A * A))
    qc = px * px / (B * B) + py * py / (A * A) - 1
    disc = qb * qb - 4 * qa * qc
    t = (-qb + math.sqrt(max(disc, 0))) / (2 * qa)
    return px + dx * t, py + dy * t, t


def ray_path(fy, th, A, B, legs=30):
    px, py = 0.0, fy
    dx, dy = math.cos(th), math.sin(th)
    pts = [(px, py)]
    for _ in range(legs):
        x, y, t = ell_hit(px, py, dx, dy, A, B)
        pts.append((x, y))
        nx, ny = x / (B * B), y / (A * A)
        nn = math.hypot(nx, ny)
        nx, ny = nx / nn, ny / nn
        dot = dx * nx + dy * ny
        dx, dy = dx - 2 * dot * nx, dy - 2 * dot * ny
        px, py = x, y
    seg = [math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)]
    cum = [0.0]
    for s in seg:
        cum.append(cum[-1] + s)
    return pts, cum


def pos_on(pts, cum, s):
    j = bisect.bisect_right(cum, s) - 1
    j = max(0, min(j, len(pts) - 2))
    u = min(1.0, (s - cum[j]) / max(1e-9, cum[j + 1] - cum[j]))
    return pts[j][0] + (pts[j + 1][0] - pts[j][0]) * u, pts[j][1] + (pts[j + 1][1] - pts[j][1]) * u, j


def render_mirror(a, meta, wav, tcol):
    pal = PALS[a.pal]
    if a.variant == 1:
        A, B, n0, maxr = 480.0, 330.0, 6, 192
    else:
        A, B, n0, maxr = 460.0, 390.0, 5, 160
    c = math.sqrt(A * A - B * B)
    tau = 1.45
    v = 2 * A / tau
    notes = [m for (m, d) in TUNES[a.tune]]
    cy0 = CY + 30
    rays = []   # dict(t0, fy, th, pts, cum, col)
    n = n0
    t = 0.0
    fy = c        # start at lower focus? screen y down; focus index alternate
    ang_sets = [2 * math.pi * (j + 0.5) / n0 + 0.0137 for j in range(n0)]
    all_ang = list(ang_sets)

    def add(t0, fy_, angs):
        for th in angs:
            pts, cum = ray_path(fy_, th, A, B)
            rays.append(dict(t0=t0, th=th, pts=pts, cum=cum, col=grad(pal["stops"], (th / (2 * math.pi)) % 1.0)))
    add(0.0, fy, all_ang)
    k = 0
    arrivals = []
    while n < maxr:
        k += 1
        tk = k * tau
        arrivals.append(tk)
        if k % 2 == 1:
            n2 = n * 2
            new = [2 * math.pi * (j + 0.5) / n2 + 0.0137 for j in range(n2) if j % 2 == 0]
            fy_now = -c if k % 2 == 1 else c
            add(tk, fy_now, new)
            n = n2
    # arrivals continue to the end
    kk = k + 1
    while kk * tau < a.T - 1:
        arrivals.append(kk * tau)
        kk += 1
    ev = []
    for i, ta in enumerate(arrivals):
        m = notes[i % len(notes)]
        ev.append((ta, "bell", m, 0.5))
        ev.append((ta, "bell", m - 12, 0.4))
    r_ev = random.Random(a.seed)
    pent = [72, 74, 76, 79, 81, 84, 86, 88, 91]
    for r in rays:
        for j in range(1, len(r["cum"])):
            th_ = r["t0"] + r["cum"][j] / v
            if th_ < a.T and j % 2 == 1 and r_ev.random() < 0.45:
                ev.append((th_, "bell", pent[r_ev.randrange(len(pent))], 0.05))
    mix(ev, a.T, wav, gain=0.8)
    OVERLAY.update(hook_top=True)
    nfr = int(a.T * FPS)
    canvas = np.zeros((H, W, 3), np.uint8)
    bg = make_bg(dict(pal, ring=pal["bg2"]), "circle", 1)
    meta.update(variant=a.variant, a=A, b=B, focal=round(c, 1), leg_s=tau, rays_final=n, verified=True)
    last_s = {}

    def gen():
        arr_seen = 0
        pulse = []
        for i in range(nfr):
            tt = i / FPS
            fr = bg.copy()
            nr = 0
            lay = np.zeros((H, W, 3), np.uint8)
            for ri, r in enumerate(rays):
                if tt < r["t0"]:
                    continue
                nr += 1
                s = (tt - r["t0"]) * v
                s0 = last_s.get(ri, 0.0)
                x, y, j = pos_on(r["pts"], r["cum"], s)
                # draw the new part of the line, split at vertices
                sp = s0
                while sp < s - 1e-6:
                    jj = bisect.bisect_right(r["cum"], sp) - 1
                    jj = max(0, min(jj, len(r["pts"]) - 2))
                    se = min(s, r["cum"][jj + 1])
                    if se <= sp:
                        se = s
                    xa, ya, _ = pos_on(r["pts"], r["cum"], sp)
                    xb, yb, _ = pos_on(r["pts"], r["cum"], se)
                    cv2.line(canvas, (int(CX + xa), int(cy0 + ya)), (int(CX + xb), int(cy0 + yb)), tuple(int(q * 0.5) for q in r["col"]), 2, cv2.LINE_AA)
                    sp = se
                last_s[ri] = s
                cv2.circle(lay, (int(CX + x), int(cy0 + y)), 7, (255, 255, 255), -1, cv2.LINE_AA)
            fr = cv2.add(fr, canvas)
            lay2 = np.zeros((H, W, 3), np.uint8)
            cv2.ellipse(lay2, (CX, cy0), (int(B), int(A)), 0, 0, 360, hexbgr(pal["ring"]), 9, cv2.LINE_AA)
            for fyy in (-c, c):
                cv2.circle(lay2, (CX, int(cy0 + fyy)), 10, (255, 255, 255), -1, cv2.LINE_AA)
            na = bisect.bisect_right(arrivals, tt)
            if na > arr_seen:
                pulse.append((i, 0.0 if arrivals[na - 1] / tau % 2 == 0 else 0.0))
                arr_seen = na
            for (f0, _) in list(pulse):
                kf = (i - f0) / 14.0
                if kf > 1:
                    pulse.remove((f0, _))
                    continue
                fyy = -c if (arr_seen % 2 == 1) else c
                for fy2 in (-c, c):
                    cv2.circle(lay2, (CX, int(cy0 + fy2)), int(14 + 70 * kf), tuple(int(255 * (1 - kf)) for _ in range(3)), 3, cv2.LINE_AA)
            glow_add(fr, lay2, 14, 0.8)
            glow_add(fr, lay, 5, 0.7)
            title_block(fr, a.title, tcol)
            blit(fr, text_layer("BALLS", 40, tcol), CX, 1500)
            blit(fr, text_layer(str(nr), 104, tcol), CX, 1540)
            yield fr
        for _ in range(FPS):
            yield fr
    finish(a, gen(), a.T, wav)


# =====================================================================
# 5. mine: four miners, four attitudes, one reaches the diamond
# =====================================================================
MINE_SCRIPTS = {
    1: dict(
        names=["HOPE", "HABIT", "FEAR", "PRIDE"],
        win=1,
        lanes=[
            [("dig", 1.0, 11.0, 0.32, 21), ("sit", 11.6, 99)],
            [("dig", 1.0, 14.0, 0.5, 18), ("break", 14.0, 15.4), ("fix", 15.4, 18.0), ("dig", 18.0, 30.4, 0.5, 18), ("win", 30.8, 99)],
            [("dig", 1.0, 9.0, 0.36, 17), ("scared", 9.0, 10.2), ("climb", 10.2, 14.0), ("idle", 14.0, 99, "HOME")],
            [("dig", 1.0, 13.4, 0.4, 18), ("stuck", 13.4, 16.4, 0.4), ("break", 16.4, 17.8), ("sit", 17.8, 99)],
        ],
        caps=[(0.3, 3.4, "FOUR MINERS. ONE DIAMOND."), (9.2, 12.5, "Fear heard a noise and went home."),
              (11.2, 14.4, "Hope sat down 250 m short."), (14.2, 17.6, "Habit's pick broke. He fixed it."),
              (17.2, 20.6, "Pride hit granite and quit."), (30.9, 34.0, "Only Habit reached the diamond.")]),
    2: dict(
        names=["TALENT", "EFFORT", "LUCK", "EXCUSES"],
        win=1,
        lanes=[
            [("dig", 1.0, 7.0, 0.25, 26), ("sit", 7.6, 99)],
            [("dig", 1.0, 15.0, 0.5, 18), ("break", 15.0, 16.4), ("fix", 16.4, 19.0), ("dig", 19.0, 30.4, 0.5, 18), ("win", 30.8, 99)],
            [("dig", 1.0, 7.0, 0.5, 14), ("find", 7.0, 9.4), ("sit", 9.6, 99)],
            [("dig", 1.0, 2.5, 0.75, 6), ("idle", 2.5, 99, "LATER")],
        ],
        caps=[(0.3, 3.4, "FOUR MINERS. ONE DIAMOND."), (7.2, 10.4, "Talent got bored two thirds down."),
              (9.6, 12.8, "Luck found glass. It was only glass."), (3.0, 6.0, "Excuses will start tomorrow."),
              (16.0, 19.4, "Effort's pick broke. He fixed it."), (30.9, 34.0, "Only Effort reached the diamond.")]),
}


def build_lane(segs):
    hits = []
    for sg in segs:
        if sg[0] in ("dig", "stuck"):
            t0, t1, P = sg[1], sg[2], sg[3]
            inc = sg[4] if sg[0] == "dig" else 0
            k = 1
            while t0 + P * k <= t1 + 1e-6:
                hits.append((t0 + P * k, inc))
                k += 1
    return sorted(hits)


def render_mine(a, meta, wav, tcol):
    pal = PALS[a.pal]
    sc = MINE_SCRIPTS[a.variant]
    names, lanes_s = sc["names"], sc["lanes"]
    LW = W // 4
    YS, YB = 560, 1560          # surface and shaft bottom
    DIAM = 900.0
    lane_cols = [(60, 150, 255), (230, 190, 40), (190, 90, 255), (80, 200, 120)] if a.variant == 1 else [(240, 150, 60), (90, 210, 120), (220, 100, 200), (80, 90, 240)]
    hits_l = [build_lane(s) for s in lanes_s]
    rng = random.Random(a.seed)
    # static background
    bgs = make_bg(dict(pal, ring=pal["bg2"]), "circle", 1)
    stat = bgs.copy()
    cv2.rectangle(stat, (0, YS), (W, YB + 40), (28, 44, 66), -1)
    for _ in range(900):
        x, y = rng.randrange(0, W), rng.randrange(YS, YB + 40)
        w_, h_ = rng.randrange(8, 34), rng.randrange(6, 20)
        sh = rng.randrange(-10, 14)
        cv2.rectangle(stat, (x, y), (x + w_, y + h_), (28 + sh, 44 + sh, 66 + sh), -1)
    cv2.rectangle(stat, (0, YS - 14), (W, YS), (60, 140, 70), -1)
    for i in range(1, 4):
        cv2.line(stat, (i * LW, YS - 14), (i * LW, YB + 40), (10, 12, 18), 6)
    ev = []
    nfr = int(a.T * FPS)
    dig_depths = []

    def seg_at(i, t):
        cur = None
        for sg in lanes_s[i]:
            if sg[1] <= t:
                cur = sg
        return cur

    def depth_at(i, t):
        d = 0.0
        for (th, inc) in hits_l[i]:
            if th <= t:
                d += inc
            elif th - 0.0 > t:
                break
        # smooth the last hit
        for (th, inc) in hits_l[i]:
            if th <= t < th + 0.12 and inc:
                d -= inc * (1 - (t - th) / 0.12) ** 2 * 0.9
        sg = seg_at(i, t)
        if any(q[0] == "climb" and q[2] <= t for q in lanes_s[i]):
            return 0.0
        if sg and sg[0] == "climb":
            d0 = 0.0
            for (th, inc) in hits_l[i]:
                if th <= sg[1]:
                    d0 += inc
            u = min(1.0, (t - sg[1]) / (sg[2] - sg[1]))
            d = d0 * (1 - u)
        return d

    # audio events
    for i, hl in enumerate(hits_l):
        for (th, inc) in hl:
            ev.append((th, "thunk", 0, 0.28 if inc else 0.35))
        for sg in lanes_s[i]:
            if sg[0] == "break":
                ev.append((sg[1], "noise", 0.3, 1.0)); ev.append((sg[1] + 0.05, "bell", 48, 0.5))
            if sg[0] == "fix":
                for q in range(5):
                    ev.append((sg[1] + 0.3 + q * 0.4, "bell", 84 + (q % 2) * 3, 0.25))
            if sg[0] == "sit":
                ev.append((sg[1], "bell", 52, 0.5)); ev.append((sg[1] + 0.25, "bell", 48, 0.4))
            if sg[0] == "scared":
                ev.append((sg[1], "noise", 0.5, 0.8)); ev.append((sg[1], "bell", 45, 0.5))
            if sg[0] == "find":
                for q, m in enumerate([76, 79, 83]):
                    ev.append((sg[1] + q * 0.12, "bell", m, 0.4))
                ev.append((sg[1] + 1.5, "bell", 50, 0.5))
            if sg[0] == "win":
                for q, m in enumerate([72, 76, 79, 84, 88, 91]):
                    ev.append((sg[1] + q * 0.11, "bell", m, 0.5))
    mix(ev, a.T, wav)
    OVERLAY.update(hook_top=False)
    meta.update(variant=a.variant, win=names[sc["win"]], verified=True)

    def miner(fr, cx, yf, col, phi, crouch=0.0, arms_up=False, sit=False, broken=None):
        s = 1.0
        hip = (cx, yf - 46 * (1 - crouch * 0.4))
        if sit:
            hip = (cx, yf - 22)
        neck = (cx + (10 if sit else 0), hip[1] - 38 * (1 - crouch * 0.3) + (8 if sit else 0))
        # legs
        if sit:
            cv2.line(fr, (int(hip[0]), int(hip[1])), (int(cx + 26), int(yf - 6)), (60, 60, 70), 10, cv2.LINE_AA)
            cv2.line(fr, (int(cx + 26), int(yf - 6)), (int(cx + 40), int(yf)), (60, 60, 70), 10, cv2.LINE_AA)
        else:
            cv2.line(fr, (int(hip[0]), int(hip[1])), (int(cx - 14), int(yf)), (60, 60, 70), 10, cv2.LINE_AA)
            cv2.line(fr, (int(hip[0]), int(hip[1])), (int(cx + 14), int(yf)), (60, 60, 70), 10, cv2.LINE_AA)
        cv2.line(fr, (int(hip[0]), int(hip[1])), (int(neck[0]), int(neck[1])), col, 20, cv2.LINE_AA)
        head = (int(neck[0]), int(neck[1] - 20 + (10 if sit else 0)))
        cv2.circle(fr, head, 17, (170, 200, 235), -1, cv2.LINE_AA)
        cv2.ellipse(fr, (head[0], head[1] - 3), (19, 17), 0, 180, 360, (40, 210, 250), -1, cv2.LINE_AA)
        cv2.circle(fr, (head[0] + 12, head[1] - 4), 5, (255, 255, 255), -1, cv2.LINE_AA)
        sh = (neck[0], neck[1] + 6)
        if arms_up:
            for sg in (-1, 1):
                cv2.line(fr, (int(sh[0]), int(sh[1])), (int(sh[0] + sg * 26), int(sh[1] - 40)), (170, 200, 235), 8, cv2.LINE_AA)
            return
        if sit:
            cv2.line(fr, (int(sh[0]), int(sh[1])), (int(sh[0] + 14), int(sh[1] + 30)), (170, 200, 235), 8, cv2.LINE_AA)
            return
        hand = (sh[0] + 10, sh[1] + 12)
        cv2.line(fr, (int(sh[0]), int(sh[1])), (int(hand[0]), int(hand[1])), (170, 200, 235), 8, cv2.LINE_AA)
        if phi is None:
            return
        L = 70
        tip = (hand[0] + L * math.sin(phi), hand[1] + L * math.cos(phi))
        cv2.line(fr, (int(hand[0]), int(hand[1])), (int(tip[0]), int(tip[1])), (50, 90, 150), 7, cv2.LINE_AA)
        if broken is None:
            px, py = -math.cos(phi), math.sin(phi)
            cv2.line(fr, (int(tip[0] - px * 24), int(tip[1] - py * 24)), (int(tip[0] + px * 24), int(tip[1] + py * 24)), (200, 200, 210), 8, cv2.LINE_AA)
        else:
            tb = broken
            hx = tip[0] + 40 * tb
            hy = tip[1] + 130 * tb * tb - 80 * tb
            cv2.line(fr, (int(hx - 20), int(hy)), (int(hx + 20), int(hy + 8 * math.sin(tb * 20))), (200, 200, 210), 8, cv2.LINE_AA)

    def swing_phi(i, t):
        hl = hits_l[i]
        ts = [h[0] for h in hl]
        j = bisect.bisect_right(ts, t)
        if j >= len(ts):
            return None
        nxt = ts[j]
        prv = ts[j - 1] if j > 0 else None
        seg = seg_at(i, t)
        if prv is None:
            P = (seg[3] if seg and len(seg) > 3 else 0.5)
            prv = nxt - P
        P = nxt - prv
        if t - prv > 1.2:
            return 0.5
        p = (t - prv) / max(P, 1e-3)
        if p < 0.78:
            u = p / 0.78
            u = u * u * (3 - 2 * u)
            return 0.5 - 2.7 * u
        q = (p - 0.78) / 0.22
        return -2.2 + 2.7 * q * q

    def gen():
        for i in range(nfr):
            t = i / FPS
            fr = stat.copy()
            for li in range(4):
                x0 = li * LW
                cx = x0 + LW // 2
                d = depth_at(li, t)
                sg = seg_at(li, t)
                st = sg[0] if sg else "idle"
                # shaft
                shake = int(3 * math.sin(t * 90)) if st in ("scared",) else 0
                cv2.rectangle(fr, (cx - 100 + shake, YS), (cx + 100 + shake, int(YS + d + 8)), (14, 18, 26), -1)
                # diamond
                dy = int(YS + DIAM + 40)
                dp = np.array([[cx, dy - 34], [cx + 30, dy - 8], [cx, dy + 30], [cx - 30, dy - 8]], np.int32)
                tw = 0.5 + 0.5 * math.sin(t * 6 + li)
                cv2.fillPoly(fr, [dp], (255, 235, 120))
                cv2.polylines(fr, [dp], True, (255, 255, 255), 3, cv2.LINE_AA)
                if tw > 0.8:
                    cv2.line(fr, (cx + 22, dy - 30), (cx + 22, dy - 10), (255, 255, 255), 3, cv2.LINE_AA)
                    cv2.line(fr, (cx + 12, dy - 20), (cx + 32, dy - 20), (255, 255, 255), 3, cv2.LINE_AA)
                yf = YS + d
                phi = None
                crouch = 0.0
                arms_up = False
                sit = False
                broken = None
                if st in ("dig", "stuck"):
                    phi = swing_phi(li, t)
                    if phi is None:
                        phi = 0.5
                elif st == "break":
                    phi = 0.5
                    broken = max(0.0, t - sg[1])
                elif st == "fix":
                    crouch = 1.0
                    phi = 1.2 + 0.25 * math.sin(t * 22)
                elif st == "sit":
                    sit = True
                elif st in ("scared",):
                    phi = 0.5 + 0.2 * math.sin(t * 40)
                elif st == "climb":
                    phi = None
                elif st == "idle":
                    phi = 0.8
                elif st in ("win", "find"):
                    arms_up = True
                elif st == "sit":
                    sit = True
                if sg is None:
                    phi = 0.8
                x_off = 0
                if st == "climb" and d < 40:
                    x_off = -int((1 - d / 40.0) * 0)
                miner(fr, cx + x_off, int(yf), lane_cols[li], phi, crouch, arms_up, sit, broken)
                # chips
                for (th, inc) in hits_l[li]:
                    if 0 <= t - th < 0.45 and inc:
                        for q in range(6):
                            ang = -0.4 - q * 0.45
                            vx = 150 * math.cos(ang) * (1 if q % 2 else -1)
                            vy = 260 * math.sin(ang)
                            tt = t - th
                            px = cx + 30 + vx * tt
                            py = (YS + depth_at(li, th + 0.12)) + vy * tt + 900 * tt * tt * 0.5
                            cv2.circle(fr, (int(px), int(py)), 4, (120, 150, 190), -1, cv2.LINE_AA)
                # labels
                blit(fr, text_layer(names[li], 40, "#%02x%02x%02x" % (lane_cols[li][2], lane_cols[li][1], lane_cols[li][0])), cx, 440)
                blit(fr, text_layer(f"{int(d)} m", 34, tcol), cx, 480)
                if st == "idle" and sg and len(sg) > 3 and sg[3]:
                    blit(fr, text_layer(sg[3], 30, "#ffffff"), cx, int(yf - 190))
                if st == "sit":
                    blit(fr, text_layer("...", 40, "#aaaaaa"), cx, int(yf - 130))
                if st == "find" and t - sg[1] > 1.4:
                    blit(fr, text_layer("just glass", 30, "#ffffff"), cx, int(yf - 190))
                if st == "win":
                    for q in range(10):
                        aq = q / 10 * 6.283 + t * 3
                        rr_ = 40 + 30 * ((t - sg[1]) % 1.0)
                        cv2.circle(fr, (int(cx + rr_ * math.cos(aq)), int(yf - 60 + rr_ * math.sin(aq))), 4, (255, 255, 255), -1, cv2.LINE_AA)
            title_block(fr, a.title, tcol)
            for (c0, c1, txt) in sc["caps"]:
                if c0 <= t < c1:
                    blit_fade(fr, text_layer(txt, 46, tcol), CX, 1700, fade(t, c0, c1, 0.35))
            yield fr
        for _ in range(FPS):
            yield fr
    finish(a, gen(), a.T, wav)


# =====================================================================
def render(a):
    pal = PALS[a.pal]
    tc = hexbgr(pal["text"])
    tcol = "#%02x%02x%02x" % (tc[2], tc[1], tc[0])
    meta = dict(kind=a.kind, pal=a.pal, title=a.title, T=a.T, seed=a.seed)
    wav = a.out.replace(".mp4", ".wav")
    fn = globals()["render_" + a.kind]
    fn(a, meta, wav, tcol)
    if a.meta:
        json.dump(meta, open(a.meta, "w"), indent=2)
    print(json.dumps(meta))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["polyfit", "escape", "squeeze", "mirror", "mine"])
    ap.add_argument("--pal", required=True, choices=list(PALS))
    ap.add_argument("--title", required=True)
    ap.add_argument("--T", type=float, default=32.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta")
    ap.add_argument("--hook")
    ap.add_argument("--cta")
    ap.add_argument("--series")
    ap.add_argument("--sides", type=int, default=3, help="polyfit: polygon sides")
    ap.add_argument("--tune", default="ode", choices=sorted(TUNES))
    ap.add_argument("--variant", type=int, default=1, help="escape/mirror/mine/squeeze: 1 or 2")
    a = ap.parse_args()
    OVERLAY.update(hook=a.hook, cta=a.cta, series=a.series, T=a.T, n=0)
    render(a)
