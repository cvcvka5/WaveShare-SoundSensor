#!/usr/bin/env python3
"""simlab: original long-form physics-sim Reels (1080x1920, 30fps, H.264 + AAC).
Kinds: fill (growing ball leaves rings, fills the arena), paint (block paints a square),
shrink (ball shrinks to nothing, live size counter).
Usage: simlab.py --kind fill --pal bb00 --title "..." --T 50 --seed 7 --out out/x.mp4 [--meta out/x.json]
"""
import argparse, json, math, os, random, subprocess, sys
from functools import lru_cache
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from scipy.io import wavfile

W, H, FPS, SR = 1080, 1920, 30, 44100
FONT = "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf"
CX, CY = 540, 960
PALS = {
    "bb00": dict(bg="#05060b", bg2="#0f1430", ring="#22d3ee",
                 stops=["#22d3ee", "#6366f1", "#8b5cf6", "#e879f9", "#22d3ee"], text="#ffffff"),
    "lab": dict(bg="#0a0705", bg2="#241609", ring="#ffb020",
                stops=["#ffb020", "#ff6b2c", "#ffd23f", "#fff3c4", "#ffb020"], text="#fff6e0"),
    "tt": dict(bg="#04100a", bg2="#0c2a14", ring="#a3e635",
               stops=["#a3e635", "#34d399", "#4ade80", "#d9f99d", "#a3e635"], text="#f7fee7"),
    # soft, low-contrast palettes for TikTok/YouTube: one hue family per gradient, no harsh jumps
    "tt_peach": dict(bg="#140a0e", bg2="#2b1621", ring="#fda4af",
                     stops=["#fda4af", "#fb9bb0", "#fdba9a", "#fde0b0", "#fda4af"], text="#fff1f2"),
    "tt_ocean": dict(bg="#041014", bg2="#0a2a30", ring="#5eead4",
                     stops=["#5eead4", "#67e8f9", "#7dd3fc", "#a7f3d0", "#5eead4"], text="#f0fdfa"),
    "tt_lav": dict(bg="#0d0a16", bg2="#221a3a", ring="#c4b5fd",
                   stops=["#c4b5fd", "#d8b4fe", "#f0abfc", "#fbcfe8", "#c4b5fd"], text="#faf5ff"),
    "tt_sand": dict(bg="#100e0a", bg2="#292317", ring="#fcd9a8",
                    stops=["#fcd9a8", "#f7c9a0", "#f3d9b1", "#fff1d6", "#fcd9a8"], text="#fffaf0"),
}
SCALE = [60, 62, 64, 67, 69, 72, 74, 76, 79, 81, 84, 86, 88, 91]


def hexbgr(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def grad(stops, u):
    u = (u % 1.0) * (len(stops) - 1)
    i = int(u)
    f = u - i
    a, b = hexbgr(stops[i]), hexbgr(stops[min(i + 1, len(stops) - 1)])
    return tuple(int(a[k] + (b[k] - a[k]) * f) for k in range(3))


# ---------------- audio ----------------
@lru_cache(maxsize=64)
def bell(midi, dur=1.1):
    f = 440 * 2 ** ((midi - 69) / 12)
    t = np.arange(int(dur * SR)) / SR
    y = (np.sin(2 * np.pi * f * t) * np.exp(-t * 4.2)
         + 0.35 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t * 7)
         + 0.15 * np.sin(2 * np.pi * f * 5.4 * t) * np.exp(-t * 11))
    n = int(0.003 * SR)
    y[:n] *= np.linspace(0, 1, n)
    return y


def make_audio(events, T, path):
    buf = np.zeros(int((T + 1.5) * SR))
    for t, midi, vel in events:
        s = bell(midi)
        i0 = int(t * SR)
        if i0 + len(s) > len(buf):
            s = s[: len(buf) - i0]
        buf[i0:i0 + len(s)] += s * vel
    buf = np.tanh(buf * 0.9) * 0.85
    buf = buf[: int((T + 1.0) * SR)]
    n = int(0.05 * SR)
    buf[-n:] *= np.linspace(1, 0, n)
    wavfile.write(path, SR, (buf * 32767).astype(np.int16))


# ---------------- text ----------------
@lru_cache(maxsize=512)
def text_layer(txt, size, color):
    font = ImageFont.truetype(FONT, size)
    l, t, r, b = font.getbbox(txt)
    img = Image.new("RGBA", (r - l + 24, b - t + 24), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((12 - l, 12 - t), txt, font=font, fill=color)
    return np.array(img)


def blit(frame, layer, cx, top):
    h, w = layer.shape[:2]
    x0 = int(cx - w / 2)
    y0 = int(top)
    a = layer[:, :, 3:4].astype(np.float32) / 255.0
    rgb = layer[:, :, 2::-1].astype(np.float32)
    roi = frame[y0:y0 + h, x0:x0 + w]
    roi[:] = (roi * (1 - a) + rgb * a).astype(np.uint8)


OVERLAY = dict(hook=None, cta=None, series=None, T=50.0, n=0)


def blit_fade(frame, layer, cx, top, k):
    if k <= 0:
        return
    lay = layer.copy()
    lay[:, :, 3] = (lay[:, :, 3].astype(np.float32) * min(1.0, k)).astype(np.uint8)
    blit(frame, lay, cx, top)


def fade(t, t0, t1, ramp=0.4):
    """0..1 envelope: fade in at t0, fade out ending at t1."""
    return max(0.0, min(1.0, (t - t0) / ramp, (t1 - t) / ramp))


def overlays(frame, color):
    """Optional series label, hook (first 3.5 s) and follow line (last 3 s). Off unless set."""
    o = OVERLAY
    t = o["n"] / FPS
    o["n"] += 1
    if o["series"]:
        blit_fade(frame, text_layer(o["series"], 34, color), CX, 130, 0.7)
    if o["hook"]:
        y = 1625
        for ln in o["hook"].split("|"):
            blit_fade(frame, text_layer(ln, 44, color), CX, y, fade(t, 0.3, 3.6))
            y += 62
    if o["cta"]:
        blit_fade(frame, text_layer(o["cta"], 46, color), CX, 1640, fade(t, o["T"] - 3.2, o["T"] + 5.0))


def title_block(frame, title, color):
    lines = title.split("|")
    y = 215
    for ln in lines:
        blit(frame, text_layer(ln, 64, color), CX, y)
        y += 88
    overlays(frame, color)


# ---------------- background ----------------
def make_bg(pal, shape="circle", R=430):
    bg = np.zeros((H, W, 3), np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.hypot(xx - CX, yy - CY) / 1100.0
    c1, c2 = np.array(hexbgr(pal["bg2"]), np.float32), np.array(hexbgr(pal["bg"]), np.float32)
    k = np.clip(d, 0, 1)[:, :, None]
    bg[:] = (c1 * (1 - k) + c2 * k).astype(np.uint8)
    ring = np.zeros_like(bg)
    col = hexbgr(pal["ring"])
    if shape == "circle":
        cv2.circle(ring, (CX, CY), R + 6, col, 10, cv2.LINE_AA)
    else:
        cv2.rectangle(ring, (CX - R - 8, CY - R - 8), (CX + R + 8, CY + R + 8), col, 8, cv2.LINE_AA)
    glow = cv2.GaussianBlur(ring, (0, 0), 22)
    out = cv2.add(bg, (glow * 0.9).astype(np.uint8))
    out = cv2.add(out, ring)
    return out


def draw_ball(frame, x, y, r, col):
    x, y = int(x), int(y)
    pad = int(r * 0.7) + 34
    x0, x1 = max(0, x - r - pad), min(W, x + r + pad)
    y0, y1 = max(0, y - r - pad), min(H, y + r + pad)
    x0, y0 = int(x0), int(y0)
    x1, y1 = int(x1), int(y1)
    roi = frame[y0:y1, x0:x1]
    f = 4 if r > 60 else 1
    small = np.zeros((max(1, (y1 - y0) // f), max(1, (x1 - x0) // f), 3), np.uint8)
    cv2.circle(small, ((x - x0) // f, (y - y0) // f), int((r + pad * 0.45) / f), col, -1, cv2.LINE_AA)
    small = cv2.GaussianBlur(small, (0, 0), max(1.0, pad * 0.33 / f))
    glow = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) if f > 1 else small
    roi[:] = cv2.add(roi, (glow * 0.55).astype(np.uint8))
    rr = max(1, int(round(r)))
    cv2.circle(roi, (x - x0, y - y0), rr, col, -1, cv2.LINE_AA)
    if rr > 6:
        cv2.circle(roi, (x - x0 - rr // 3, y - y0 - rr // 3), max(1, rr // 4), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(roi, (x - x0, y - y0), rr, (255, 255, 255), 2, cv2.LINE_AA)


# ---------------- simulations ----------------
def sim_circle(T, seed, r_of, grow, R=424, g=900.0, v0=620.0, sub=600):
    """Ball in circular arena. r_of(t, nb) -> radius. Returns per-frame (x,y,r,nb) and events."""
    rng = random.Random(seed)
    ang = rng.uniform(0, 2 * math.pi)
    x, y = rng.uniform(-60, 60), rng.uniform(-60, 60)
    vx, vy = v0 * math.cos(ang), v0 * math.sin(ang)
    dt = 1.0 / sub
    frames, events = [], []
    nb = 0
    nsteps = int(T * sub)
    per = sub // FPS
    for i in range(nsteps):
        t = i * dt
        vy += g * dt
        x += vx * dt
        y += vy * dt
        r = r_of(t, nb)
        d = math.hypot(x, y)
        if d + r >= R:
            nx, ny = x / d, y / d
            dot = vx * nx + vy * ny
            if dot > 0:
                vx -= 2 * dot * nx
                vy -= 2 * dot * ny
                nb += 1
                events.append((t, nb, abs(dot)))
                r = r_of(t, nb)
            lim = R - r - 0.5
            if lim < 0:
                lim = 0
            x, y = nx * lim, ny * lim
        if i % per == 0:
            frames.append((x, y, r, nb))
    return frames, events


def solve_fill(T, seed, target=0.955, R=424):
    lo, hi = 0.02, 14.0
    best = None
    for _ in range(16):
        gr = (lo + hi) / 2
        frames, events = sim_circle(T, seed, lambda t, nb: min(R - 1.0, 6 + gr * nb), gr, R)
        rf = frames[-1][2]
        best = (gr, frames, events)
        if rf < target * R:
            lo = gr
        else:
            hi = gr
    return best


def sim_paint(T, seed, size=64, A=860, target=0.985, sub=600):
    rng = random.Random(seed)
    ang = math.radians(rng.choice([31, 37, 43, 52, 58]) + rng.random())
    sx, sy = math.cos(ang), math.sin(ang)

    def run(speed, keep):
        x, y = rng.uniform(0, A - size) if keep == "x" else 0, 0
        x, y = 0.0, 0.0
        vx, vy = speed * sx, speed * sy
        dt = 1.0 / sub
        mask = np.zeros((A // 4 + 1, A // 4 + 1), bool)
        path, events = [], []
        per = sub // FPS
        for i in range(int(T * sub)):
            x += vx * dt
            y += vy * dt
            hit = False
            if x < 0:
                x, vx, hit = -x, -vx, True
            elif x > A - size:
                x, vx, hit = 2 * (A - size) - x, -vx, True
            if y < 0:
                y, vy, hit = -y, -vy, True
            elif y > A - size:
                y, vy, hit = 2 * (A - size) - y, -vy, True
            mask[int(y) // 4:int(y + size) // 4 + 1, int(x) // 4:int(x + size) // 4 + 1] = True
            if hit:
                events.append((i * dt, len(events), 1.0))
            if i % (sub // 60) == 0:
                path.append((x, y, mask.mean()))
        return path, events

    lo, hi = 200.0, 6000.0
    best = None
    for _ in range(14):
        sp = (lo + hi) / 2
        path, events = run(sp, "")
        cov = path[-1][2]
        best = (sp, path, events, cov)
        if cov < target:
            lo = sp
        else:
            hi = sp
    return best


# ---------------- renderers ----------------
def encode(out, audio, frame_iter, T):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-i", audio, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frame_iter:
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        sys.exit("ffmpeg failed")


def pick_notes(events, key, seed):
    rng = random.Random(seed)
    out, idx, direction = [], 0, 1
    for (t, n, v) in events:
        if key == "walk":
            idx += direction * rng.choice([1, 1, 2])
            if idx >= len(SCALE) - 1:
                idx, direction = len(SCALE) - 1, -1
            elif idx <= 0:
                idx, direction = 0, 1
            out.append((t, SCALE[idx], 0.32 + 0.1 * min(1, v / 900)))
    return out


def render(a):
    pal = PALS[a.pal]
    T = a.T
    meta = dict(kind=a.kind, pal=a.pal, title=a.title, T=T, seed=a.seed)
    wav = a.out.replace(".mp4", ".wav")
    tcol = hexbgr(pal["text"])
    tcol_rgb = "#%02x%02x%02x" % (tcol[2], tcol[1], tcol[0])

    if a.kind == "fill":
        R = 424
        bg = make_bg(pal, "circle", R)
        gr, frames, events = solve_fill(T, a.seed, R=R)
        meta.update(growth_px_per_bounce=round(gr, 3), bounces=len(events), final_fill=round(100 * (frames[-1][2] / R) ** 2, 1))
        make_audio(pick_notes(events, "walk", a.seed), T, wav)
        trail = np.zeros((H, W, 3), np.uint8)

        def gen():
            for i, (x, y, r, nb) in enumerate(frames):
                u = i / len(frames)
                col = grad(pal["stops"], u * 1.6)
                cv2.circle(trail, (CX + int(x), CY + int(y)), int(r), tuple(int(c * 0.5) for c in col), 3, cv2.LINE_AA)
                fr = cv2.add(bg, trail)
                draw_ball(fr, CX + x, CY + y, r, col)
                title_block(fr, a.title, tcol_rgb)
                pct = int(round(100 * (r / R) ** 2))
                blit(fr, text_layer(f"Filled: {pct}%", 56, tcol_rgb), CX, 1500)
                yield fr
            for _ in range(int(1.0 * FPS)):
                yield fr
        encode(a.out, wav, gen(), T)

    elif a.kind == "shrink":
        R = 424
        r0 = 190.0
        bg = make_bg(pal, "circle", R)
        rof = lambda t, nb: max(0.8, r0 * max(0.0, 1 - t / (T - 3.0)) ** 1.55 + 0.8) if t < T - 3.0 else 0.8
        frames, events = sim_circle(T, a.seed, rof, 0, R=R, g=700.0, v0=520.0)
        meta.update(bounces=len(events))
        notes = []
        rng = random.Random(a.seed)
        for (t, n, v) in events:
            r = rof(t, n)
            idx = min(len(SCALE) - 1, int((1 - r / r0) * len(SCALE) * 0.98))
            notes.append((t, SCALE[idx], 0.25 + 0.2 * (r / r0) + 0.08))
        make_audio(notes, T, wav)

        def gen():
            n = len(frames)
            for i, (x, y, r, nb) in enumerate(frames):
                u = i / n
                ring_col = grad(pal["stops"][:4], u * 0.9)
                fr = bg.copy()
                ov = np.zeros_like(fr)
                cv2.circle(ov, (CX, CY), R + 6, ring_col, 10, cv2.LINE_AA)
                fr = cv2.add(fr, (ov * 0.6).astype(np.uint8))
                if r > 1.0 or True:
                    draw_ball(fr, CX + x, CY + y, r, grad(pal["stops"], 0.0))
                title_block(fr, a.title, tcol_rgb)
                size = max(1, int(round(r * 2)))
                lab = f"Size: {size} px" if i < n - 1 else "Size: 1 px"
                blit(fr, text_layer(lab, 56, tcol_rgb), CX, 1500)
                yield fr
            for _ in range(int(1.0 * FPS)):
                yield fr
        encode(a.out, wav, gen(), T)

    elif a.kind == "paint":
        A = 860
        size = 64
        sp, path, events, cov = sim_paint(T, a.seed, size=size, A=A)
        meta.update(speed=round(sp), final_coverage=round(100 * cov, 1), bounces=len(events))
        make_audio(pick_notes(events, "walk", a.seed), T, wav)
        bg = make_bg(pal, "square", A // 2 + 4)
        canvas = np.zeros((A, A, 3), np.uint8)
        x0, y0 = CX - A // 2, CY - A // 2
        painted = np.zeros((A, A), bool)
        hist = {}

        def gen():
            n = int(T * FPS)
            for i in range(n):
                k0 = i * 2
                for k in range(max(0, k0 - 2), min(len(path), k0 + 2)):
                    x, y, c = path[k]
                    col = grad(pal["stops"], (k / len(path)) * 2.2)
                    xi, yi = int(x), int(y)
                    cv2.rectangle(canvas, (xi, yi), (xi + size, yi + size), col, -1)
                    cv2.rectangle(canvas, (xi, yi), (xi + size, yi + size), tuple(int(v * 0.55) for v in col), 2)
                x, y, c = path[min(len(path) - 1, k0)]
                fr = bg.copy()
                fr[y0:y0 + A, x0:x0 + A] = np.where(canvas.any(axis=2, keepdims=True), canvas, fr[y0:y0 + A, x0:x0 + A])
                xi, yi = int(x), int(y)
                cv2.rectangle(fr, (x0 + xi, y0 + yi), (x0 + xi + size, y0 + yi + size), (255, 255, 255), 3)
                title_block(fr, a.title, tcol_rgb)
                blit(fr, text_layer(f"Covered: {c * 100:.1f}%", 56, tcol_rgb), CX, 1500)
                yield fr
            for _ in range(int(1.0 * FPS)):
                yield fr
        encode(a.out, wav, gen(), T)

    if a.meta:
        json.dump(meta, open(a.meta, "w"), indent=2)
    print(json.dumps(meta))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["fill", "shrink", "paint"])
    ap.add_argument("--pal", required=True, choices=list(PALS))
    ap.add_argument("--title", required=True)
    ap.add_argument("--T", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta")
    ap.add_argument("--hook", help="on-screen line(s) for the first 3.5 s, lines split with |")
    ap.add_argument("--cta", help="on-screen follow line for the last 3 s")
    ap.add_argument("--series", help="small series label at the top, e.g. 'BOUNCE TEST 02'")
    a = ap.parse_args()
    OVERLAY.update(hook=a.hook, cta=a.cta, series=a.series, T=a.T, n=0)
    render(a)
