# Task 4: Computer Vision Object Detector & Frame-by-Frame Segmenter Pipeline (code as run in Google Colab)

# ---- Step 1
import cv2, numpy as np, pandas as pd
import matplotlib.pyplot as plt

W, H, FPS, N_FRAMES = 640, 480, 25, 250            # 10-second video
rng = np.random.default_rng(42)
KINDS = ["circle", "square", "triangle"]

# 7 moving shapes, each in its own horizontal lane so they never overlap;
# each one appears and disappears at its own time
shapes = []
for i in range(7):
    start = int(rng.integers(0, 120))
    shapes.append({"kind": KINDS[i % 3], "size": int(rng.integers(16, 26)),
                   "x": float(rng.uniform(60, W - 60)), "lane": 50 + 63 * i, "y": 50 + 63 * i,
                   "vx": float(rng.choice([-1, 1]) * rng.uniform(2, 6)),
                   "start": start, "end": start + int(rng.integers(90, 200))})

def draw_shape(img, s):
    x, y, r = int(s["x"]), int(s["y"]), s["size"]
    if s["kind"] == "circle":
        cv2.circle(img, (x, y), r, 235, -1)
    elif s["kind"] == "square":
        cv2.rectangle(img, (x - r, y - r), (x + r, y + r), 235, -1)
    else:
        pts = np.array([[x, y - r], [x - r, y + r], [x + r, y + r]], np.int32)
        cv2.fillPoly(img, [pts], 235)

# uneven lighting: bright on the left, dark on the right (this is why we need ADAPTIVE threshold)
light = np.tile(np.linspace(110, 20, W), (H, 1))

writer = cv2.VideoWriter("shapes.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
truth = []                                          # ground truth: real count of each shape per frame
for f in range(N_FRAMES):
    frame = np.zeros((H, W), np.uint8)
    visible = [s for s in shapes if s["start"] <= f < s["end"]]
    for s in visible:
        draw_shape(frame, s)
    frame = frame.astype(float) * (0.45 + light / 200)          # shapes get darker on the right
    frame = np.clip(np.maximum(frame, light) + rng.normal(0, 12, (H, W)), 0, 255).astype(np.uint8)  # + camera noise
    writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    truth.append({"frame": f, **{k: sum(s["kind"] == k for s in visible) for k in KINDS}})
    for s in shapes:                                # move sideways (with a small wobble) and bounce off the edges
        s["x"] += s["vx"]
        if not 45 < s["x"] < W - 45: s["vx"] *= -1
        s["y"] = s["lane"] + 6 * np.sin(f / 8 + s["lane"])
writer.release()

truth = pd.DataFrame(truth)
truth.to_csv("ground_truth.csv", index=False)
print("Video saved: shapes.mp4,", N_FRAMES, "frames")
print("Average shapes on screen:", truth[KINDS].sum(axis=1).mean().round(1))

cap = cv2.VideoCapture("shapes.mp4"); cap.set(cv2.CAP_PROP_POS_FRAMES, 120); ok, frame = cap.read(); cap.release()
plt.figure(figsize=(7, 5)); plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); plt.title("Frame 120 of the test video"); plt.axis("off"); plt.show()

# ---- Step 2
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                 # 1. colour -> grey
blur = cv2.GaussianBlur(gray, (7, 7), 0)                        # 2. Gaussian filter removes grainy noise

# 3a. ONE threshold for the whole picture (Otsu picks it automatically)
_, global_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 3b. ADAPTIVE threshold: each pixel is compared with its own 101x101 neighbourhood;
#     it becomes white only if it is at least 15 brighter than its surroundings
mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, -15)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))   # 4. remove tiny white specks

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, img, title in zip(axes.ravel(), [gray, blur, global_mask, mask],
                          ["1. Grayscale frame", "2. After Gaussian blur",
                           "3a. Global threshold (Otsu)", "3b. Adaptive threshold + cleanup"]):
    ax.imshow(img, cmap="gray"); ax.set_title(title); ax.axis("off")
plt.tight_layout(); plt.savefig("threshold_comparison.png", dpi=150); plt.show()

# ---- Step 3
COLORS = {"circle": (255, 80, 80), "square": (80, 200, 80), "triangle": (80, 160, 255)}   # BGR colours

def classify(contour):
    """Name a shape by counting the corners of its simplified outline."""
    perimeter = cv2.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    if len(corners) == 3:
        return "triangle"
    if len(corners) == 4:
        return "square"
    return "circle"                      # many corners = round

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, -15)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 400]           # ignore tiny blobs
    counts = {k: 0 for k in KINDS}
    out = frame.copy()
    for c in contours:
        kind = classify(c)
        counts[kind] += 1
        x, y, w, h = cv2.boundingRect(c)
        cv2.drawContours(out, [c], -1, COLORS[kind], 2)                    # outline the shape
        cv2.putText(out, kind, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[kind], 1)
    # dynamic count overlay in the top-left corner
    cv2.rectangle(out, (0, 0), (150, 78), (0, 0, 0), -1)
    for i, k in enumerate(KINDS):
        cv2.putText(out, f"{k}s: {counts[k]}", (8, 22 + 23 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[k], 2)
    return out, counts

annotated, counts = process_frame(frame)
print("Detected:", counts)
print("Truth:   ", truth.loc[120, KINDS].to_dict())
plt.figure(figsize=(8, 6)); plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)); plt.axis("off")
plt.title("Frame 120 with detections"); plt.savefig("annotated_frame.png", dpi=150, bbox_inches="tight"); plt.show()

# ---- Step 4
import time

cap = cv2.VideoCapture("shapes.mp4")
out_video = cv2.VideoWriter("shapes_annotated.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
rows = []
f = 0
while True:
    ok, frm = cap.read()
    if not ok:
        break
    t0 = time.perf_counter()
    annotated, counts = process_frame(frm)                     # the full pipeline on one frame
    ms = (time.perf_counter() - t0) * 1000
    out_video.write(annotated)

    # baseline for comparison: the same pipeline but with ONE global (Otsu) threshold
    g = cv2.GaussianBlur(cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    _, gm = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    gc, _ = cv2.findContours(gm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    global_total = len([c for c in gc if cv2.contourArea(c) > 400])

    t = truth.loc[f]
    rows.append({"frame": f, "ms": ms, **{f"pred_{k}": counts[k] for k in KINDS},
                 **{f"true_{k}": int(t[k]) for k in KINDS}, "global_total": global_total})
    f += 1
cap.release(); out_video.release()

m = pd.DataFrame(rows)
m["pred_total"] = m[[f"pred_{k}" for k in KINDS]].sum(axis=1)
m["true_total"] = m[[f"true_{k}" for k in KINDS]].sum(axis=1)
m.to_csv("frame_metrics.csv", index=False)                    # per-frame inference log

print(f"Frames processed: {len(m)}")
print(f"Average time per frame: {m.ms.mean():.2f} ms  ->  {1000 / m.ms.mean():.0f} FPS  (slowest frame {m.ms.max():.2f} ms)")
for k in KINDS:
    print(f"{k:9s} correct count in {100 * (m[f'pred_{k}'] == m[f'true_{k}']).mean():5.1f}% of frames | "
          f"mean abs. error {(m[f'pred_{k}'] - m[f'true_{k}']).abs().mean():.3f}")
print(f"TOTAL objects correct in {100 * (m.pred_total == m.true_total).mean():.1f}% of frames (adaptive)")
print(f"TOTAL objects correct in {100 * (m.global_total == m.true_total).mean():.1f}% of frames (global Otsu baseline)")
all_right = (m[[f"pred_{k}" for k in KINDS]].values == m[[f"true_{k}" for k in KINDS]].values).all(axis=1)
print(f"Every class exactly right in {100 * all_right.mean():.1f}% of frames")

# ---- Step 5
# Chart 1: processing time of every frame
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.plot(m.frame, m.ms, lw=1, color="#1f77b4")
ax.axhline(m.ms.mean(), color="red", ls="--", label=f"average {m.ms.mean():.2f} ms")
ax.axhline(1000 / FPS, color="green", ls=":", label=f"real-time limit at {FPS} FPS = {1000 / FPS:.0f} ms")
ax.set_xlabel("Frame"); ax.set_ylabel("ms per frame"); ax.set_title("Inference time per frame"); ax.legend()
plt.tight_layout(); plt.savefig("latency_per_frame.png", dpi=150); plt.show()

# Chart 2: objects counted vs. real number, over the whole video
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.step(m.frame, m.true_total, where="mid", lw=3, color="#cccccc", label="Ground truth")
ax.step(m.frame, m.pred_total, where="mid", lw=1.2, color="#1f77b4", label="Adaptive pipeline")
ax.step(m.frame, m.global_total, where="mid", lw=1, color="#d62728", alpha=0.7, label="Global threshold")
ax.set_xlabel("Frame"); ax.set_ylabel("Objects counted"); ax.set_title("Object count per frame"); ax.legend()
plt.tight_layout(); plt.savefig("count_over_time.png", dpi=150); plt.show()

# Chart 3: which frames had a wrong label, and what went wrong
wrong = m[~all_right]
print("Frames with a wrong label:", wrong.frame.tolist())
print(wrong[["frame"] + [f"pred_{k}" for k in KINDS] + [f"true_{k}" for k in KINDS]].head(10).to_string(index=False))

cap = cv2.VideoCapture("shapes.mp4"); cap.set(cv2.CAP_PROP_POS_FRAMES, int(wrong.frame.iloc[0])); ok, bad = cap.read(); cap.release()
plt.figure(figsize=(8, 6)); plt.imshow(cv2.cvtColor(process_frame(bad)[0], cv2.COLOR_BGR2RGB)); plt.axis("off")
plt.title(f"Frame {int(wrong.frame.iloc[0])}: an example mistake"); plt.savefig("error_example.png", dpi=150, bbox_inches="tight"); plt.show()
