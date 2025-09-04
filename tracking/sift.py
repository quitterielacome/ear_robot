#!/usr/bin/env python3
"""
SIFT anchors + local NCC (feature parity with klt_full.py)
- Two SIFT-chosen anchor patches tracked via NCC within a padded window
- Affine (similarity) from initial anchor centers propagates a GT-initialised ROI
- Glare ablation: --glare_mode off|mask|inpaint (+HSV thresholds); optional --clahe
- HUD with IoU & CLE color-coded; LOST overlay when anchors fail
- Metrics: OP/DP, Success/Precision curves (+AUCs), DP@0.10·min(W,H)
- Poster-quality plots for success/precision in addition to standard plots
- CSV & report match klt_full.py format (so aggregation tooling can be shared)
"""

import cv2, argparse, time, csv, json, pathlib, datetime, os, re
from collections import deque
import numpy as np
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

# -------- Matplotlib headless if needed --------
import matplotlib
if "DISPLAY" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- helpers ----------
# --- dotted trajectory helper ---
def interpolate(p1, p2, n):
    """Return n evenly-spaced integer points between p1 and p2 (exclusive)."""
    pts = []
    for i in range(1, n+1):
        a = i / (n + 1.0)
        x = int((1-a)*p1[0] + a*p2[0])
        y = int((1-a)*p1[1] + a*p2[1])
        pts.append((x, y))
    return pts

def poster_axes(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def annotate_point(ax, x, y, text, va="bottom", ha="left"):
    ax.scatter([x], [y], s=28, zorder=3)
    ax.annotate(text, (x, y), xytext=(6, 6), textcoords="offset points",
                ha=ha, va=va, fontsize=9)

def glare_mask_hsv(bgr, v_hi=240, s_lo=40, v_hi2=220):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv)
    m1 = (V >= v_hi)
    m2 = (S <= s_lo) & (V >= v_hi2)
    mask = (m1 | m2).astype(np.uint8)*255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, 1)
    mask = cv2.dilate(mask, k, 1)
    return mask

def inpaint_if(mask, gray):
    if mask is None or mask.max()==0: return gray
    return cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)

def preprocess_gray(bgr, glare_mode="off", v_hi=240, s_lo=40, v_hi2=220, clahe=False):
    if glare_mode == "off":
        gmask = np.zeros(bgr.shape[:2], np.uint8)
    else:
        gmask = glare_mask_hsv(bgr, v_hi=v_hi, s_lo=s_lo, v_hi2=v_hi2)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if clahe:
        gray = cv2.createCLAHE(2.0,(8,8)).apply(gray)
    gi = inpaint_if(gmask, gray) if glare_mode == "inpaint" else gray
    return gi, gmask

def compute_center(b):
    x,y,w,h=b
    return np.array([x + w/2.0, y + h/2.0], dtype=float)

def iou(boxA, boxB):
    if (boxA is None) or (boxB is None): return None
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    if inter <= 0: return 0.0
    areaA = max(0, boxA[2]) * max(0, boxA[3])
    areaB = max(0, boxB[2]) * max(0, boxB[3])
    denom = areaA + areaB - inter
    return float(inter/denom) if denom>0 else 0.0

def center_distance(boxA, boxB):
    if (boxA is None) or (boxB is None): return None
    cA = (boxA[0]+boxA[2]/2.0, boxA[1]+boxA[3]/2.0)
    cB = (boxB[0]+boxB[2]/2.0, boxB[1]+boxB[3]/2.0)
    return float(np.hypot(cA[0]-cB[0], cA[1]-cB[1]))

def load_gt_json(path):
    with open(path,"r") as f:
        data=json.load(f)
    id_to_fname={im["id"]:im["file_name"] for im in data["images"]}
    gt={}
    for ann in data["annotations"]:
        fname=id_to_fname.get(ann["image_id"],"")
        m=re.search(r"frame_(\d+)",fname)
        if not m: continue
        frame_idx=int(m.group(1))
        gt[frame_idx]=tuple(ann["bbox"])
    return dict(sorted(gt.items()))

def running_ratio(bools):
    out=[]; good=0
    for i,flag in enumerate(bools, start=1):
        if flag: good += 1
        out.append(good / i)
    return out

def search_patch(gray, tpl, box, pad, method=cv2.TM_CCOEFF_NORMED):
    x,y,w,h=box
    sx,sy=max(x-pad,0),max(y-pad,0)
    ex,ey=min(x+w+pad,gray.shape[1]),min(y+h+pad,gray.shape[0])
    R=gray[sy:ey,sx:ex]
    r=cv2.matchTemplate(R,tpl,method)
    _,v,_,loc=cv2.minMaxLoc(r)
    return v,(loc[0]+sx,loc[1]+sy)

def crop_clamped(img, box):
    x,y,w,h = map(int, box)
    H, W = img.shape[:2]
    x = max(0, min(x, W-2)); y = max(0, min(y, H-2))
    w = max(2, min(w, W-x)); h = max(2, min(h, H-y))
    return img[y:y+h, x:x+w].copy()

def sift_in_box(gray, box, sift, mask=None, max_kp=200):
    """Detect SIFT inside a box; return (kp_xy Nx2 float32, des Nx128)."""
    x,y,w,h = map(int, box)
    sub = gray[y:y+h, x:x+w]
    sub_mask = None
    if mask is not None:
        sub_mask = mask[y:y+h, x:x+w]
    kps, des = sift.detectAndCompute(sub, sub_mask)
    if not kps or des is None:
        return np.empty((0,2), np.float32), None
    # absolute coords in working image
    pts = np.array([ (kp.pt[0]+x, kp.pt[1]+y) for kp in kps ], dtype=np.float32)
    if len(pts) > max_kp:
        idx = np.argsort([-kp.response for kp in kps])[:max_kp]
        pts, des = pts[idx], des[idx]
    return pts, des

def redetect_by_sift(memory, curr_gray, sift, ratio=0.75, min_inliers=8, glare_mask=None):
    """
    Try each memory snapshot: match SIFT to current frame, RANSAC a similarity/affine,
    warp stored ROI corners -> candidate box. Returns best_box or None.
    memory item: dict(keys=['roi_box','kp','des'])
    """
    if not memory:
        return None, 0
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    best = (None, 0)
    Hc, Wc = curr_gray.shape[:2]
    roi_corners = lambda b: roi_corners_xywh(b).astype(np.float32)

    # use inverse glare mask to avoid saturated regions
    detect_mask = None if glare_mask is None else cv2.bitwise_not(glare_mask)

    for snap in memory:
        kp0, des0 = snap["kp"], snap["des"]
        if des0 is None or len(kp0) < 4:
            continue
        kps1, des1 = sift.detectAndCompute(curr_gray, detect_mask)
        if not kps1 or des1 is None:
            continue
        pts1 = np.array([k.pt for k in kps1], dtype=np.float32)

        # Lowe ratio
        mnn = bf.knnMatch(des0, des1, k=2)
        good = []
        for m,n in mnn:
            if m.distance < ratio * n.distance:
                good.append((m.queryIdx, m.trainIdx))
        if len(good) < 4:
            continue

        p0 = kp0[[i for i,_ in good]]
        p1 = pts1[[j for _,j in good]]

        M, inliers = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC,
                                                 ransacReprojThreshold=3.0,
                                                 maxIters=3000, confidence=0.99)
        if M is None or inliers is None:
            continue
        inl = int(inliers.sum())
        if inl < min_inliers:
            continue

        # warp stored ROI corners
        c0 = roi_corners(snap["roi_box"])
        c0h = np.hstack([c0, np.ones((4,1), np.float32)])
        c1 = (M @ c0h.T).T
        box = box_from_corners(c1, Wc, Hc)
        if inl > best[1]:
            best = (box, inl)

    return best

def clamp_box(x, y, w, h, W, H):
    x = max(0, min(int(x), W-1)); y = max(0, min(int(y), H-1))
    w = max(2, min(int(w), W - x)); h = max(2, min(int(h), H - y))
    return x, y, w, h

def roi_corners_xywh(box):
    x,y,w,h = box
    return np.array([[x, y],
                     [x+w, y],
                     [x+w, y+h],
                     [x, y+h]], dtype=np.float32)

def box_from_corners(corners, W, H):
    xs = corners[:,0]; ys = corners[:,1]
    x0, y0 = max(0, int(np.floor(xs.min()))), max(0, int(np.floor(ys.min())))
    x1, y1 = min(W-1, int(np.ceil(xs.max()))),  min(H-1, int(np.ceil(ys.max())))
    w, h = max(2, x1-x0), max(2, y1-y0)
    return (x0, y0, w, h)


def redetect_by_ncc(memory, curr_gray, thresh=0.80):
    """
    Slide each stored template over the whole frame via NCC. Return best box if over thresh.
    memory item: dict(keys=['tpl','tpl_size','roi_box_at_capture'])
    """
    if not memory:
        return None, 0.0
    H, W = curr_gray.shape[:2]
    best_val, best_box = -1.0, None
    for snap in memory:
        tpl = snap.get("tpl", None)
        if tpl is None or tpl.size == 0:
            continue
        res = cv2.matchTemplate(curr_gray, tpl, cv2.TM_CCOEFF_NORMED)
        minv, maxv, minl, maxl = cv2.minMaxLoc(res)
        if maxv > best_val:
            th, tw = tpl.shape[:2]
            best_val = maxv
            best_box = (maxl[0], maxl[1], tw, th)
    if best_val >= thresh:
        return best_box, best_val
    return None, best_val


# ---------- main ----------

def main():
    p=argparse.ArgumentParser(description="SIFT anchors + local NCC (feature parity)")
    p.add_argument("--video", required=True)
    p.add_argument("--gt_json", required=True)
    p.add_argument("--work_w", type=int, default=99999)
    p.add_argument("--box_size", type=int, default=75)
    p.add_argument("--pad", type=int, default=25)
    p.add_argument("--thresh", type=float, default=0.70)
    p.add_argument("--op_thr", type=float, default=0.5)
    p.add_argument("--dp_thr", type=float, default=20.0)
    p.add_argument("--dp_rel", type=float, default=0.0, help="if >0, dp_thr = dp_rel*min(H,W)")
    p.add_argument("--glare_mode", choices=["off","mask","inpaint"], default="off")
    p.add_argument("--v_hi",  type=int, default=240)
    p.add_argument("--s_lo",  type=int, default=40)
    p.add_argument("--v_hi2", type=int, default=220)
    p.add_argument("--clahe", action="store_true")
    p.add_argument("--gt_smooth", type=int, default=0)
    p.add_argument("--plot_out", type=str, default="metrics")
    p.add_argument("--save_csv", default="sift_track_metrics.csv")
    p.add_argument("--report_md", default="metrics_report.md")
    p.add_argument("--success_steps", type=int, default=101)
    p.add_argument("--prec_max", type=float, default=50.0)
    p.add_argument("--prec_step", type=float, default=1.0)
    p.add_argument("--pts", type=int, default=20, help="# of dots along center→predicted center")
    # NEW: relative sizing (fractions of min(work_w, work_h))
    p.add_argument("--box_rel", type=float, default=0.10, help="If >0, box size = box_rel * min(work_w, work_h)")
    p.add_argument("--pad_rel", type=float, default=0.3, help="If >0, pad = pad_rel * min(work_w, work_h)")
    p.add_argument("--mem_size", type=int, default=5,
                   help="# of appearance snapshots (templates+SIFT) to remember")
    p.add_argument("--mem_update_every", type=int, default=5,
                   help="add a new memory snapshot every N successful frames")
    p.add_argument("--sift_ratio", type=float, default=0.75,
                   help="Lowe ratio for SIFT matching during re-detection")
    p.add_argument("--min_inliers_recover", type=int, default=8,
                   help="min RANSAC inliers to accept SIFT re-detection")
    p.add_argument("--ncc_r", type=float, default=0.80,
                   help="NCC threshold for template re-detection over the whole frame")


    args=p.parse_args()

    cv2.setUseOptimized(True)
    cv2.setRNGSeed(1337)

    gt=load_gt_json(args.gt_json)
    if args.gt_smooth and gt:
        frames = sorted(gt.keys())
        arr = np.array([gt[f] for f in frames], dtype=np.float32)
        k = args.gt_smooth
        if k>1:
            out = np.copy(arr); csum = np.cumsum(arr, axis=0)
            for i in range(len(frames)):
                j0 = max(0, i-k+1); n = i - j0 + 1
                out[i] = (csum[i] - (csum[j0-1] if j0>0 else 0)) / n
            gt = {f: tuple(out[i]) for i,f in enumerate(frames)}

    cap=cv2.VideoCapture(args.video)
    if not cap.isOpened(): raise RuntimeError("can't open video")

    ok, first = cap.read()
    if not ok: raise RuntimeError("can't read first frame")
    H, W = first.shape[:2]
    n0 = (W/2.0, H/2.0)   # image center in native coordinates

    if args.dp_rel and args.dp_rel>0:
        args.dp_thr = args.dp_rel * min(H,W)

    scale = min(1.0, float(args.work_w)/W)
    wW, wH = int(W*scale), int(H*scale)
    first_w = cv2.resize(first, (wW,wH), cv2.INTER_AREA) if scale<1.0 else first
    gray0i, gmask0 = preprocess_gray(first_w, args.glare_mode, args.v_hi, args.s_lo, args.v_hi2, clahe=args.clahe)

    # --- NEW: resolve box size and pad in working-image pixels ---
    minwh = max(1, min(wW, wH))  # avoid zero
    # Prefer relative if provided; otherwise clamp absolutes to sane fractions
    box_size_px = (int(round(args.box_rel * minwh)) if args.box_rel > 0.0
                   else min(args.box_size, int(0.40 * minwh)))
    pad_px = (int(round(args.pad_rel * minwh)) if args.pad_rel > 0.0
              else min(args.pad, int(0.20 * minwh)))
    # Ensure ≥2 px to avoid zero-size templates/windows
    box_size_px = max(2, box_size_px)
    pad_px = max(1, pad_px)

    # initial ROI from GT (native & work space)
    gx,gy,gw,gh = map(int, gt[min(gt.keys())])
    gxw, gyw = int(round(gx*scale)), int(round(gy*scale))
    gww, ghw = max(2, int(round(gw*scale))), max(2, int(round(gh*scale)))
    t0_w = np.array([gxw + gww/2.0, gyw + ghw/2.0], dtype=float)
    tw, th = gw, gh
    tw_w, th_w = gww, ghw

    # SIFT anchors in 4 quads -> pick 2 best (exclude glare)
    sift=cv2.SIFT_create()
    quads=[(0,0,wW//2,wH//2),(wW//2,0,wW-wW//2,wH//2),
           (0,wH//2,wW//2,wH-wH//2),(wW//2,wH//2,wW-wW//2,wH//2)]
    cands=[]
    # --- appearance memory: store templates and SIFT of the *predicted* ROI ---
    mem_templ = deque(maxlen=args.mem_size)  # {'tpl','tpl_size','roi_box_at_capture'}
    mem_sift  = deque(maxlen=args.mem_size)  # {'roi_box','kp','des'}
    success_frames = 0


    for qx,qy,qw,qh in quads:
        sub = gray0i[qy:qy+qh, qx:qx+qw]
        sub_mask = None if args.glare_mode=="off" else cv2.bitwise_not(gmask0[qy:qy+qh, qx:qx+qw])
        kps,_ = sift.detectAndCompute(sub, sub_mask)
        if kps:
            best = max(kps, key=lambda k:k.response)
            cx,cy = best.pt; cx += qx; cy += qy
        else:
            cx,cy = qx+qw/2, qy+qh/2
        bx = int(round(cx - box_size_px/2)); by = int(round(cy - box_size_px/2))
        bx = max(0, min(bx, wW - box_size_px)); by = max(0, min(by, wH - box_size_px))
        cands.append((bx,by,box_size_px,box_size_px))
    boxes = cands[:2]
    tmpls = [gray0i[y:y+h, x:x+w].copy() for x,y,w,h in boxes]
    init_cent = [compute_center(b) for b in boxes]

    rows=[]; iou_series=[]; cle_series=[]; fps_series=[]; relock_counts=[]; bad_streak=0
    recover_sift = 0
    recover_ncc  = 0
    tprev=time.time()

    cv2.namedWindow("Track", cv2.WINDOW_NORMAL)

    while True:
        ok, fr = cap.read()
        if not ok: break
        frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        work = cv2.resize(fr, (wW,wH), cv2.INTER_AREA) if scale<1.0 else fr
        gi, gmask = preprocess_gray(work, args.glare_mode, args.v_hi, args.s_lo, args.v_hi2, clahe=args.clahe)

        curr=[]; matched=0; lost=0; tx=ty=None
        # track each anchor via NCC
        for i,b in enumerate(boxes):
            v,(nx,ny)=search_patch(gi, tmpls[i], b, pad_px)
            if v < args.thresh:
                curr.append(None)
                continue
            matched += 1
            w,h=b[2],b[3]
            boxes[i]=(nx,ny,w,h)
            tmpls[i]=gi[ny:ny+h, nx:nx+w].copy()  # refresh template on accept
            curr.append(compute_center(boxes[i]))

            # draw anchor boxes on native frame
            rx, ry = int(round(nx/scale)), int(round(ny/scale))
            rw, rh = int(round(w/scale)),  int(round(h/scale))
            cv2.rectangle(fr,(rx,ry),(rx+rw,ry+rh),(0,128,255),2)

        pred_box=None
        if all(c is not None for c in curr):
            src = np.array(init_cent, dtype=np.float32).reshape(-1, 2)
            dst = np.array(curr,      dtype=np.float32).reshape(-1, 2)
            M, _ = cv2.estimateAffinePartial2D(src, dst)
            pred_box = None
            if M is not None:
                # 1) Warp the initial ROI corners (in working coords)
                init_roi_w = (gxw, gyw, gww, ghw)  # initial GT ROI in working image
                c0 = compute_center(init_roi_w).reshape(1,2).astype(np.float32)
                c0h = np.hstack([c0, np.ones((1,1), np.float32)])  # (1,3)
                c1  = (M @ c0h.T).T                                # (1,2)
                cx_w, cy_w = c1.ravel()

                # Keep original width/height fixed (from GT in working space)
                pw_w, ph_w = gww, ghw
                px_w = int(round(cx_w - pw_w/2))
                py_w = int(round(cy_w - ph_w/2))

                # Map back to native coordinates
                px, py = int(round(px_w/scale)), int(round(py_w/scale))
                pw, ph = int(round(pw_w/scale)), int(round(ph_w/scale))
                pred_box = (px, py, pw, ph)

                # (Optional) keep a working-space version if you use it elsewhere:
                tx_w, ty_w, tw_w, th_w = px_w, py_w, pw_w, ph_w
                                # --- appearance memory update (every N successful frames) ---
                success_frames += 1
                if success_frames % args.mem_update_every == 0:
                    # template snapshot (working gray)
                    roi_w = (int(round(tx_w)), int(round(ty_w)), int(round(tw_w)), int(round(th_w)))
                    tpl = crop_clamped(gi, (int(round(tx_w)), int(round(ty_w)), int(round(tw_w)), int(round(th_w))))
                    if tpl.size:
                        mem_templ.append({
                            "tpl": tpl,  # keep working-res template
                            "tpl_size": (tpl.shape[1], tpl.shape[0]),  # (w,h) if you ever need it
                            "roi_box_at_capture": (int(round(tx_w)), int(round(ty_w)), int(round(tw_w)), int(round(th_w)))
                        })

                    # SIFT snapshot in working image
                    kp_xy, des = sift_in_box(gi, (int(round(tx_w)), int(round(ty_w)), int(round(tw_w)), int(round(th_w))), sift)
                    mem_sift.append({
                        "roi_box": (int(round(tx_w)), int(round(ty_w)), int(round(tw_w)), int(round(th_w))),
                        "kp": kp_xy, "des": des
                    })

        else:
            lost = 1
            # --- RE-DETECTION: (A) SIFT+RANSAC using appearance memory ---
            # Try SIFT memory first (working coords)
            (rb, inl) = redetect_by_sift(list(mem_sift), gi, sift,
                                         ratio=args.sift_ratio,
                                         min_inliers=args.min_inliers_recover)
            if rb is not None:
                recover_sift += 1
                # accept recovered box (working coords -> native coords)
                tx_w, ty_w, tw_w2, th_w2 = rb
                tx, ty = int(round(tx_w/scale)), int(round(ty_w/scale))
                tw, th = int(round(tw_w2/scale)), int(round(th_w2/scale))
                pred_box = (tx,ty,tw,th)
                lost = 0  # recovered!

                # re-initialize anchors around recovered ROI (use same quad logic but biased near ROI center)
                cxw, cyw = tx_w + tw_w2/2.0, ty_w + th_w2/2.0
                # small neighborhood around ROI center in working image
                neigh = clamp_box(cxw - box_size_px, cyw - box_size_px, 2*box_size_px, 2*box_size_px, wW, wH)
                # pick two strongest SIFT points in the neighborhood for anchor boxes
                kps, _ = sift.detectAndCompute(gi[neigh[1]:neigh[1]+neigh[3], neigh[0]:neigh[0]+neigh[2]], None)
                sel = []
                if kps:
                    kps = sorted(kps, key=lambda k:k.response, reverse=True)[:2]
                    for k in kps:
                        ax = int(round(neigh[0] + k.pt[0] - box_size_px/2))
                        ay = int(round(neigh[1] + k.pt[1] - box_size_px/2))
                        ax = max(0, min(ax, wW - box_size_px))
                        ay = max(0, min(ay, wH - box_size_px))
                        sel.append((ax, ay, box_size_px, box_size_px))
                if len(sel) < 2:
                    # fallback: use ROI corners as anchors
                    sel = [
                        (max(0, int(tx_w - box_size_px//2)), max(0, int(ty_w - box_size_px//2)), box_size_px, box_size_px),
                        (min(wW-box_size_px, int(tx_w + tw_w2 - box_size_px//2)),
                         min(wH-box_size_px, int(ty_w + th_w2 - box_size_px//2)), box_size_px, box_size_px)
                    ]
                boxes = sel[:2]
                tmpls = [gi[y:y+h, x:x+w].copy() for x,y,w,h in boxes]
                init_cent = [compute_center(b) for b in boxes]

            else:
                # --- RE-DETECTION: (B) NCC over whole frame with stored templates (native coords) ---
                rb2, val = redetect_by_ncc(list(mem_templ), gi, thresh=args.ncc_r)  # use working gray
                if rb2 is not None:
                    recover_ncc += 1
                    tx_w, ty_w, tw_w2, th_w2 = rb2  # working coords
                    tx, ty = int(round(tx_w/scale)), int(round(ty_w/scale))
                    tw, th = int(round(tw_w2/scale)), int(round(th_w2/scale))
                    pred_box = (tx,ty,tw,th)
                    lost = 0
                    # re-init anchors around recovered ROI (working coords already available)
                    cxw, cyw = tx_w + tw_w2/2.0, ty_w + th_w2/2.0
                    neigh = clamp_box(cxw - box_size_px, cyw - box_size_px,
                                    2*box_size_px, 2*box_size_px, wW, wH)

                    kps, _ = sift.detectAndCompute(gi[neigh[1]:neigh[1]+neigh[3], neigh[0]:neigh[0]+neigh[2]], None)
                    sel = []
                    if kps:
                        kps = sorted(kps, key=lambda k:k.response, reverse=True)[:2]
                        for k in kps:
                            ax = int(round(neigh[0] + k.pt[0] - box_size_px/2))
                            ay = int(round(neigh[1] + k.pt[1] - box_size_px/2))
                            ax = max(0, min(ax, wW - box_size_px))
                            ay = max(0, min(ay, wH - box_size_px))
                            sel.append((ax, ay, box_size_px, box_size_px))
                    if len(sel) < 2:
                        sel = [
                            (max(0, int(tx_w - box_size_px//2)), max(0, int(ty_w - box_size_px//2)), box_size_px, box_size_px),
                            (min(wW-box_size_px, int(tx_w + tw_w2 - box_size_px//2)),
                             min(wH-box_size_px, int(ty_w + th_w2 - box_size_px//2)), box_size_px, box_size_px)
                        ]
                    boxes = sel[:2]
                    tmpls = [gi[y:y+h, x:x+w].copy() for x,y,w,h in boxes]
                    init_cent = [compute_center(b) for b in boxes]

                else:
                    # --- FALLBACK: your original quadrant SIFT reseed ---
                    cands=[]
                    for qx,qy,qw,qh in quads:
                        sub = gi[qy:qy+qh, qx:qx+qw]
                        sub_mask = None if args.glare_mode=="off" else cv2.bitwise_not(gmask[qy:qy+qh, qx:qx+qw])
                        kps,_ = sift.detectAndCompute(sub, sub_mask)
                        if kps:
                            best=max(kps,key=lambda k:k.response)
                            cx,cy=best.pt; cx+=qx; cy+=qy
                        else:
                            cx,cy=qx+qw/2, qy+qh/2
                        bx=int(round(cx - box_size_px/2)); by=int(round(cy - box_size_px/2))
                        bx=max(0,min(bx,wW - box_size_px)); by=max(0,min(by,wH - box_size_px))
                        cands.append((bx,by,box_size_px,box_size_px))
                    boxes = cands[:2]
                    tmpls = [gi[y:y+h, x:x+w].copy() for x,y,w,h in boxes]
                    init_cent = [compute_center(b) for b in boxes]
        # FPS
        now=time.time(); inst_fps = 1.0/max(1e-9, now - tprev); tprev=now
        fps_series.append(inst_fps)

        # Metrics (strict: only if we have a prediction this frame)
        gt_box = gt.get(frame_no, None)
        iou_v = iou(pred_box, gt_box) if (gt_box and pred_box) else None
        cle_v = center_distance(pred_box, gt_box) if (gt_box and pred_box) else None
        iou_series.append(iou_v if iou_v is not None else np.nan)
        cle_series.append(cle_v if cle_v is not None else np.nan)

        # Recoverability accounting
        if gt_box:
            below = (iou_v is None) or (iou_v <= args.op_thr) or bool(lost)
            if below: bad_streak += 1
            else:
                if bad_streak > 0: relock_counts.append(bad_streak); bad_streak=0

        # Draw boxes + HUD
        if pred_box:
            px,py,pw,ph = pred_box
            cv2.rectangle(fr,(px,py),(px+pw,py+ph),(0,255,0),2)
            # --- dotted trajectory center→predicted center ---
            tcx = px + pw/2.0
            tcy = py + ph/2.0
            traj = interpolate((int(n0[0]), int(n0[1])), (int(tcx), int(tcy)), args.pts)
            for qx, qy in traj:
                cv2.circle(fr, (qx, qy), 3, (0,255,0), -1)   # green dots
            cv2.circle(fr, (int(n0[0]), int(n0[1])), 5, (0,0,255), -1)  # red center dot

        if gt_box:
            gx,gy,gw,gh = map(int, gt_box)
            cv2.rectangle(fr,(gx,gy),(gx+gw,gy+gh),(255,0,0),2)
        if lost:
            cv2.putText(fr,"LOST",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)
        elif pred_box and matched < 2:
            cv2.putText(fr,"RE-DETECTED",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)


        cv2.putText(fr,f"SIFT+NCC  FPS:{inst_fps:5.1f}  anchors:{matched}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        if gt_box and (iou_v is not None):
            ok_op = iou_v > args.op_thr
            ok_dp = (cle_v is not None) and (cle_v < args.dp_thr)
            cv2.putText(fr, f"IoU:{iou_v:.2f}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0,255,0) if ok_op else (0,0,255), 2)
            if cle_v is not None:
                cv2.putText(fr,f"CLE:{cle_v:.1f}px", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0,255,0) if ok_dp else (0,0,255), 2)

        cv2.imshow("Track", fr)
        if cv2.waitKey(1)==27: break

        # CSV row (match klt_full.py header; use matched anchors as "inliers")
        rows.append([
            frame_no, matched,
            pred_box[0] if pred_box else "", pred_box[1] if pred_box else "",
            gt_box[0] if gt_box else "", gt_box[1] if gt_box else "",
            gt_box[2] if gt_box else "", gt_box[3] if gt_box else "",
            iou_v if iou_v is not None else "", cle_v if cle_v is not None else "",
            inst_fps, int(lost)
        ])

    cap.release(); cv2.destroyAllWindows()
    print(f"Recoveries — SIFT: {recover_sift}, NCC: {recover_ncc}, Mem size: tpl={len(mem_templ)}, sift={len(mem_sift)}")

    if bad_streak > 0: relock_counts.append(bad_streak)

    # summaries
    ious = np.array([v for v in iou_series if not np.isnan(v)], dtype=float)
    dists= np.array([v for v in cle_series if not np.isnan(v)], dtype=float)
    mean_iou = float(ious.mean()) if ious.size else float("nan")

    op_flags=[(v>args.op_thr) for v in ious] if ious.size else []
    dp_flags=[(v<args.dp_thr) for v in dists] if dists.size else []
    final_op=(op_flags.count(True)/len(op_flags)) if op_flags else float("nan")
    final_dp=(dp_flags.count(True)/len(dp_flags)) if dp_flags else float("nan")

    success_thr = np.linspace(0.0,1.0,args.success_steps) if ious.size else np.array([])
    success_curve = np.array([(ious >= t).mean() for t in success_thr]) if ious.size else np.array([])
    success_auc = float(np.trapezoid(success_curve, success_thr)) if success_curve.size else float("nan")

    prec_thr = np.arange(0.0, args.prec_max + 1e-6, args.prec_step) if dists.size else np.array([])
    precision_curve = np.array([(dists <= t).mean() for t in prec_thr]) if dists.size else np.array([])
    precision_auc = float(np.trapezoid(precision_curve, prec_thr) / max(prec_thr[-1], 1e-9)) if precision_curve.size else float("nan")

    mean_fps = float(np.mean(fps_series)) if fps_series else float("nan")

    dp10 = 0.10 * min(H,W)
    dp10_val = float(np.interp(dp10, prec_thr, precision_curve)) if precision_curve.size else float("nan")

    relock_mean = float(np.mean(relock_counts)) if relock_counts else float("nan")
    relock_max  = int(max(relock_counts)) if relock_counts else 0


    with open(args.save_csv,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["frame","inliers","pred_x","pred_y","gt_x","gt_y","gt_w","gt_h",
                    "iou","cle_px","fps","lost"])
        w.writerows(rows)

    # OP/DP running plot
    if ious.size or dists.size:
        op_running = np.cumsum(op_flags)/np.arange(1,len(op_flags)+1) if op_flags else []
        dp_running = np.cumsum(dp_flags)/np.arange(1,len(dp_flags)+1) if dp_flags else []
        if len(op_running) or len(dp_running):
            plt.figure()
            if len(op_running): plt.plot(op_running, label=f"OP (IoU>{args.op_thr})")
            if len(dp_running): plt.plot(dp_running, label=f"DP (dist<{args.dp_thr:.0f}px)")
            plt.xlabel("Frames with GT"); plt.ylabel("Cumulative precision"); plt.ylim([0,1.05])
            plt.legend(); plt.title("Running OP & DP"); plt.tight_layout()
            plt.savefig(f"{args.plot_out}_op_dp.png", dpi=150); plt.close()

    # per-frame plots
    if len(iou_series):
        plt.figure(figsize=(6,3)); plt.plot(iou_series, lw=1.2)
        plt.axhline(args.op_thr, ls="--", label=f"OP thr {args.op_thr}")
        plt.xlabel("Frame"); plt.ylabel("IoU"); plt.ylim([0,1.02]); plt.legend(frameon=False); plt.tight_layout()
        plt.savefig(f"{args.plot_out}_iou.png", dpi=200); plt.close()

    if len(cle_series):
        plt.figure(figsize=(6,3)); plt.plot(cle_series, lw=1.2)
        plt.axhline(args.dp_thr, ls="--", label=f"DP thr {args.dp_thr}px")
        plt.xlabel("Frame"); plt.ylabel("CLE [px]"); plt.legend(frameon=False); plt.tight_layout()
        plt.savefig(f"{args.plot_out}_cle.png", dpi=200); plt.close()

    if fps_series:
        plt.figure(figsize=(6,3)); plt.plot(fps_series, lw=1.2)
        plt.xlabel("Frame"); plt.ylabel("FPS"); plt.tight_layout()
        plt.savefig(f"{args.plot_out}_fps.png", dpi=200); plt.close()

    # success
    if success_curve.size:
        op_at_05 = float(np.interp(0.5, success_thr, success_curve))
        plt.figure(figsize=(6,4))
        plt.plot(success_thr, success_curve, lw=2.0, label=f"Success (AUC={success_auc:.3f})")
        plt.vlines(0.5, 0, op_at_05, linestyles="--"); plt.scatter([0.5],[op_at_05], s=28)
        plt.xlabel("IoU threshold"); plt.ylabel("Success rate"); plt.ylim([0,1.02])
        plt.legend(frameon=False); plt.tight_layout()
        plt.savefig(f"{args.plot_out}_success_curve.png", dpi=300); plt.close()

        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(success_thr, success_curve, linewidth=2.0, label=f"Success (AUC={success_auc:.3f})")
        ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.02)
        poster_axes(ax, "IoU threshold", "Success rate")
        ax.vlines(0.5, 0, op_at_05, linestyles="--", linewidth=1.5)
        annotate_point(ax, 0.5, op_at_05, f"OP@0.5={op_at_05:.2f}")
        ax.legend(loc="lower left", frameon=False)
        fig.tight_layout(); fig.savefig(f"{args.plot_out}_success_curve_poster.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # precision
    if precision_curve.size:
        dp_at_thr = float(np.interp(args.dp_thr, prec_thr, precision_curve))
        plt.figure(figsize=(6,4))
        plt.plot(prec_thr, precision_curve, lw=2.0, label=f"Precision (AUC@{args.prec_max:.0f}px={precision_auc:.3f})")
        plt.vlines(args.dp_thr, 0, dp_at_thr, linestyles="--"); plt.scatter([args.dp_thr],[dp_at_thr], s=28)
        plt.xlabel("Distance threshold (px)"); plt.ylabel("Precision"); plt.ylim([0,1.02])
        plt.legend(frameon=False); plt.tight_layout()
        plt.savefig(f"{args.plot_out}_precision_curve.png", dpi=300); plt.close()

        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(prec_thr, precision_curve, linewidth=2.0,
                label=f"Precision (AUC@{args.prec_max:.0f}px={precision_auc:.3f})")
        ax.set_xlim(0, args.prec_max); ax.set_ylim(0, 1.02)
        poster_axes(ax, "Distance threshold (px)", "Precision")
        ax.vlines(args.dp_thr, 0, dp_at_thr, linestyles="--", linewidth=1.5)
        annotate_point(ax, args.dp_thr, dp_at_thr, f"DP@{args.dp_thr:.0f}px={dp_at_thr:.2f}")
        ax.legend(loc="lower right", frameon=False)
        fig.tight_layout(); fig.savefig(f"{args.plot_out}_precision_curve_poster.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    with open(f"{args.plot_out}_curves.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["type","threshold","value"])
        for t,v in zip(success_thr, success_curve): w.writerow(["success_iou", t, v])
        for t,v in zip(prec_thr, precision_curve): w.writerow(["precision_dist", t, v])

    # report
    vid_name = pathlib.Path(args.video).name
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(args.report_md, "w") as f:
        f.write(f"# Tracking Metrics Report\n\n**Video:** `{vid_name}`  \n**Generated:** {ts}\n\n")
        f.write("## Headline Results\n")
        if ious.size:
            f.write(f"- Mean IoU: {mean_iou:.3f}\n")
            f.write(f"- Final OP (IoU>{args.op_thr}): {final_op*100:.1f}%\n")
            f.write(f"- Final DP (dist<{args.dp_thr:.0f}px): {final_dp*100:.1f}%\n")
            f.write(f"- Success AUC (IoU 0→1): {success_auc:.3f}\n")
            f.write(f"- Precision AUC (0→{args.prec_max:.0f}px): {precision_auc:.3f}\n")
        else:
            f.write("- No GT frames available for metrics\n")
        f.write(f"- DP@0.10·min(W,H) [{int(dp10)}px]: "
                f"{'n/a' if np.isnan(dp10_val) else f'{dp10_val*100:.1f}%'}\n")
        f.write(f"- Mean FPS: {mean_fps:.1f}\n\n")
        f.write("## Parameters\n")
        f.write(f"- work_w: {args.work_w}\n")
        f.write(f"- box_size(arg): {args.box_size}, pad(arg): {args.pad}, NCC thresh: {args.thresh}\n")
        f.write(f"- box_size(resolved px): {box_size_px}, pad(resolved px): {pad_px}\n")
        f.write(f"- box_rel: {args.box_rel}, pad_rel: {args.pad_rel}\n")
        f.write(f"- glare_mode: {args.glare_mode}, clahe: {args.clahe}\n")
        f.write(f"- HSV v_hi/s_lo/v_hi2: {args.v_hi}/{args.s_lo}/{args.v_hi2}\n")
        f.write(f"- OP thr: {args.op_thr}, DP thr: {args.dp_thr}\n")
        f.write(f"- GT JSON: `{args.gt_json}`\n")
        if relock_counts:
            f.write(f"- Recoverability: mean frames to re-lock: {relock_mean:.1f}, max: {relock_max}\n")
        else:
            f.write("- Recoverability: no loss events with GT\n")

    print("\n=== Summary ===")
    print(f"Frames processed                 : {len(fps_series)}")
    print(f"Mean FPS                         : {round(mean_fps,1)}")
    if ious.size:
        print(f"Mean IoU                         : {round(mean_iou,3)}")
        print(f"Final OP (IoU>{args.op_thr})     : {round(100*final_op,1)}%")
        print(f"Final DP (dist<{args.dp_thr:.0f}px)  : {round(100*final_dp,1)}%")
        print(f"Success AUC (IoU 0→1)            : {round(success_auc,3)}")
        print(f"Precision AUC (0→{args.prec_max:.0f}px) : {round(precision_auc,3)}")
    print(f"DP@0.10·min(W,H) [{int(dp10)}px]     : "
          f"{'n/a' if np.isnan(dp10_val) else f'{dp10_val*100:.1f}%'}")
    if relock_counts:
        print(f"Recoverability: mean frames to re-lock : {relock_mean:.1f}")
        print(f"Recoverability: max frames to re-lock  : {relock_max}")
    else:
        print("Recoverability                   : no loss events with GT")

    print(f"\nSaved: {args.save_csv}, {args.plot_out}_*.png, {args.plot_out}_curves.csv, {args.report_md}")

if __name__=="__main__":
    main()
