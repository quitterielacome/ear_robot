#!/usr/bin/env python3
"""
KLT + RANSAC (similarity) tracker with robust occlusion handling:
- Shi–Tomasi/LK corners each frame (orange)
- Dotted trajectory from frame center to current target center (--pts)
- Glare handling: off/mask/inpaint (HSV), optional CLAHE (--clahe)
- GT first box initialises ROI; optional GT smoothing
- LK forward–backward check + residual gating before RANSAC
- Model acceptance uses both inlier count and inlier ratio
- LOST handling: adaptive corner reseed around last pose + NCC re-lock
- NCC re-lock resets the pose (no compounding stale transforms)
- Metrics: OP/DP, IoU/CLE per-frame, Success & Precision curves (+AUCs)
- Adds DP@0.10·min(W,H) to report; saves poster-quality plots
- CSV matches the SIFT/NCC script for easy aggregation
- ESC to quit
"""

import cv2, argparse, time, csv, numpy as np, json, os, pathlib, datetime, re
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

# -------- Matplotlib headless if needed --------
import matplotlib
if "DISPLAY" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------- helpers --------------------
def compute_center(b):
    x,y,w,h = b
    return np.array([x + w/2.0, y + h/2.0], dtype=np.float32)

def interpolate(p1,p2,n):
    pts=[]
    for i in range(1,n+1):
        a=i/(n+1.0)
        pts.append((int((1-a)*p1[0]+a*p2[0]), int((1-a)*p1[1]+a*p2[1])))
    return pts

def poster_axes(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

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

def scale_affine_2x3(M_work, s):
    A = np.vstack([M_work, [0,0,1]]).astype(np.float32)
    S  = np.array([[s,0,0],[0,s,0],[0,0,1]], np.float32)
    Si = np.array([[1/s,0,0],[0,1/s,0],[0,0,1]], np.float32)
    return (Si @ A @ S)[:2,:]

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
    with open(path,"r") as f: data = json.load(f)
    id_to_fname = {im["id"]: im["file_name"] for im in data["images"]}
    gt = {}
    for ann in data["annotations"]:
        fname = id_to_fname.get(ann["image_id"], "")
        m = re.search(r"frame_(\d+)", fname)
        if not m: continue
        gt[int(m.group(1))] = tuple(ann["bbox"])
    return gt

def smooth_gt_boxes(gt_dict, window):
    if window is None or window < 2 or not gt_dict: return gt_dict
    frames = sorted(gt_dict.keys())
    arr = np.array([gt_dict[f] for f in frames], dtype=np.float32)
    out = np.copy(arr); csum = np.cumsum(arr, axis=0)
    for i in range(len(frames)):
        j0 = max(0, i-window+1); n = i - j0 + 1
        out[i] = (csum[i] - (csum[j0-1] if j0>0 else 0)) / n
    return {f: tuple(out[i]) for i,f in enumerate(frames)}

def running_ratio(bools):
    out=[]; good=0
    for i,flag in enumerate(bools, start=1):
        if flag: good += 1
        out.append(good / i)
    return out

# -------------------- main --------------------
def main():
    print(">> KLT NEW + FB/LK gating + NCC re-lock v1.1")
    ap = argparse.ArgumentParser(description="KLT + RANSAC (similarity) with robust occlusion handling")
    ap.add_argument("--video", required=True)
    ap.add_argument("--work_w", type=int, default=720)
    ap.add_argument("--n_pts", type=int, default=150)
    ap.add_argument("--pts", type=int, default=20, help="# of dots along center→target line")
    ap.add_argument("--inpaint", action="store_true", help="alias for --glare_mode inpaint")
    ap.add_argument("--save_csv", default="klt_track_metrics.csv")

    ap.add_argument("--glare_mode", choices=["off","mask","inpaint"], default="mask")
    ap.add_argument("--v_hi",  type=int, default=240)
    ap.add_argument("--s_lo",  type=int, default=40)
    ap.add_argument("--v_hi2", type=int, default=220)
    ap.add_argument("--clahe", action="store_true")

    ap.add_argument("--gt_json", type=str, required=True)
    ap.add_argument("--gt_smooth", type=int, default=0)
    ap.add_argument("--op_thr", type=float, default=0.5)
    ap.add_argument("--dp_thr", type=float, default=20.0)
    ap.add_argument("--dp_rel", type=float, default=0.0)

    ap.add_argument("--plot_out", type=str, default="metrics")
    ap.add_argument("--success_steps", type=int, default=101)
    ap.add_argument("--prec_max", type=float, default=50.0)
    ap.add_argument("--prec_step", type=float, default=1.0)
    ap.add_argument("--report_md", type=str, default="metrics_report.md")

    # Robustness knobs
    ap.add_argument("--fb_thr", type=float, default=1.5, help="forward–backward error (px)")
    ap.add_argument("--lk_err_thr", type=float, default=20.0, help="LK residual threshold")
    ap.add_argument("--min_inlier_ratio", type=float, default=0.25, help="RANSAC inliers / candidates")
    ap.add_argument("--ncc_thr", type=float, default=0.80, help="template re-lock threshold")
    ap.add_argument("--roi_expand", type=float, default=0.35, help="expand last ROI for reseed window")

    ap.add_argument("--ransac_reproj", type=float, default=3.0)

    args = ap.parse_args()
    if args.inpaint:
        args.glare_mode = "inpaint"

    cv2.setUseOptimized(True)
    cv2.setRNGSeed(1337)

    # Load & optionally smooth GT
    gt_raw = load_gt_json(args.gt_json) if args.gt_json else {}
    gt = smooth_gt_boxes(gt_raw, args.gt_smooth) if args.gt_smooth and gt_raw else gt_raw

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened(): raise RuntimeError("Cannot open video")
    ok, frame0 = cap.read()
    if not ok: raise RuntimeError("Cannot read first frame")
    H,W = frame0.shape[:2]

    if args.dp_rel and args.dp_rel > 0:
        args.dp_thr = args.dp_rel * min(H, W)

    # Frame index helper
    def current_frame_idx():
        f = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        return max(0, f)

    scale = min(1.0, float(args.work_w)/W)
    wW, wH = int(W*scale), int(H*scale)
    work0 = cv2.resize(frame0, (wW,wH), cv2.INTER_AREA) if scale<1.0 else frame0

    gray0i, gmask0 = preprocess_gray(work0, args.glare_mode, args.v_hi, args.s_lo, args.v_hi2, clahe=args.clahe)
    detect_mask0 = None if args.glare_mode=="off" else cv2.bitwise_not(gmask0)

    p0 = cv2.goodFeaturesToTrack(gray0i, maxCorners=args.n_pts, qualityLevel=0.01, minDistance=7,
                                 blockSize=7, mask=detect_mask0, useHarrisDetector=False)
    if p0 is None or len(p0)<10: raise RuntimeError("Too few features found.")

    # initial ROI from first GT box (native)
    if not gt: raise RuntimeError("No ground truth available to initialise ROI")
    first_gt_frame = min(gt.keys())
    gx,gy,gw,gh = map(int, gt[first_gt_frame])
    t0 = compute_center((gx,gy,gw,gh))
    tw, th = gw, gh

    lk_params = dict(winSize=(21,21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    prev_gray = gray0i.copy()
    prev_pts  = p0.copy()
    last_M_full = np.array([[1,0,0],[0,1,0]], np.float32)

    # visuals helpers
    n0 = np.array([W/2.0, H/2.0], dtype=np.float32)  # frame center

    # accumulators
    rows=[]; iou_series=[]; cle_series=[]; fps_series=[]; relock_counts=[]; bad_streak=0
    fb_total=0; fb_kept_total=0; ncc_recovs=0
    tprev=time.time()

    cv2.namedWindow("Track", cv2.WINDOW_NORMAL)

    # cached ROI template (working-res) for NCC re-lock
    roi_tpl_work = None
    roi_tpl_wh   = None

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame_no = current_frame_idx()

        work = cv2.resize(frame, (wW,wH), cv2.INTER_AREA) if scale<1.0 else frame
        grayi, gmask = preprocess_gray(work, args.glare_mode, args.v_hi, args.s_lo, args.v_hi2, clahe=args.clahe)

        # Re-detect if needed (global or masked by glare)
        if prev_pts is None or len(prev_pts)<10:
            detect_mask = None if args.glare_mode=="off" else cv2.bitwise_not(gmask)
            prev_pts = cv2.goodFeaturesToTrack(grayi, maxCorners=args.n_pts, qualityLevel=0.01, minDistance=7,
                                               blockSize=7, mask=detect_mask, useHarrisDetector=False)
            prev_gray = grayi.copy()

        disp = frame.copy(); lost=1; inl_count=0; tx=ty=None
        p = None

        if prev_pts is not None and len(prev_pts) >= 10:
            # --- Forward LK ---
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, grayi, prev_pts, None, **lk_params)
            # --- Backward LK for FB check ---
            p0r, st_b, err_b = cv2.calcOpticalFlowPyrLK(grayi, prev_gray, p1, None, **lk_params)

            if p1 is not None and st is not None and p0r is not None and st_b is not None:
                old = prev_pts.reshape(-1,2)
                new = p1.reshape(-1,2)
                back= p0r.reshape(-1,2)
                m_fwd = (st.reshape(-1) == 1)
                m_bwd = (st_b.reshape(-1) == 1)
                fb_err = np.linalg.norm(old - back, axis=1)
                lk_err = err.reshape(-1) if err is not None else np.full(len(m_fwd), np.inf)

                keep = m_fwd & m_bwd & (fb_err < args.fb_thr) & (lk_err < args.lk_err_thr)
                fb_total += keep.size
                fb_kept_total += int(keep.sum())

                good_old = old[keep].astype(np.float32)
                good_new = new[keep].astype(np.float32)

                if good_old.shape[0] >= 3:
                    # Similarity/affine (partial) with RANSAC
                    try:
                        M_work, inliers = cv2.estimateAffinePartial2D(
                            good_old, good_new, method=cv2.RANSAC,
                            ransacReprojThreshold=float(args.ransac_reproj),
                            maxIters=2000, confidence=0.99
                        )
                    except cv2.error:
                        M_work, inliers = None, None

                    inl_count = int(inliers.sum()) if (inliers is not None) else 0
                    inlier_ratio = inl_count / max(1, len(good_old))
                    ok_model = (M_work is not None) and (inl_count >= 6) and (inlier_ratio >= args.min_inlier_ratio)

                    if ok_model:
                        # scale to native and accumulate
                        M_full_2x3 = scale_affine_2x3(M_work, scale)
                        H_step  = np.vstack([M_full_2x3, [0,0,1]]).astype(np.float32)
                        H_cum   = np.vstack([last_M_full, [0,0,1]]).astype(np.float32)
                        H_cum   = H_step @ H_cum
                        last_M_full = H_cum[:2,:]

                        p = last_M_full.dot(np.array([t0[0], t0[1], 1.0], dtype=np.float32))
                        tx, ty = int(p[0] - tw/2), int(p[1] - th/2)

                        # cache a fresh working-res template around the predicted ROI
                        tx_w, ty_w = int(round(p[0]/scale - tw/2)), int(round(p[1]/scale - th/2))
                        xw = max(0, tx_w); yw = max(0, ty_w)
                        ww = min(tw, wW - xw); hh = min(th, wH - yw)
                        if ww > 4 and hh > 4:
                            roi_tpl_work = grayi[yw:yw+hh, xw:xw+ww].copy()
                            roi_tpl_wh   = (ww, hh)

                        # draw predicted box & trajectory
                        cv2.rectangle(disp, (tx,ty), (tx+tw,ty+th), (0,255,0), 2)
                        traj = interpolate((int(n0[0]),int(n0[1])), (int(p[0]),int(p[1])), args.pts)
                        for q in traj: cv2.circle(disp, q, 3, (0,255,0), -1)
                        cv2.circle(disp, (int(n0[0]),int(n0[1])), 5, (0,0,255), -1)

                        # show LK corners kept (orange)
                        for pt in good_new:
                            x,y = int(pt[0]/scale), int(pt[1]/scale)
                            cv2.rectangle(disp, (x-2,y-2), (x+2,y+2), (0,128,255), 1)

                        lost = 0

                # update prev_pts only if we have good_new
                if not lost and 'good_new' in locals() and good_new is not None and len(good_new)>0:
                    prev_pts = good_new.reshape(-1,1,2).copy()
                else:
                    prev_pts = None  # force reseed next loop

        # LOST handling (adaptive reseed + NCC re-lock)
        if lost:
            cv2.putText(disp, "LOST", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

            # Try reseed around last known pose (if we have any)
            if p is not None or tx is not None:
                cx = (tx + tw/2) if tx is not None else (p[0] if p is not None else W/2)
                cy = (ty + th/2) if ty is not None else (p[1] if p is not None else H/2)
                cx_w, cy_w = int(round(cx*scale)), int(round(cy*scale))
                ww = int(round((1.0 + 2*args.roi_expand) * tw * scale))
                hh = int(round((1.0 + 2*args.roi_expand) * th * scale))
                x0 = max(0, cx_w - ww//2); y0 = max(0, cy_w - hh//2)
                x1 = min(wW, x0 + ww);     y1 = min(wH, y0 + hh)
                roi_mask = np.zeros_like(grayi, np.uint8)
                roi_mask[y0:y1, x0:x1] = 255
                if args.glare_mode != "off":
                    roi_mask = cv2.bitwise_and(roi_mask, cv2.bitwise_not(gmask))

                prev_pts = cv2.goodFeaturesToTrack(
                    grayi, maxCorners=args.n_pts, qualityLevel=0.01, minDistance=7,
                    blockSize=7, mask=roi_mask, useHarrisDetector=False
                )

            # If still nothing: try NCC re-lock from cached template
            if (prev_pts is None or len(prev_pts) < 10) and (roi_tpl_work is not None):
                res = cv2.matchTemplate(grayi, roi_tpl_work, cv2.TM_CCOEFF_NORMED)
                minv, maxv, minl, maxl = cv2.minMaxLoc(res)
                if maxv >= args.ncc_thr:
                    ncc_recovs += 1
                    xw, yw = maxl
                    ww, hh = roi_tpl_wh
                    # native coords top-left
                    tx, ty = int(round(xw/scale)), int(round(yw/scale))
                    # reset the pose so it maps t0 to this new center (pure translation)
                    cx, cy = tx + tw/2, ty + th/2
                    last_M_full = np.array([[1,0, cx - t0[0]],
                                            [0,1, cy - t0[1]]], dtype=np.float32)
                    # reseed features strictly inside this box (avoid glare)
                    box_mask = np.zeros_like(grayi, np.uint8)
                    x1, y1 = min(wW, xw+ww), min(wH, yw+hh)
                    box_mask[yw:y1, xw:x1] = 255
                    if args.glare_mode != "off":
                        box_mask = cv2.bitwise_and(box_mask, cv2.bitwise_not(gmask))
                    prev_pts = cv2.goodFeaturesToTrack(grayi, maxCorners=args.n_pts, qualityLevel=0.01,
                                                       minDistance=7, blockSize=7, mask=box_mask, useHarrisDetector=False)
                    # draw the re-locked box
                    cv2.rectangle(disp, (tx,ty), (tx+tw,ty+th), (0,255,255), 2)
                    lost = 0

        # FPS
        now=time.time(); inst_fps = 1.0/max(1e-9, now - tprev); tprev=now
        fps_series.append(inst_fps)

        # Metrics vs GT
        frame_idx = frame_no
        gt_box = gt.get(frame_idx, None)
        pred_box = (tx,ty,tw,th) if (tx is not None) else None
        iou_v = iou(pred_box, gt_box) if gt_box else None
        cle_v = center_distance(pred_box, gt_box) if gt_box else None
        iou_series.append(iou_v if iou_v is not None else np.nan)
        cle_series.append(cle_v if cle_v is not None else np.nan)

        # Draw GT & HUD
        if gt_box:
            gx,gy,gw,gh = map(int, gt_box)
            cv2.rectangle(disp, (gx,gy), (gx+gw,gy+gh), (255,0,0), 2)
            if (iou_v is None) or (iou_v <= args.op_thr):
                bad_streak += 1
            else:
                if bad_streak>0: relock_counts.append(bad_streak); bad_streak=0

        hud = f"KLT  FPS:{inst_fps:5.1f}  inl:{inl_count:3d}"
        if fb_total>0:
            hud += f"  FBkeep:{(100.0*fb_kept_total/max(1,fb_total)):.0f}%"
        cv2.putText(disp, hud, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2)

        if gt_box and (iou_v is not None):
            ok_op = iou_v > args.op_thr
            ok_dp = (cle_v is not None) and (cle_v < args.dp_thr)
            cv2.putText(disp, f"IoU:{iou_v:.2f}", (10,90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if ok_op else (0,0,255), 2)
            if cle_v is not None:
                cv2.putText(disp, f"CLE:{cle_v:.1f}px", (10,120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if ok_dp else (0,0,255), 2)

        cv2.imshow("Track", disp)
        if cv2.waitKey(1) & 0xFF == 27: break

        # CSV row
        rows.append([
            frame_idx, inl_count,
            pred_box[0] if pred_box else "", pred_box[1] if pred_box else "",
            gt_box[0] if gt_box else "", gt_box[1] if gt_box else "",
            gt_box[2] if gt_box else "", gt_box[3] if gt_box else "",
            iou_v if iou_v is not None else "", cle_v if cle_v is not None else "",
            inst_fps, int(lost)
        ])

        prev_gray = grayi.copy()

    cap.release(); cv2.destroyAllWindows()
    if bad_streak > 0: relock_counts.append(bad_streak)

    # ---------- Summaries + curves ----------
    op_flags = [ (v > args.op_thr) for v in iou_series if not np.isnan(v) ]
    dp_flags = [ (v < args.dp_thr) for v in cle_series if not np.isnan(v) ]
    op_running = running_ratio(op_flags) if op_flags else []
    dp_running = running_ratio(dp_flags) if dp_flags else []

    mean_iou = np.nanmean(np.array(iou_series)) if iou_series else float("nan")
    final_op = op_running[-1] if op_running else float("nan")
    final_dp = dp_running[-1] if dp_running else float("nan")
    mean_fps = float(np.mean(fps_series)) if fps_series else float("nan")
    relock_mean = float(np.mean(relock_counts)) if relock_counts else float("nan")
    relock_max  = int(max(relock_counts)) if relock_counts else 0

    # Success curve (IoU sweep)
    ious = np.array([v for v in iou_series if not np.isnan(v)], dtype=np.float32)
    success_thresh = np.linspace(0.0, 1.0, args.success_steps) if len(ious)>0 else np.array([])
    success_curve = np.array([(ious >= t).mean() for t in success_thresh]) if len(ious)>0 else np.array([])
    success_auc = float(np.trapezoid(success_curve, success_thresh)) if len(success_curve)>0 else float("nan")

    # Precision curve (distance sweep)
    dists = np.array([v for v in cle_series if not np.isnan(v)], dtype=np.float32)
    prec_thresh = np.arange(0.0, args.prec_max + 1e-6, args.prec_step) if len(dists)>0 else np.array([])
    precision_curve = np.array([(dists <= t).mean() for t in prec_thresh]) if len(dists)>0 else np.array([])
    precision_auc = float(np.trapezoid(precision_curve, prec_thresh) / max(prec_thresh[-1], 1e-9)) if len(precision_curve)>0 else float("nan")

    # DP@0.10·min(W,H)
    dp10 = 0.10 * min(H,W)
    dp10_val = float(np.interp(dp10, prec_thresh, precision_curve)) if precision_curve.size else float("nan")

    # Save per-frame CSV
    with open(args.save_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame","inliers","pred_x","pred_y","gt_x","gt_y","gt_w","gt_h",
                    "iou","cle_px","fps","lost"])
        w.writerows(rows)

    # Save curves CSV
    with open(f"{args.plot_out}_curves.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["type","threshold","value"])
        for t,v in zip(success_thresh, success_curve): w.writerow(["success_iou", t, v])
        for t,v in zip(prec_thresh, precision_curve): w.writerow(["precision_dist", t, v])

    # ----- Plots -----
    if op_running or dp_running:
        plt.figure()
        if op_running: plt.plot(op_running, label=f"OP (IoU>{args.op_thr})")
        if dp_running: plt.plot(dp_running, label=f"DP (dist<{args.dp_thr:.0f}px)")
        plt.xlabel("Frames with GT"); plt.ylabel("Cumulative precision"); plt.ylim([0,1.05])
        plt.legend(); plt.title("Running OP & DP"); plt.tight_layout()
        plt.savefig(f"{args.plot_out}_op_dp.png", dpi=150); plt.close()

    if any(not np.isnan(v) for v in iou_series):
        plt.figure(); plt.plot([v if not np.isnan(v) else np.nan for v in iou_series])
        plt.axhline(args.op_thr, linestyle="--"); plt.xlabel("Frame"); plt.ylabel("IoU"); plt.ylim([0,1.05])
        plt.tight_layout(); plt.savefig(f"{args.plot_out}_iou.png", dpi=150); plt.close()

    if any(not np.isnan(v) for v in cle_series):
        plt.figure(); plt.plot([v if not np.isnan(v) else np.nan for v in cle_series])
        plt.axhline(args.dp_thr, linestyle="--"); plt.xlabel("Frame"); plt.ylabel("Pixels")
        plt.tight_layout(); plt.savefig(f"{args.plot_out}_cle.png", dpi=150); plt.close()

    if fps_series:
        plt.figure(); plt.plot(fps_series); plt.xlabel("Frame"); plt.ylabel("FPS")
        plt.tight_layout(); plt.savefig(f"{args.plot_out}_fps.png", dpi=150); plt.close()

    if len(success_curve) > 0:
        op_at_05 = float(np.interp(0.5, success_thresh, success_curve))
        plt.figure(figsize=(6,4))
        plt.plot(success_thresh, success_curve, lw=2.0, label=f"Success (AUC={success_auc:.3f})")
        plt.vlines(0.5, 0, op_at_05, linestyles="--"); plt.scatter([0.5],[op_at_05], s=28)
        plt.xlabel("IoU threshold"); plt.ylabel("Success rate"); plt.ylim([0,1.02])
        plt.legend(frameon=False); plt.tight_layout()
        plt.savefig(f"{args.plot_out}_success_curve.png", dpi=300); plt.close()

        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(success_thresh, success_curve, linewidth=2.0, label=f"Success (AUC={success_auc:.3f})")
        ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.02)
        poster_axes(ax, "IoU threshold", "Success rate")
        ax.vlines(0.5, 0, op_at_05, linestyles="--", linewidth=1.5)
        annotate_point(ax, 0.5, op_at_05, f"OP@0.5={op_at_05:.2f}")
        ax.legend(loc="lower left", frameon=False)
        fig.tight_layout(); fig.savefig(f"{args.plot_out}_success_curve_poster.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    if len(precision_curve) > 0:
        dp_at_thr = float(np.interp(args.dp_thr, prec_thresh, precision_curve))
        plt.figure(figsize=(6,4))
        plt.plot(prec_thresh, precision_curve, lw=2.0, label=f"Precision (AUC@{args.prec_max:.0f}px={precision_auc:.3f})")
        plt.vlines(args.dp_thr, 0, dp_at_thr, linestyles="--"); plt.scatter([args.dp_thr],[dp_at_thr], s=28)
        plt.xlabel("Distance threshold (px)"); plt.ylabel("Precision"); plt.ylim([0,1.02])
        plt.legend(frameon=False); plt.tight_layout()
        plt.savefig(f"{args.plot_out}_precision_curve.png", dpi=300); plt.close()

        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(prec_thresh, precision_curve, linewidth=2.0,
                label=f"Precision (AUC@{args.prec_max:.0f}px={precision_auc:.3f})")
        ax.set_xlim(0, args.prec_max); ax.set_ylim(0, 1.02)
        poster_axes(ax, "Distance threshold (px)", "Precision")
        ax.vlines(args.dp_thr, 0, dp_at_thr, linestyles="--", linewidth=1.5)
        annotate_point(ax, args.dp_thr, dp_at_thr, f"DP@{args.dp_thr:.0f}px={dp_at_thr:.2f}")
        ax.legend(loc="lower right", frameon=False)
        fig.tight_layout(); fig.savefig(f"{args.plot_out}_precision_curve_poster.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # --------- Markdown report ----------
    vid_name = pathlib.Path(args.video).name
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(args.report_md, "w") as f:
        f.write(f"# Tracking Metrics Report\n\n")
        f.write(f"**Video:** `{vid_name}`  \n")
        f.write(f"**Generated:** {ts}\n\n")
        f.write("## Headline Results\n")
        f.write(f"- Mean IoU: {('n/a' if np.isnan(mean_iou) else f'{mean_iou:.3f}')}\n")
        f.write(f"- Final OP (IoU>{args.op_thr}): {('n/a' if np.isnan(final_op) else f'{final_op*100:.1f}%')}\n")
        f.write(f"- Final DP (dist<{args.dp_thr:.0f}px): {('n/a' if np.isnan(final_dp) else f'{final_dp*100:.1f}%')}\n")
        f.write(f"- Success AUC (IoU 0→1): {('n/a' if np.isnan(success_auc) else f'{success_auc:.3f}')}\n")
        f.write(f"- Precision AUC (0→{args.prec_max:.0f}px): {('n/a' if np.isnan(precision_auc) else f'{precision_auc:.3f}')}\n")
        f.write(f"- DP@0.10·min(W,H) [{int(dp10)}px]: {('n/a' if np.isnan(dp10_val) else f'{dp10_val*100:.1f}%')}\n")
        f.write(f"- Mean FPS: {mean_fps:.1f}\n")
        if relock_counts:
            f.write(f"- Recoverability: mean frames to re-lock: {relock_mean:.1f}, max: {relock_max}\n")
        else:
            f.write(f"- Recoverability: no loss events with GT\n")
        f.write("\n## Parameters\n")
        f.write(f"- work_w: {args.work_w}\n- n_pts: {args.n_pts}\n")
        f.write(f"- glare_mode: {args.glare_mode}, clahe: {args.clahe}\n")
        f.write(f"- HSV v_hi/s_lo/v_hi2: {args.v_hi}/{args.s_lo}/{args.v_hi2}\n")
        f.write(f"- OP thr: {args.op_thr}, DP thr: {args.dp_thr:.1f}\n")
        if args.dp_rel and args.dp_rel>0:
            f.write(f"- (used dp_rel={args.dp_rel:.2f} → dp_thr={args.dp_thr:.1f}px)\n")
        f.write(f"- GT JSON: `{args.gt_json}`\n")
        if args.gt_smooth:
            f.write(f"- GT smoothing window: {args.gt_smooth}\n")
        f.write("\n## Diagnostics\n")
        keep_pct = (100.0*fb_kept_total/max(1,fb_total))
        f.write(f"- FB kept ratio: {keep_pct:.1f}%  (N={fb_total})\n")
        f.write(f"- NCC re-locks: {ncc_recovs}\n")
        f.write("\n## Plots\n")
        f.write(f"- OP/DP: `{args.plot_out}_op_dp.png`\n")
        f.write(f"- IoU per-frame: `{args.plot_out}_iou.png`\n")
        f.write(f"- CLE per-frame: `{args.plot_out}_cle.png`\n")
        f.write(f"- FPS per-frame: `{args.plot_out}_fps.png`\n")
        f.write(f"- Success curve: `{args.plot_out}_success_curve.png` and `_poster.png`\n")
        f.write(f"- Precision curve: `{args.plot_out}_precision_curve.png` and `_poster.png`\n")
        f.write(f"- Curves CSV: `{args.plot_out}_curves.csv`\n\n")
        f.write("## Per-frame CSV\n")
        f.write(f"- `{args.save_csv}`\n")

    # Console summary
    print("\n=== Summary ===")
    print(f"Frames with GT                   : {sum(1 for v in iou_series if not np.isnan(v))}")
    print(f"Mean IoU                         : {np.nan if np.isnan(mean_iou) else round(mean_iou,3)}")
    print(f"Final OP (IoU>{args.op_thr})     : {np.nan if np.isnan(final_op) else round(100*final_op,1)}%")
    print(f"Final DP (dist<{args.dp_thr:.0f}px)  : {np.nan if np.isnan(final_dp) else round(100*final_dp,1)}%")
    print(f"DP@0.10·min(W,H) [{int(dp10)}px]     : {('n/a' if np.isnan(dp10_val) else f'{dp10_val*100:.1f}%')}")
    print(f"Mean FPS                         : {round(mean_fps,1)}")
    print(f"Success AUC (IoU 0→1)            : {np.nan if np.isnan(success_auc) else round(success_auc,3)}")
    print(f"Precision AUC (0→{args.prec_max:.0f}px) : {np.nan if np.isnan(precision_auc) else round(precision_auc,3)}")
    if relock_counts:
        print(f"Recoverability: mean frames to re-lock : {round(relock_mean,1)}")
        print(f"Recoverability: max frames to re-lock  : {relock_max}")
    else:
        print("Recoverability                   : no loss events with GT")
    keep_pct = (100.0*fb_kept_total/max(1,fb_total))
    print(f"FB kept ratio                    : {keep_pct:.1f}%  (N={fb_total})")
    print(f"NCC re-locks                     : {ncc_recovs}")
    print(f"\nSaved: {args.save_csv}, {args.plot_out}_*.png, {args.plot_out}_curves.csv, {args.report_md}")

if __name__ == "__main__":
    main()
