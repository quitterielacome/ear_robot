#!/usr/bin/env python3
"""
Template Matching (TM) baseline with shared preprocessing/metrics/HUD
- GT-initialised template from frame 0; NCC in a padded search window
- Optional template update after accepted matches (--update_tpl)
- Glare ablation + CLAHE flags identical to klt_full.py / new_sift.py
- HUD with IoU & CLE (threshold-colored) + LOST overlay
- Metrics parity: OP/DP, Success/Precision curves (+AUCs), DP@0.10·min(W,H)
- Poster-quality plots + standard plots, curves CSV, markdown report
- CSV header identical to klt_full.py ("inliers" field used as NCC*100)
"""

import cv2, argparse, time, csv, json, pathlib, datetime, os, re
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


def search_patch(gray, tpl, box, pad, method=cv2.TM_CCOEFF_NORMED):
    x,y,w,h=box
    sx,sy=max(x-pad,0),max(y-pad,0)
    ex,ey=min(x+w+pad,gray.shape[1]),min(y+h+pad,gray.shape[0])
    R=gray[sy:ey,sx:ex]
    r=cv2.matchTemplate(R,tpl,method)
    _,v,_,loc=cv2.minMaxLoc(r)
    return v,(loc[0]+sx,loc[1]+sy)

# ---------- main ----------

def main():
    p=argparse.ArgumentParser(description="Template Matching (NCC) baseline")
    p.add_argument("--video", required=True)
    p.add_argument("--gt_json", required=True)
    p.add_argument("--work_w", type=int, default=720)
    p.add_argument("--pad", type=int, default=35)
    p.add_argument("--thresh", type=float, default=0.70)
    p.add_argument("--method", choices=["ccoeff_normed","ccorr_normed","sqdiff"], default="ccoeff_normed")
    p.add_argument("--update_tpl", action="store_true", help="refresh template after accepted match")
    p.add_argument("--op_thr", type=float, default=0.5)
    p.add_argument("--dp_thr", type=float, default=20.0)
    p.add_argument("--dp_rel", type=float, default=0.0, help="if >0, dp_thr = dp_rel*min(H,W)")
    p.add_argument("--glare_mode", choices=["off","mask","inpaint"], default="mask")
    p.add_argument("--v_hi",  type=int, default=240)
    p.add_argument("--s_lo",  type=int, default=40)
    p.add_argument("--v_hi2", type=int, default=220)
    p.add_argument("--clahe", action="store_true")
    p.add_argument("--gt_smooth", type=int, default=0)
    p.add_argument("--plot_out", type=str, default="metrics")
    p.add_argument("--save_csv", default="tm_track_metrics.csv")
    p.add_argument("--report_md", default="metrics_report.md")
    p.add_argument("--success_steps", type=int, default=101)
    p.add_argument("--prec_max", type=float, default=50.0)
    p.add_argument("--prec_step", type=float, default=1.0)
    p.add_argument("--pts", type=int, default=20,
               help="# of dots along center→predicted center")

    args=p.parse_args()

    cv2.setUseOptimized(True)
    cv2.setRNGSeed(1337)

    method_map = {
        "ccoeff_normed": cv2.TM_CCOEFF_NORMED,
        "ccorr_normed":  cv2.TM_CCORR_NORMED,
        "sqdiff":        cv2.TM_SQDIFF_NORMED,
    }
    tm_method = method_map[args.method]

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

    # initial ROI from GT (native & work space)
    gx,gy,gw,gh = map(int, gt[min(gt.keys())])
    gxw, gyw = int(round(gx*scale)), int(round(gy*scale))
    gww, ghw = max(2, int(round(gw*scale))), max(2, int(round(gh*scale)))
    tpl = gray0i[gyw:gyw+ghw, gxw:gxw+gww].copy()

    rows=[]; iou_series=[]; cle_series=[]; fps_series=[]; relock_counts=[]; bad_streak=0
    tprev=time.time()

    cv2.namedWindow("Track", cv2.WINDOW_NORMAL)

    # current predicted box (native)
    pred_box = (gx,gy,gw,gh)

    while True:
        ok, fr = cap.read()
        if not ok: break
        frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        work = cv2.resize(fr, (wW,wH), cv2.INTER_AREA) if scale<1.0 else fr
        gi, gmask = preprocess_gray(work, args.glare_mode, args.v_hi, args.s_lo, args.v_hi2, clahe=args.clahe)

        # search around last prediction (or GT on frame 0)
        if pred_box is None:
            # fall back to GT window in work space if missing
            cx, cy, cw, ch = gxw, gyw, gww, ghw
        else:
            px,py,pw,ph = pred_box
            cx, cy = int(round(px*scale)), int(round(py*scale))
            cw, ch = max(2,int(round(pw*scale))), max(2,int(round(ph*scale)))

        score,(nx,ny) = search_patch(gi, tpl, (cx,cy,cw,ch), args.pad, tm_method)
        if tm_method == cv2.TM_SQDIFF_NORMED:
            # for SQDIFF, lower is better; convert to similarity
            score = 1.0 - score

        # accept/reject
        lost = 0
        if score >= args.thresh:
            # update pred box (native)
            tx_w, ty_w = nx, ny
            tx, ty = int(round(tx_w/scale)), int(round(ty_w/scale))
            pred_box = (tx, ty, gw, gh)
            if args.update_tpl:
                tpl = gi[ny:ny+ch, nx:nx+cw].copy()
        else:
            lost = 1
            pred_box = None

        # FPS
        now=time.time(); inst_fps = 1.0/max(1e-9, now - tprev); tprev=now
        fps_series.append(inst_fps)

        # Metrics
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

        # draw
        if pred_box:
            px,py,pw,ph = pred_box
            cv2.rectangle(fr,(px,py),(px+pw,py+ph),(0,255,0),2)

            # --- dotted trajectory center→predicted center ---
            tcx = px + pw/2.0
            tcy = py + ph/2.0
            traj = interpolate((int(n0[0]), int(n0[1])), (int(tcx), int(tcy)), args.pts)
            for qx, qy in traj:
                cv2.circle(fr, (qx, qy), 3, (0,255,0), -1)
            cv2.circle(fr, (int(n0[0]), int(n0[1])), 5, (0,0,255), -1)

        if gt_box:
            gx1,gy1,gw1,gh1 = map(int, gt_box)
            cv2.rectangle(fr,(gx1,gy1),(gx1+gw1,gy1+gh1),(255,0,0),2)
        if lost:
            cv2.putText(fr,"LOST",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)

        cv2.putText(fr,f"TM(NCC)  FPS:{inst_fps:5.1f}  score:{score:0.2f}", (10,60),
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

        # CSV row; use NCC*100 as "inliers" surrogate
        rows.append([
            frame_no, int(round(score*100)),
            pred_box[0] if pred_box else "", pred_box[1] if pred_box else "",
            gt_box[0] if gt_box else "", gt_box[1] if gt_box else "",
            gt_box[2] if gt_box else "", gt_box[3] if gt_box else "",
            iou_v if iou_v is not None else "", cle_v if cle_v is not None else "",
            inst_fps, int(lost)
        ])

    cap.release(); cv2.destroyAllWindows()
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
        f.write(f"- pad: {args.pad}, NCC thresh: {args.thresh}, method: {args.method}, update_tpl: {args.update_tpl}\n")
        f.write(f"- glare_mode: {args.glare_mode}, clahe: {args.clahe}\n")
        f.write(f"- HSV v_hi/s_lo/v_hi2: {args.v_hi}/{args.s_lo}/{args.v_hi2}\n")
        f.write(f"- OP thr: {args.op_thr}, DP thr: {args.dp_thr}\n")
        f.write(f"- GT JSON: `{args.gt_json}`\n")
        # Recoverability (write while file is open)
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
    # Console recoverability (print, not write)
    if relock_counts:
        print(f"Recoverability: mean frames to re-lock : {relock_mean:.1f}")
        print(f"Recoverability: max frames to re-lock  : {relock_max}")
    else:
        print("Recoverability                   : no loss events with GT")

    print(f"\nSaved: {args.save_csv}, {args.plot_out}_*.png, {args.plot_out}_curves.csv, {args.report_md}")

if __name__=="__main__":
    main()
