#!/usr/bin/env bash
set -euo pipefail

# ------------------------- USER CONFIG -------------------------
EASY_NAME="part15"
EASY_VIDEO="videos/part15.mp4"
EASY_GT="labels/part15umbo1.json"

PY=python3
KLT=./klt.py
RUNS_DIR="./runs"

# Repeats
REPEATS_KLT=1

# KLT parameter ablation (added 75 pts)
KLT_NPTS=(75 100 150 250)
KLT_RANSAC=(2.0 3.0 5.0)
KLT_RES_LIST=(200 720 1080)

# ----------------------- INTERNAL HELPERS ----------------------

mkdir -p "$RUNS_DIR"

clamp_0_255 () {
  local x=$1
  (( x < 0 )) && echo 0 && return
  (( x > 255 )) && echo 255 && return
  echo "$x"
}

write_meta_json () {
  local out="$1/meta.json"; shift
  {
    echo "{"
    local first=1
    for kv in "$@"; do
      key="${kv%%=*}"; val="${kv#*=}"
      [[ $first -eq 1 ]] || echo ","
      first=0
      if [[ "$val" =~ ^-?[0-9]+(\.[0-9]+)?$ || "$val" =~ ^(true|false|null)$ ]]; then
        printf '  "%s": %s' "$key" "$val"
      else
        printf '  "%s": "%s"' "$key" "$val"
      fi
    done
    echo
    echo "}"
  } > "$out"
}

run_one () {
  local tracker="$1" seq="$2" video="$3" gt="$4" work_w="$5" glare="$6" delta="$7" rep="$8" n="$9" rr="${10}"; shift 10
  local extra_args=("$@")

  local base_v_hi=240 base_s_lo=40 base_v_hi2=220
  local v_hi=$(( base_v_hi + delta ))
  local s_lo=$(( base_s_lo - delta ))
  local v_hi2=$(( base_v_hi2 + delta ))
  v_hi=$(clamp_0_255 "$v_hi"); s_lo=$(clamp_0_255 "$s_lo"); v_hi2=$(clamp_0_255 "$v_hi2")

  local script="$KLT"
  rr_sanitized=$(echo "$rr" | sed 's/\./p/g')
  local outdir="${RUNS_DIR}/klt_ablate/${tracker}/${seq}/w${work_w}/glare_${glare}/d${delta}/n${n}_r${rr_sanitized}/run${rep}"
  mkdir -p "$outdir"

  local common=(
    --video "$video"
    --gt_json "$gt"
    --work_w "$work_w"
    --glare_mode "$glare"
    --v_hi "$v_hi" --s_lo "$s_lo" --v_hi2 "$v_hi2"
    --plot_out "$outdir/plots"
    --report_md "$outdir/report.md"
    --save_csv "$outdir/per_frame.csv"
  )

  local cmd=( "$PY" "$script" "${common[@]}" )
  if ((${#extra_args[@]})); then
    cmd+=("${extra_args[@]}")
  fi

  write_meta_json "$outdir" \
    tracker="$tracker" seq="$seq" exp="klt_ablate" run="$rep" \
    work_w="$work_w" glare_mode="$glare" v_hi="$v_hi" s_lo="$s_lo" v_hi2="$v_hi2" \
    n_pts="$n" ransac="$rr"

  echo "[RUN] ${cmd[*]}"
  "${cmd[@]}" 2>&1 | tee "$outdir/run.log"
}

# --- KLT PARAM ABLATION ---
run_klt_ablate () {
  local seq="$EASY_NAME" video="$EASY_VIDEO" gt="$EASY_GT"
  local glare="mask" delta=0
  for w in "${KLT_RES_LIST[@]}"; do
    for n in "${KLT_NPTS[@]}"; do
      for rr in "${KLT_RANSAC[@]}"; do
        for r in $(seq 1 "$REPEATS_KLT"); do
          run_one "klt" "$seq" "$video" "$gt" "$w" "$glare" "$delta" "$r" "$n" "$rr" \
            --n_pts "$n" --ransac_reproj "$rr"
        done
      done
    done
  done
}

# ---------------------- MASTER ----------------------
echo ">>> Starting ADDITIONAL KLT parameter ablations (200p, 720p, 1080p; n_pts incl. 75)..."
run_klt_ablate
echo ">>> Done"
