#!/usr/bin/env bash
# Full-scale feature pipeline. Designed to run unattended in tmux.
#
#   cd /home/ngocvo/Desktop/ngocvo/Dynamic_Fusion/Dataset/fullscale_features
#   tmux new -s feats -d 'bash run_pipeline.sh'
#   tmux attach -t feats     # watch;  Ctrl-b then d to detach
#
# Safe to interrupt at any time, including a hard power-off: stage 1 writes each
# chunk atomically and skips chunks already on disk, so re-running this script
# resumes from the last completed chunk (worst case ~13 min of work lost).
#
# EVERYTHING lives inside main(), called on the last line. That is deliberate:
# bash reads a script incrementally by byte offset, so editing this file while it
# is running makes bash resume at a stale offset and execute garbage. Wrapping
# the body in a function forces bash to parse it all up front, making the running
# job immune to later edits.

set -u

main() {
    cd "$(dirname "$0")" || exit 1
    local STAMP LOG
    STAMP=$(date +%Y%m%d_%H%M%S)
    LOG="pipeline_${STAMP}.log"

    log() { echo "$*" | tee -a "$LOG"; }

    log "=== full-scale feature pipeline — started $(date) ==="
    log "    REDUCED set: 23 features (katz / closeness / eigenvector dropped)"
    log "    all 2,973,489 nodes; TOP10_FEATURE_NAMES unchanged — none of the"
    log "    three dropped centralities is in it, so model input is unaffected."
    log ""

    # ── clear write-in-progress files from an interrupted run / power-off ────
    # A leftover temp is by definition incomplete (chunks are renamed into place
    # atomically), so deleting it is always safe: that chunk is recomputed below.
    local n_stale
    n_stale=$(find group3_chunks -maxdepth 1 \( -name '.tmp_*' -o -name '*.tmp.npy' \) 2>/dev/null | wc -l)
    if [ "${n_stale:-0}" -gt 0 ]; then
        log "[resume] clearing $n_stale incomplete chunk file(s) from a previous interruption"
        find group3_chunks -maxdepth 1 \( -name '.tmp_*' -o -name '*.tmp.npy' \) -delete
    fi

    python3 - <<'PY' | tee -a "$LOG"
import numpy as np, glob
fs = [f for f in sorted(glob.glob('group3_chunks/schunk_*.npy')) if '.tmp.' not in f]
red = [f for f in fs if np.isnan(np.load(f)[:, 0]).all()]
print(f"[inventory] {len(fs)} chunks banked ({len(fs)*1500:,} nodes): "
      f"{len(fs)-len(red)} full-set, {len(red)} reduced — both reusable")
PY

    run() {
        log ""
        log "─── $* ───"
        log ""
        "$@" 2>&1 | tee -a "$LOG"
        local rc=${PIPESTATUS[0]}
        if [ "$rc" -ne 0 ]; then
            log "[✗] FAILED (rc=$rc): $*"
            return "$rc"
        fi
        return 0
    }

    # ── stage 1: group-3 centralities, reduced set, resumable ───────────────
    # Retried, because this is the only stage that can die for a transient
    # reason (a hub-ball OOM in one worker) and every retry starts from the
    # chunks already banked -- so a retry is cheap and never redoes work.
    local attempt rc=1
    for attempt in 1 2 3; do
        log ""
        log "########## stage 1/4 — group-3 centralities (attempt $attempt/3) ##########"
        if run python3 -u 05_group3_centrality.py --workers 16 --chunk 1500 --reduced; then
            rc=0
            break
        fi
        rc=$?
        log "[!] stage 1 attempt $attempt failed (rc=$rc); banked chunks are kept, retrying in 60s"
        sleep 60
    done
    if [ "$rc" -ne 0 ]; then
        log "[✗] stage 1 failed 3 times — stopping before building a partial table."
        log "    Re-run this script to resume; nothing computed so far is lost."
        exit "$rc"
    fi

    log ""
    log "########## stage 2/4 — assemble final CSV ##########"
    run python3 -u 08_assemble_csv.py || exit $?

    log ""
    log "########## stage 3/4 — end-to-end spot-check vs the ORIGINAL script ##########"
    # Non-fatal: a failure here must not discard a completed 23h extraction. It is
    # reported loudly and the verdict is preserved in the summary below.
    if run python3 -u 09_final_spotcheck.py 20; then
        SPOTCHECK="PASS"
    else
        SPOTCHECK="FAIL — see log"
        log "[!] spot-check did not pass; the CSV is still on disk, but read the log before trusting it."
    fi

    log ""
    log "########## stage 4/4 — diagnostic feature ranking (does NOT change top-10) ##########"
    if run python3 -u 11_rerank_features.py; then RANK="ok"; else RANK="failed — see log"; fi

    # ── durable summary, so the result is readable without scrolling the log ──
    {
        echo "# Full-scale feature extraction — run summary"
        echo
        echo "- finished:            $(date)"
        echo "- log:                 $LOG"
        echo "- mode:                REDUCED (23 features; katz/closeness/eigenvector dropped)"
        echo "- top-10 features:     UNCHANGED (none of the dropped three is in it)"
        echo "- spot-check vs original: $SPOTCHECK"
        echo "- diagnostic ranking:  $RANK"
        echo
        echo "## Outputs"
        echo
        for f in ../../raw_data/MulDiGraph/features_output_fullscale.csv \
                 ../../raw_data/MulDiGraph/features_output_top10_MG_fullscale.csv \
                 feature_ranking_fullscale.csv; do
            if [ -f "$f" ]; then
                echo "- \`$(basename "$f")\` — $(du -h "$f" | cut -f1), $(( $(wc -l < "$f") - 1 )) rows"
            else
                echo "- \`$(basename "$f")\` — MISSING"
            fi
        done
        echo
        echo "## Next step"
        echo
        echo "Point mg_refit_features.py at features_output_fullscale.csv"
        echo "(it currently hardcodes the 5,655-row features_output_split.csv),"
        echo "or use features_output_top10_MG_fullscale.csv directly — it is already"
        echo "StandardScaler-normalised with mean/std fit on the 'train' partition only."
    } > RUN_SUMMARY.md

    log ""
    log "=== pipeline finished $(date) ==="
    log "    summary written to RUN_SUMMARY.md"
    cat RUN_SUMMARY.md | tee -a "$LOG"
}

main "$@"
