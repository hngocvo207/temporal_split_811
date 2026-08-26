# Full-scale feature extraction — run summary

- finished:            Mon Aug 10 10:19:01 PM +07 2026
- log:                 pipeline_20260810_000935.log
- mode:                REDUCED (23 features; katz/closeness/eigenvector dropped)
- top-10 features:     UNCHANGED (none of the dropped three is in it)
- spot-check vs original: PASS
- diagnostic ranking:  ok (RE-RUN after a label fix, see below)

## CORRECTION applied after the unattended run

The first assembly labelled rows from `phisher_accounts.txt` (5,480 addresses in
graph), the file `select_add_features.py` uses. The downstream pipeline scores
against `labels.pkl` (graph `isp`, 1,165 positives) instead, and the two sets
overlap on only 963 addresses (only-txt 4,517 | only-isp 202). The built-in
cross-check flagged this as MISMATCH.

Fixed: `label` is now `isp`, and `.txt` membership is kept as `is_phisher_txt`.
Stages 2 and 4 were re-run (no re-extraction needed). Per-partition positives now
reproduce split_stats.json exactly: train 519 / val 117 / overlap 217 / pure_test 312.

Note this affects the ORIGINAL script too: its `label`/`is_phisher` columns -- and
therefore the Spearman selection that produced TOP10_FEATURE_NAMES -- were computed
against the `.txt` label set, not the labels the model is trained on.

## Outputs

- `features_output_fullscale.csv` — 540M, 2973489 rows
- `features_output_top10_MG_fullscale.csv` — 725M, 2973489 rows
- `feature_ranking_fullscale.csv` — 4.0K, 23 rows

## Next step

Point mg_refit_features.py at features_output_fullscale.csv
(it currently hardcodes the 5,655-row features_output_split.csv),
or use features_output_top10_MG_fullscale.csv directly — it is already
StandardScaler-normalised with mean/std fit on the 'train' partition only.
