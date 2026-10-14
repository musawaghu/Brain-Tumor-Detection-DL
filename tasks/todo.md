# Portfolio upgrade — todo

## Code (done by Claude)
- [x] main.py: relative paths (`data/`, `brain_tumor_labels.csv`)
- [x] main.py: stratified 70/15/15 train/val/test split (`split_data`), saves `test_split.csv`
- [x] main.py: load best-validation checkpoint, evaluate once on test set
- [x] main.py: bootstrap 95% confidence intervals (`bootstrap_ci`)
- [x] main.py: remove `verbose=True` from ReduceLROnPlateau (crashes on current PyTorch)
- [x] gradcam.py: hand-written Grad-CAM on layer4 + overlay
- [x] app.py: Gradio demo (prediction, confidence, heatmap, examples, disclaimer)
- [x] requirements.txt, .gitignore (space_README.md later removed: no hosted demo)
- [x] tests/test_project.py (split, Grad-CAM, app predict)
- [x] README.md rewrite
- [x] main.py: select final model by lowest validation loss
- [x] main.py: duplicate-aware split (`duplicate_groups`) + test

## You
- [x] Download PMRAM (doi 10.17632/m7w55sw88b.1), put glioma + normal images in `data/`
- [x] `python main.py`: train, note the test metrics + CIs printed at the end
- [x] Copy 3 glioma + 3 normal images listed in `test_split.csv` into `samples/`
- [x] Fill the results table in README.md, check `pytest` passes (app test runs now)
- [x] Screenshots `demo_screenshot.png`, `demo_screenshot_normal.png`
- [x] Decided against hosting (portfolio shows screenshots); weights published as GitHub release v1.0
- [x] README: "Try the demo" with weights download, screenshots

## Review
- 2026-10-14 smoke test (synthetic images, 1 epoch, throwaway copy): split → train → best-val checkpoint → test
  eval with CIs → checkpoint loads with `weights_only=True` → app predict + Grad-CAM all work. The heatmap
  lands on the synthetic lesion (mean 0.65 inside vs 0.11 overall). `pytest`: 5/5 pass with weights, 4 pass + 1 skip without.
- 2026-10-14 real-data runs (same seed, same 15 epochs):
  1. Original best-val-accuracy selection: kept epoch 1 (ties never replaced it), test acc 94.0%.
  2. Lowest-val-loss selection: epoch 14, test acc 95.7%. Found exact duplicates crossing train/test
     (769 images = 681 unique scans); removing duplicated test images cost ~1-1.5 points.
  3. Duplicate-aware split (hash <= 5 bits AND 64px mean diff <= 4; hash alone chained different scans
     into a 71-image mixed-label group): epoch 2, test acc 95.7% [91.3, 99.1], AUC 0.989,
     glioma recall 91.1% (5/56 missed). Final, reported in README. No changes made after seeing it.
- Grad-CAM on real test scans: mostly on/near tumors; some off-target; corner artifacts on 2 normals.
- Disclosed in README: same-patient adjacent slices can still cross splits; T1 (glioma) vs T2/FLAIR
  (normal) sequence imbalance is a possible shortcut.
- Open notes: the PDF mentions early stopping but the code has none; `plt.show()` blocks until windows are closed.
