# Portfolio upgrade — todo

## Code (done by Claude)
- [x] main.py: relative paths (`data/`, `brain_tumor_labels.csv`)
- [x] main.py: stratified 70/15/15 train/val/test split (`split_data`), saves `test_split.csv`
- [x] main.py: load best-validation checkpoint, evaluate once on test set
- [x] main.py: bootstrap 95% confidence intervals (`bootstrap_ci`)
- [x] main.py: remove `verbose=True` from ReduceLROnPlateau (crashes on current PyTorch)
- [x] gradcam.py: hand-written Grad-CAM on layer4 + overlay
- [x] app.py: Gradio demo (prediction, confidence, heatmap, examples, disclaimer)
- [x] requirements.txt, .gitignore, space_README.md
- [x] tests/test_project.py (split, Grad-CAM, app predict)
- [x] README.md rewrite (results placeholders until training is run)

## You
- [ ] Download PMRAM (doi 10.17632/m7w55sw88b.1), put glioma + normal images in `data/`
- [ ] `python main.py`: train, note the test metrics + CIs printed at the end
- [ ] Copy 3 glioma + 3 normal images listed in `test_split.csv` into `samples/`
- [ ] Fill the results table in README.md, check `pytest` passes (app test runs now)
- [ ] `python app.py`: try it locally, save a screenshot as `demo_screenshot.png`
- [ ] Create the Hugging Face Space and upload (commands in README)
- [ ] Put the Space URL in README.md

## Review
- 2026-10-14 smoke test (synthetic images, 1 epoch, throwaway copy): split → train → best-val checkpoint → test
  eval with CIs → checkpoint loads with `weights_only=True` → app predict + Grad-CAM all work. The heatmap
  lands on the synthetic lesion (mean 0.65 inside vs 0.11 overall). `pytest`: 5/5 pass with weights, 4 pass + 1 skip without.
- Real-data results still pending (dataset not downloaded yet).
- Open notes: the PDF mentions early stopping but the code has none; `plt.show()` blocks until windows are closed.
