# Brain-Tumor-Detection-DL
This project is a deep learning-based binary image classification model that detects the presence of brain tumors in MRI scans. Leveraging convolutional neural networks (CNNs) and data augmentation techniques, the model distinguishes between tumor (e.g., glioma) and normal brain images. The dataset is manually labeled and preprocessed for optimal performance. This tool is designed to support medical diagnostics by offering fast, automated, and accurate tumor detection.

> **Research demo, not a medical device.** This model is a learning project and must not be used for diagnosis.

**Live demo:** _coming soon (Hugging Face Space link)_. Upload an MRI or click a sample scan to get a prediction, a confidence score and a Grad-CAM heatmap.

<!-- After running app.py, save a screenshot as demo_screenshot.png and uncomment:
![Demo](demo_screenshot.png)
-->

## How it works
- **Model:** ResNet50 pretrained on ImageNet. Only `layer4` and a new `Dropout + Linear` head are trained (transfer learning).
- **Augmentation:** random horizontal flip, rotation (±10°), brightness/contrast jitter.
- **Training:** Adam (lr 1e-3, weight decay 1e-4), ReduceLROnPlateau, 15 epochs, batch size 32.
- **Evaluation:** stratified **70 / 15 / 15** train / validation / test split. The validation set picks the best epoch. The test set is used **once**, at the end, so its score isn't tuned to the data.
- **Confidence intervals:** test metrics are reported with bootstrap 95% CIs (1,000 resamples). With ~116 test images, a single accuracy number can easily move a few percent, and the interval shows how far.
- **Explainability:** Grad-CAM on the last convolutional block (`layer4`), in `gradcam.py`.

## Results (held-out test set)

| Metric | Score | 95% CI |
|---|---|---|
| Accuracy | _TBD_ | _TBD_ |
| Recall (macro) | _TBD_ | _TBD_ |
| Precision (macro) | _TBD_ | _TBD_ |
| F1 (macro) | _TBD_ | _TBD_ |
| ROC-AUC | _TBD_ | _TBD_ |

![Confusion matrix](confusion_matrix.png)
![Training history](training_history.png)

> The earlier write-up (`Brain Tumor Detection Using Deep Learning.pdf`) reports 98% accuracy and recall. That number was measured on the validation set that was also used to pick the best model, so it is optimistic. The table above comes from a separate test set and supersedes it.

## About the Grad-CAM heatmap
Grad-CAM highlights the regions that most influenced the model's decision: **where the model is looking**. It is not a tumor outline or a segmentation. On glioma scans it usually lands on the tumor, which suggests the model learned something meaningful. Getting real tumor boundaries would need a segmentation model trained on pixel-level masks.

## Dataset
[PMRAM: Bangladeshi Brain Cancer - MRI Dataset](https://data.mendeley.com/datasets/m7w55sw88b/1) (Mendeley Data, doi:10.17632/m7w55sw88b.1), licensed CC BY 4.0. This project uses only the **glioma** and **normal (no tumor)** classes: 769 images, listed with labels in `brain_tumor_labels.csv` (1 = glioma, 0 = normal).

1. Download the dataset from Mendeley.
2. Copy the **raw** (non-augmented) glioma and normal images into `./data/` (subfolders are fine; images are found by filename).
   Don't use the dataset's pre-augmented images. Augmented copies of the same scan would end up in both train and test and inflate the score.

## Setup and usage
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt  # for a GPU build of PyTorch, see pytorch.org/get-started

python main.py   # trains, prints test metrics + 95% CIs, saves final_brain_mri_model.pth and plots
python app.py    # starts the Gradio demo at http://127.0.0.1:7860
pytest           # runs the tests (the app test runs once final_brain_mri_model.pth exists)
```

### Sample scans for the demo
The demo's sample images must come from the **test** split, so visitors only see scans the model never trained on. After training:
```bash
python -c "import pandas as pd, shutil, glob; t = pd.read_csv('test_split.csv'); [shutil.copy(glob.glob(f'data/**/{f}', recursive=True)[0], 'samples/') for f in pd.concat([t[t.label == 1].head(3), t[t.label == 0].head(3)]).filename]"
```
(Create the `samples/` folder first.)

### Deploying to Hugging Face Spaces
1. Create a new **Gradio** Space at huggingface.co/new-space.
2. Log in and upload (the weights file is uploaded with LFS automatically):
```bash
pip install -U huggingface_hub
hf auth login
hf upload <hf-username>/<space-name> space_README.md README.md --repo-type=space
hf upload <hf-username>/<space-name> . . --repo-type=space --include "app.py" "gradcam.py" "requirements.txt" "final_brain_mri_model.pth" "samples/*"
```

## Limitations
- **Single dataset, single source.** Performance on scans from other hospitals or scanners is unknown.
- **No patient IDs.** The dataset doesn't say which slices come from the same patient, so slices from one patient may appear in both train and test. That would make the test score optimistic.
- **Binary only.** Glioma vs normal; meningioma, pituitary tumors and tumor grade aren't handled.
- **Grad-CAM is not segmentation.** See above.

## Project structure
```
main.py                   training + evaluation (split, bootstrap CIs, plots)
gradcam.py                Grad-CAM heatmap + overlay
app.py                    Gradio web demo
tests/test_project.py     pytest tests
brain_tumor_labels.csv    filename -> label (1 = glioma, 0 = normal)
space_README.md           README (with config header) for the Hugging Face Space
```
