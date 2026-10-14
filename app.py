import os
import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models

from gradcam import grad_cam, overlay

MODEL_PATH = "final_brain_mri_model.pth"
SAMPLES_DIR = "samples"
LABELS = ["Normal", "Glioma (tumor)"]  # class 0 and class 1 in brain_tumor_labels.csv

DISCLAIMER = """
**Research demo, not a medical device.** Do not use it for diagnosis.
The heatmap (Grad-CAM) shows **where the model is looking** when it makes its decision,
not the outline of a tumor.
"""

# Same preprocessing as val_transform in main.py
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model():
    # Same architecture as get_model() in main.py, but weights=None: the trained weights
    # come from our checkpoint, so there is no need to download ImageNet weights at startup.
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, len(LABELS)))
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


model = load_model()


def predict(image):
    """Takes a PIL image, returns class probabilities and a Grad-CAM overlay."""
    if image is None:
        raise gr.Error("Please upload an MRI image.")
    image = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(img_tensor), dim=1)[0]
    pred_idx = int(probs.argmax())

    # Explain the predicted class (needs gradients, so outside no_grad)
    cam = grad_cam(model, img_tensor, pred_idx)

    confidences = {LABELS[i]: float(p) for i, p in enumerate(probs)}
    return confidences, overlay(image, cam)


examples = sorted(os.path.join(SAMPLES_DIR, f) for f in os.listdir(SAMPLES_DIR)) if os.path.isdir(SAMPLES_DIR) else []

with gr.Blocks(title="Brain Tumor Detection") as demo:
    gr.Markdown("# Brain Tumor Detection (glioma vs normal MRI)")
    gr.Markdown(DISCLAIMER)
    with gr.Row():
        input_image = gr.Image(type="pil", label="MRI scan", image_mode="RGB")
        with gr.Column():
            label_output = gr.Label(label="Prediction")
            cam_output = gr.Image(label="Grad-CAM: where the model is looking")
    submit = gr.Button("Analyze", variant="primary")
    submit.click(predict, inputs=input_image, outputs=[label_output, cam_output])
    if examples:
        gr.Examples(examples=examples, inputs=input_image, outputs=[label_output, cam_output],
                    fn=predict, cache_examples=False, label="Sample scans from the held-out test set")

if __name__ == "__main__":
    # queue() limits concurrent requests; max_file_size rejects oversized uploads
    demo.queue(max_size=20).launch(max_file_size="10mb")
