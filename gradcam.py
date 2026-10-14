import numpy as np
import torch.nn.functional as F
import matplotlib
from PIL import Image


# Grad-CAM (Selvaraju et al., 2017): shows which regions of the image the model relied on
# for its prediction. It is NOT a tumor segmentation, just "where the model is looking".
def grad_cam(model, img_tensor, target_class):
    """
    Args:
        model: ResNet model (we hook its last conv block, layer4)
        img_tensor: preprocessed image, shape (1, 3, H, W)
        target_class: index of the class to explain
    Returns:
        numpy array (H, W) with values in [0, 1]
    """
    store = {}

    # Save layer4's output (feature maps) and, during backward, its gradient
    def forward_hook(module, inputs, output):
        store['activations'] = output
        output.register_hook(lambda grad: store.__setitem__('gradients', grad))

    handle = model.layer4.register_forward_hook(forward_hook)
    try:
        model.eval()
        model.zero_grad()
        outputs = model(img_tensor)
        outputs[0, target_class].backward()
    finally:
        handle.remove()  # always detach the hook so the model is left unchanged

    activations = store['activations'].detach()[0]  # (C, h, w)
    gradients = store['gradients'].detach()[0]      # (C, h, w)

    # Weight each feature map by its average gradient, sum, keep only positive evidence
    weights = gradients.mean(dim=(1, 2))
    cam = F.relu((weights[:, None, None] * activations).sum(dim=0))

    # Upsample from 7x7 to the input size and scale to [0, 1]
    cam = F.interpolate(cam[None, None], size=img_tensor.shape[2:], mode='bilinear', align_corners=False)[0, 0]
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.cpu().numpy()


def overlay(pil_img, cam, alpha=0.4):
    """Blend a jet-colored heatmap over the image (resized to the heatmap size)."""
    img = np.asarray(pil_img.convert('RGB').resize((cam.shape[1], cam.shape[0])), dtype=np.float32) / 255
    heatmap = matplotlib.colormaps['jet'](cam)[..., :3]
    blended = (1 - alpha) * img + alpha * heatmap
    return Image.fromarray(np.uint8(blended * 255))
