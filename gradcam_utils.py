import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class GradCAMModelWrapper(nn.Module):
    """
    Wrap Hugging Face image classification models so Grad-CAM receives logits tensor,
    not ImageClassifierOutputWithNoAttention.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        return outputs.logits


def get_target_layer(model):
    """
    Robust Grad-CAM layer selector for HuggingFace + custom MobileNet models.
    """

    core = model
    for attr in ["model", "base_model", "backbone", "mobilenet_v2"]:
        if hasattr(core, attr):
            core = getattr(core, attr)

    if hasattr(core, "features") and len(core.features) > 0:
        return core.features[-1]

    if hasattr(core, "conv_head"):
        return core.conv_head

    for module in reversed(list(core.modules())):
        if isinstance(module, torch.nn.Conv2d):
            return module

    raise ValueError("No suitable target layer found for Grad-CAM")


def generate_gradcam(model, processor, pil_img):
    model.eval()

    inputs = processor(images=pil_img, return_tensors="pt")
    input_tensor = inputs["pixel_values"]

    

    wrapped_model = GradCAMModelWrapper(model)
    wrapped_model.eval()
    target_layer = get_target_layer(wrapped_model.model)

    with GradCAM(model=wrapped_model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor)[0]

    img = pil_img.resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0

    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    return Image.fromarray(visualization)