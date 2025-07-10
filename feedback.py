import io
import base64
import torch
import torch.nn as nn
from openai import OpenAI
import numpy as np
from PIL import Image
from config import HIGH_CONF, LOW_CONF, OPENROUTER_API_KEY
from torchcam.methods import GradCAM

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

def init_cam(model):
    return GradCAM(model=model, target_layer="layer4")

def selective_indices(probs):
    max_probs, _ = probs.max(dim=1)
    return ((max_probs > HIGH_CONF) | (max_probs < LOW_CONF)).nonzero(as_tuple=True)[0]

def _freeze_batchnorm(model: nn.Module):
    """Put all BatchNorm modules into eval mode to avoid single-sample errors."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def generate_cams(cam_extractor, model, inputs, class_indices):
    """
    For each input tensor (CxHxW) and its predicted class index,
    run it through the model (with BN frozen) so TorchCAM can hook gradients.
    Returns list of H×W×C numpy arrays in [0,1].
    """
    # Save & prepare model modes
    was_training = model.training
    model.train()
    _freeze_batchnorm(model)

    cams = []
    for img_tensor, cls in zip(inputs, class_indices):
        input_batch = img_tensor.unsqueeze(0)

        out = model(input_batch)
        activation_map = cam_extractor(cls.item(), out)[0]  # shape H×W

        # Normalize & overlay
        hm = activation_map.cpu().detach()
        hm = hm / (hm.max() + 1e-8)

        img_np  = img_tensor.permute(1, 2, 0).cpu().numpy()  # HxWxC
        overlay = (img_np * 0.5 + hm.unsqueeze(-1).numpy() * 0.5).clip(0, 1)
        cams.append(overlay)

    # Restore original model mode
    if not was_training:
        model.eval()

    return cams

def ask_llm(cam_images, labels, confidences):
    """
    Ensures each cam_img becomes a HxWx3 uint8 numpy array,
    then encodes to PNG/base64 and sends via OpenRouter.
    """
    scores = []
    for cam_img, lbl, conf in zip(cam_images, labels, confidences):
        # Convert torch.Tensor → numpy if needed
        if isinstance(cam_img, torch.Tensor):
            cam_np = cam_img.cpu().numpy()
        else:
            cam_np = np.array(cam_img)

        # Squeeze out any leading singleton dims
        cam_np = np.squeeze(cam_np)
        # After squeeze, cam_np could be:
        #  - (H, W, 3)
        #  - (H, W)  -> replicate to RGB
        #  - (3, H, W) -> transpose to (H, W, 3)
        if cam_np.ndim == 2:
            # grayscale map → stack to RGB
            cam_np = np.stack([cam_np]*3, axis=-1)
        elif cam_np.ndim == 3 and cam_np.shape[0] in (1, 3):
            cam_np = np.transpose(cam_np, (1, 2, 0))
        elif cam_np.ndim != 3 or cam_np.shape[-1] not in (1, 3):
            raise ValueError(f"Unexpected CAM shape after squeeze: {cam_np.shape}")

        # Normalize & convert to uint8
        cam_np = (cam_np * 255).clip(0, 255).astype("uint8")

        # Encode to PNG in-memory
        pil_img = Image.fromarray(cam_np)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        prompt = (
            f"Image Class: {lbl}\n"
            f"Model Confidence: {conf:.2f}\n\n"
            "Below is the Grad-CAM overlay showing where the model looked:\n\n"
            f"![cam overlay](data:image/png;base64,{b64})\n\n"
            "On a scale from 0 (completely wrong region) to 1 (perfect focus), "
            "what score would you give this highlighted region for correctly "
            f"classifying this as “{lbl}”? Just reply with: “Score: <number>”."
        )
        resp = client.chat.completions.create(
            model="google/gemma-3-4b-it",
            messages=[{
                "role": "user",
                "content": [
                    {"type":"text",  "text": prompt},
                ]
            }]
        )
        txt = resp.choices[0].message.content
        #print(txt, conf)
        score = float(txt.split("Score:")[1].split()[0])
        scores.append(score)

    return torch.tensor(scores)
