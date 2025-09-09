import cv2
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet34_Weights
from torchvision import transforms as T

from .config import LEG_MIN_H, LEG_MIN_W

# ImageNet normalization for ResNet
_leg_tf = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

def _to_batch_rgb_tensor(bgr_np, device):
    rgb = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB)
    ten = _leg_tf(rgb).unsqueeze(0).to(device)  # [1,3,224,224]
    return ten

@torch.inference_mode()
def legibility_score(roi_bgr, leg_model, device):
    if roi_bgr is None or roi_bgr.size == 0:
        return 0.0
    h, w = roi_bgr.shape[:2]
    if h < LEG_MIN_H or w < LEG_MIN_W:
        return 0.0
    x = _to_batch_rgb_tensor(roi_bgr, device)
    # forward() already returns sigmoid probability
    prob = float(leg_model(x).item())
    return prob

def is_legible(roi_bgr, leg_model, device, thr):
    p = legibility_score(roi_bgr, leg_model, device)
    return (p >= thr), p

# === ResNet34 for legibility classification ===
class LegibilityClassifier34(nn.Module):
    def __init__(self, ckpt_path=None, finetune=False, init_from_imagenet=False, device="cpu"):
        super().__init__()
        base_weights = ResNet34_Weights.IMAGENET1K_V1 if init_from_imagenet else None
        self.model_ft = models.resnet34(weights=base_weights)

        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, 1)

        if finetune:
            for p in self.model_ft.parameters():
                p.requires_grad = False
            for p in self.model_ft.layer4.parameters():
                p.requires_grad = True
            for p in self.model_ft.fc.parameters():
                p.requires_grad = True

        if ckpt_path:
            state = torch.load(ckpt_path, map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            state = {k.replace("module.", ""): v for k, v in state.items()}
            try:
                self.load_state_dict(state, strict=True)
            except RuntimeError:
                self.load_state_dict(state, strict=False)

    def forward(self, x):
        x = self.model_ft(x)
        return torch.sigmoid(x)
