import torch
import torch.nn.functional as F

from .model import longclip


class ModelOutput:
    def __init__(self, image_embeds=None, text_embeds=None):
        self.image_embeds = image_embeds
        self.text_embeds = text_embeds


class LongCLIPWrapper:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        print(f"Loading LongCLIP model from: {model_path}")
        try:
            self.model, self.preprocess = longclip.load(model_path, device=self.device)
            self.model.eval()
            print(f"LongCLIP model loaded on {self.device.upper()}.")
        except Exception as exc:
            raise IOError(f"Failed to load LongCLIP model: {exc}")

    def __call__(self, text, images, **kwargs):
        with torch.no_grad():
            image_tensor = self.preprocess(images).unsqueeze(0).to(self.device)
            text_tensor = longclip.tokenize(text).to(self.device)

            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tensor)

            image_embeds = F.normalize(image_features, p=2, dim=-1)
            text_embeds = F.normalize(text_features, p=2, dim=-1)

        return ModelOutput(image_embeds=image_embeds, text_embeds=text_embeds)
