"""Single-image comparison of the production system and CNN."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from expert_system import recognize
from model import AnimalLiteNet


def cnn_predict(image_path, weights):
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    model = AnimalLiteNet(len(classes)); model.load_state_dict(checkpoint["model"]); model.eval()
    tf = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),
        transforms.Normalize((0.49, 0.48, 0.45), (0.24, 0.24, 0.26))])
    with torch.no_grad():
        prob = model(tf(Image.open(image_path).convert("RGB")).unsqueeze(0)).softmax(1)[0]
    return {"prediction": classes[int(prob.argmax())],
            "probabilities": {c: round(float(p), 6) for c, p in zip(classes, prob)}}


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("image"); p.add_argument("--weights")
    args = p.parse_args()
    output = {"production_system": recognize(args.image)}
    if args.weights:
        output["cnn"] = cnn_predict(args.image, args.weights)
    else:
        output["cnn"] = {"status": "not_run", "reason": "provide --weights after training"}
    print(json.dumps(output, ensure_ascii=False, indent=2))

