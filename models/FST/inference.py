import os
import time

import torch
from modeling import loss_models, transformation_models
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, resize
from torchvision.utils import save_image
from utils import deprocess_image, preprocess_image

root = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pretrained model paths
PRETRAINED_MODELS = {
    "starry_night": os.path.join(root, "models", "starry_night_pretrained.pth"),
    # "rain_princess": os.path.join(root, "models", "rain_princess_pretrained.pth"),
    # "abstract": os.path.join(root, "models", "abstract_pretrained.pth"),
    # "mosaic": os.path.join(root, "models", "mosaic_pretrained.pth"),
}

content_folder = os.path.join(root, "data", "contents")
output_folder = os.path.join(root, "data", "outputs")
os.makedirs(output_folder, exist_ok=True)

# Image processing parameters
image_size = (256, 256)  # or None to keep original size
mean = loss_models.VGG16Loss.MEAN.to(device)
std = loss_models.VGG16Loss.STD.to(device)

# Get content images
image_list = sorted(
    [
        f
        for f in os.listdir(content_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
)

# Benchmark storage
model_times = {k: [] for k in PRETRAINED_MODELS}

# Run stylization
for model_name, model_path in PRETRAINED_MODELS.items():
    # Load model
    model = transformation_models.TransformationModel().to(device).eval()
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.requires_grad_(False)

    for img_name in image_list:
        img_path = os.path.join(content_folder, img_name)
        img = pil_to_tensor(Image.open(img_path).convert("RGB")).to(device)
        if image_size:
            img = resize(img, image_size)
        img = preprocess_image(img, mean, std)

        # Stylize and time
        start = time.perf_counter()
        with torch.no_grad():
            gen = model(img)
        end = time.perf_counter()

        # Record time
        model_times[model_name].append(end - start)

        # Deprocess and save
        gen = deprocess_image(gen, mean, std)
        output_path = os.path.join(
            output_folder, f"{os.path.splitext(img_name)[0]}_{model_name}.jpg"
        )
        save_image(gen.squeeze(0), output_path)

        print(f"Saved {output_path}")

# Print average inference times
print("\n==== Average Inference Time per Model ====")
for model_name, times in model_times.items():
    avg_time = sum(times) / len(times)
    print(f"{model_name}: {avg_time:.4f} seconds/image")
