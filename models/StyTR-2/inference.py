import os
import time

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# Configs
root = os.path.dirname(os.path.abspath(__file__))
content_dir = os.path.join(root, "data", "contents")
style_dir = os.path.join(root, "data", "styles")
output_dir = os.path.join(root, "data", "outputs")
model_path = os.path.join(root, "experiments", "stytr2_scripted.pt")
image_size = 256

os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Preprocessing
def load_and_preprocess(image_path):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


# Load model
print("Loading model...")
model = torch.jit.load(model_path).to(device).eval()

# Get all images
content_images = sorted(
    [
        os.path.join(content_dir, f)
        for f in os.listdir(content_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
)
style_images = sorted(
    [
        os.path.join(style_dir, f)
        for f in os.listdir(style_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
)

print(f"{len(content_images)} content images, {len(style_images)} style images")

# Benchmarking loop
timings = {}
total_time = 0
num_inferences = 0

for style_path in style_images:
    style_name = os.path.splitext(os.path.basename(style_path))[0]
    timings[style_name] = []

    style_tensor = load_and_preprocess(style_path)

    for content_path in content_images:
        content_name = os.path.splitext(os.path.basename(content_path))[0]
        content_tensor = load_and_preprocess(content_path)

        # Time the inference
        start = time.perf_counter()
        with torch.no_grad():
            output = model(content_tensor, style_tensor)
        end = time.perf_counter()

        elapsed = end - start
        timings[style_name].append(elapsed)
        total_time += elapsed
        num_inferences += 1

        # Save output
        output_image = output.cpu().squeeze(0)
        output_path = os.path.join(
            output_dir, f"{content_name}_stylized_{style_name}.jpg"
        )
        save_image(output_image, output_path)

        print(f"Saved: {output_path} | Time: {elapsed:.4f}s")

# Print summary
print("\n==== Inference Timing Summary ====")
for style_name, times in timings.items():
    avg_time = sum(times) / len(times)
    print(f"Style: {style_name} | Avg time: {avg_time:.4f}s over {len(times)} images")

print(f"\nTotal time: {total_time:.2f}s for {num_inferences} inferences")
