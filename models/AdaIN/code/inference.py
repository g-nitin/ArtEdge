import os
import time

import torch
from mobilenet_adain import MobileNetAdaINNet
from PIL import Image
from torchvision import transforms

root = os.path.dirname(os.path.abspath(__file__))
# model_name = "brushstrokes"
# model_name = "mondrian"
model_name = "starrynight"
model_path = os.path.join(
    root, "..", "models", f"adain_mobilenet_model_{model_name}.pth"
)
content_dir = os.path.join(root, "..", "data", "contents")
style_image_path = os.path.join(root, "..", "data", "styles", "starry_night.jpg")
output_dir = os.path.join(root, "..", "data", "outputs")

image_size = 256  # should match training size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = MobileNetAdaINNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

# Image transforms
transform = transforms.Compose(
    [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
)


def load_image(path):
    return transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def save_image(tensor, path):
    tensor = tensor.squeeze(0).detach().cpu()
    image = transforms.ToPILImage()(tensor)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)


# Load fixed style image
style_img = load_image(style_image_path)

# Process all content images
content_files = [
    f for f in os.listdir(content_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
processing_times = []  # List to store time taken for each image

for filename in content_files:
    content_path = os.path.join(content_dir, filename)
    output_path = os.path.join(output_dir, f"stylized_{filename}")

    content_img = load_image(content_path)

    start_time = time.time()  # Start timer
    with torch.no_grad():
        stylized = model(content_img, style_img)
        stylized = torch.clamp(stylized, 0, 1)
    end_time = time.time()  # End timer

    duration = end_time - start_time
    processing_times.append(duration)

    save_image(stylized, output_path)
    print(f"Saved: {output_path} | Time: {duration:.4f} seconds")

# Calculate and print average time
if processing_times:
    average_time = sum(processing_times) / len(processing_times)
    print(f"\nProcessed {len(content_files)} images.")
    print(f"Average processing time per image: {average_time:.4f} seconds")
else:
    print("\nNo images were processed.")
