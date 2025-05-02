import glob
import os
import random
import sys

import torch
import torch.nn as nn
from mobilenet_adain import MobileNetAdaINNet
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = "/work/nayeem/Neuromorphic/NST"


# === Gram Matrix ===
def gram_matrix(tensor):
    B, C, H, W = tensor.size()
    features = tensor.view(B, C, H * W)
    G = torch.bmm(features, features.transpose(1, 2))
    return G / (C * H * W)


# === VGG Feature Extractor for Perceptual Loss ===
class VGGPerceptual(nn.Module):
    def __init__(self):
        super(VGGPerceptual, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.enc_1 = nn.Sequential(*vgg[:4])  # relu1_1
        self.enc_2 = nn.Sequential(*vgg[4:9])  # relu2_1
        self.enc_3 = nn.Sequential(*vgg[9:16])  # relu3_1
        self.enc_4 = nn.Sequential(*vgg[16:23])  # relu4_1
        self.enc_5 = nn.Sequential(*vgg[23:30])  # relu5_1
        for param in self.parameters():
            param.requires_grad = False
        self.eval().to(device)

    def forward(self, x):
        results = []
        x = self.enc_1(x)
        results.append(x)
        x = self.enc_2(x)
        results.append(x)
        x = self.enc_3(x)
        results.append(x)
        x = self.enc_4(x)
        results.append(x)
        x = self.enc_5(x)
        results.append(x)
        return results


class FlatImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, subset_size=None):
        self.transform = transform
        self.images = glob.glob(os.path.join(root_dir, "*.jpg"))
        if subset_size:
            self.images = random.sample(self.images, subset_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # dummy label


def get_loader(content_dir, image_size=224, batch_size=4, subset_size=10000):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = FlatImageDataset(
        content_dir, transform=transform, subset_size=subset_size
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# === Load style image ===
def load_style_image(style_path, size=224):
    image = Image.open(style_path).convert("RGB")
    transform = transforms.Compose(
        [transforms.Resize((size, size)), transforms.ToTensor()]
    )
    return transform(image).unsqueeze(0).to(device)


# === Main Training ===
def train(
    content_root=f"{root}/input/coco/train2017/",
    style_img_path=f"{root}/input/style/starry_night.jpg",
    epochs=40,
    batch_size=4,
    lr=1e-3,
    style_weight=1e4,
    content_weight=10,
    save_every=500,
):
    loader = get_loader(content_root, batch_size=batch_size)
    style_img = load_style_image(style_img_path)

    model = MobileNetAdaINNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    vgg = VGGPerceptual()

    for epoch in range(epochs):
        for i, (content_imgs, _) in enumerate(loader):
            content_imgs = content_imgs.to(device)
            style_imgs = style_img.expand_as(content_imgs)

            output = model(content_imgs, style_imgs)
            print("Output shape:", output.shape)

            # Extract VGG features
            content_feats = vgg(content_imgs)
            style_feats = vgg(style_imgs)
            output_feats = vgg(output)

            # === Content loss ===
            content_loss = nn.MSELoss()(output_feats[3], content_feats[3])  # relu3_1

            # === Style loss ===
            style_loss = (
                sum(
                    nn.MSELoss()(gram_matrix(of), gram_matrix(sf))
                    for of, sf in zip(output_feats[:3], style_feats[:3])
                )
                / 3
            )

            total_loss = content_weight * content_loss + style_weight * style_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            output = torch.clamp(output, 0, 1)

            if i % 10 == 0:
                print(
                    f"[{epoch + 1}/{epochs}][{i}/{len(loader)}] "
                    f"Loss: {total_loss.item():.4f} (C: {content_loss.item():.4f}, S: {style_loss.item():.4f})"
                )

            if i % save_every == 0:
                os.makedirs("output", exist_ok=True)
                save_image(
                    torch.cat([content_imgs[0], style_imgs[0], output[0]], dim=2),
                    f"output/epoch{epoch}_step{i}_triplet.png",
                )
            sys.stdout.flush()

    torch.save(
        model.state_dict(),
        f"{root}/saved_models/adain_mobilenet_model_starry_night.pth",
    )
    print("Model saved as adain_mobilenet_model_starry_night.pth")


if __name__ == "__main__":
    train()
