import os

import numpy as np
import thop
import torch
from blocks import test_model_load
from Config import Config
from DataSplit import DataSplit
from model import AesFA_test
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor


def load_img(img_name, img_size, device):
    img = Image.open(img_name).convert("RGB")
    img = do_transform(img, img_size).to(device)
    if len(img.shape) == 3:
        img = img.unsqueeze(0)  # make batch dimension
    return img


def im_convert(tensor):
    image = tensor.to("cpu").clone().detach().numpy()
    image = image.transpose(0, 2, 3, 1)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image


def do_transform(img, osize):
    transform = Compose(
        [
            Resize(size=osize),  # Resize to keep aspect ratio
            CenterCrop(size=osize),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return transform(img)


def save_img(
    config,
    cont_name,
    sty_name,
    content,
    style,
    stylized,
    freq=False,
    high=None,
    low=None,
):
    real_A = im_convert(content)
    real_B = im_convert(style)
    trs_AtoB = im_convert(stylized)

    A_image = Image.fromarray((real_A[0] * 255.0).astype(np.uint8))
    B_image = Image.fromarray((real_B[0] * 255.0).astype(np.uint8))
    trs_image = Image.fromarray((trs_AtoB[0] * 255.0).astype(np.uint8))

    A_image.save(
        "{}/{:s}_content_{:s}.jpg".format(config.img_dir, cont_name.stem, sty_name.stem)
    )
    B_image.save(
        "{}/{:s}_style_{:s}.jpg".format(config.img_dir, cont_name.stem, sty_name.stem)
    )
    trs_image.save(
        "{}/{:s}_stylized_{:s}.jpg".format(
            config.img_dir, cont_name.stem, sty_name.stem
        )
    )

    if freq:
        trs_AtoB_high = im_convert(high)
        trs_AtoB_low = im_convert(low)

        trsh_image = Image.fromarray((trs_AtoB_high[0] * 255.0).astype(np.uint8))
        trsl_image = Image.fromarray((trs_AtoB_low[0] * 255.0).astype(np.uint8))

        trsh_image.save(
            "{}/{:s}_stylizing_high_{:s}.jpg".format(
                config.img_dir, cont_name.stem, sty_name.stem
            )
        )
        trsl_image.save(
            "{}/{:s}_stylizing_low_{:s}.jpg".format(
                config.img_dir, cont_name.stem, sty_name.stem
            )
        )


def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)
    print(f"Style Control: HF Weight={config.hf_weight}, LF Weight={config.lf_weight}")

    # Model Load
    print("Loading checkpoint: ", config.model)
    model = AesFA_test(config)
    model = test_model_load(checkpoint=config.model, model=model)
    model.to(device)
    model.eval()  # Set model to evaluation mode

    if not os.path.exists(config.img_dir):
        os.makedirs(config.img_dir)

    # Determine Image Lists
    if config.content_dir and config.style_dir:
        print(
            f"Processing images from directories: {config.content_dir} and {config.style_dir}"
        )
        # Use DataSplit logic to get lists
        test_data = DataSplit(config=config, phase="test")
        contents = test_data.images
        styles = test_data.style_images
        multi_to_multi = config.multi_to_multi  # Use config setting

    # Start Testing
    freq_output_needed = (
        config.save_freq_components
    )  # Request freq outputs only if saving them
    count = 0
    t_during = 0
    total_flops = 0
    total_params = 0

    with torch.no_grad():
        if multi_to_multi:  # N content images, M style image -> N * M outputs
            tot_imgs = len(contents) * len(styles)
            print(
                f"Processing {len(contents)} content images against {len(styles)} style images ({tot_imgs} total pairs)..."
            )
            for idx, cont_path in enumerate(contents):
                content = load_img(cont_path, config.test_content_size, device)

                for i, sty_path in enumerate(styles):
                    style = load_img(sty_path, config.test_style_size, device)

                    # Select Mode
                    if freq_output_needed:
                        stylized, stylized_high, stylized_low, during = model(
                            content,
                            style,
                            freq=True,
                            hf_weight=config.hf_weight,
                            lf_weight=config.lf_weight,
                        )
                        save_img(
                            config,
                            cont_path,
                            sty_path,
                            content,
                            style,
                            stylized,
                            freq=True,
                            high=stylized_high,
                            low=stylized_low,
                        )
                    else:
                        stylized, during = model(
                            content,
                            style,
                            freq=False,
                            hf_weight=config.hf_weight,
                            lf_weight=config.lf_weight,
                        )
                        save_img(
                            config,
                            cont_path,
                            sty_path,
                            content,
                            style,
                            stylized,
                            freq=False,
                        )

                    # Performance Metrics (run once or less frequently)
                    if count == 0:  # Profile only on the first iteration
                        # Profile based on the standard forward pass structure
                        profile_input_style = (
                            style  # Use the primary style for profiling
                        )
                        profile_freq = False  # Profile the more common case
                        flops, params = thop.profile(
                            model,
                            inputs=(
                                content,
                                profile_input_style,
                                profile_freq,
                                config.hf_weight,
                                config.lf_weight,
                            ),
                            verbose=False,
                        )
                        total_flops = flops
                        total_params = params
                        print(
                            f"GFLOPS: {total_flops / 1e9:.4f}, Params: {total_params / 1e6:.4f} M"
                        )
                        if device.type == "cuda":
                            print(
                                f"Initial Max GPU memory allocated: {torch.cuda.max_memory_allocated(device=config.gpu) / (1024**3):.4f} GB"
                            )

                    count += 1
                    print(
                        f"Processed pair {count}/{tot_imgs}: {cont_path.name} + {sty_path.name} ({during:.4f} sec)"
                    )
                    t_during += during

        else:  # Process pairs: (content[0], style[0]), (content[1], style[1]), ... OR single pair
            num_pairs = len(contents)
            print(f"Processing {num_pairs} content/style pair(s)...")
            for idx in range(num_pairs):
                cont_path = contents[idx]
                sty_path = styles[idx]

                content = load_img(cont_path, config.test_content_size, device)
                style = load_img(sty_path, config.test_style_size, device)

            if freq_output_needed:
                stylized, stylized_high, stylized_low, during = model(
                    content,
                    style,
                    freq=True,
                    hf_weight=config.hf_weight,
                    lf_weight=config.lf_weight,
                )
                save_img(
                    config,
                    cont_path,
                    sty_path,
                    content,
                    style,
                    stylized,
                    freq=True,
                    high=stylized_high,
                    low=stylized_low,
                )
            else:
                stylized, during = model(
                    content,
                    style,
                    freq=False,
                    hf_weight=config.hf_weight,
                    lf_weight=config.lf_weight,
                )
                save_img(
                    config,
                    cont_path,
                    sty_path,
                    content,
                    style,
                    stylized,
                    freq=False,
                )

                # Performance Metrics
                if count == 0:  # Profile only on the first iteration
                    profile_input_style = style
                    profile_freq = False
                    flops, params = thop.profile(
                        model,
                        inputs=(
                            content,
                            profile_input_style,
                            profile_freq,
                            config.hf_weight,
                            config.lf_weight,
                        ),
                        verbose=False,
                    )
                    total_flops = flops
                    total_params = params
                    print(
                        f"GFLOPS: {total_flops / 1e9:.4f}, Params: {total_params / 1e6:.4f} M"
                    )
                    if device.type == "cuda":
                        print(
                            f"Initial Max GPU memory allocated: {torch.cuda.max_memory_allocated(device=config.gpu) / (1024**3):.4f} GB"
                        )

                count += 1
                print(
                    f"Processed pair {count}/{num_pairs}: {cont_path.name} + {sty_path.name} ({during:.4f} sec)"
                )
                t_during += during

    avg_time = t_during / count if count > 0 else 0
    print("\nSummary")
    print(
        f"Content size: {config.test_content_size}, Style size: {config.test_style_size}"
    )
    print(f"Total images processed: {count}")
    print(f"Avg Testing time per pair: {avg_time:.4f} sec")
    if total_flops > 0:
        print(f"GFLOPS: {total_flops / 1e9:.4f}, Params: {total_params / 1e6:.4f} M")
    if device.type == "cuda":
        print(
            f"Final Max GPU memory allocated: {torch.cuda.max_memory_allocated(device=config.gpu) / (1024**3):.4f} GB"
        )


if __name__ == "__main__":
    main()
