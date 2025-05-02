import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import lr_scheduler


def model_save(ckpt_dir, model, optim_E, optim_S, optim_G, epoch, itr=None):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save(
        {
            "netE": model.netE.state_dict(),
            "netS": model.netS.state_dict(),
            "netG": model.netG.state_dict(),
            "optim_E": optim_E.state_dict(),
            "optim_S": optim_S.state_dict(),
            "optim_G": optim_G.state_dict(),
        },
        "%s/model_iter_%d_epoch_%d.pth" % (ckpt_dir, itr + 1, epoch + 1),
    )


def model_load(checkpoint, ckpt_dir, model, optim_E, optim_S, optim_G):
    if not os.path.exists(ckpt_dir):
        epoch = -1
        return model, optim_E, optim_S, optim_G, epoch

    ckpt_path = Path(ckpt_dir)
    if checkpoint:
        model_ckpt = ckpt_path / checkpoint
    else:
        ckpt_lst = list(
            ckpt_path.glob("model_iter_*")
        )  # Convert generator to list for sorting
        # Ensure sorting is robust if filenames vary slightly
        ckpt_lst.sort(key=lambda x: int(x.stem.split("iter_")[1].split("_epoch")[0]))
        if not ckpt_lst:
            epoch = -1
            print(f"Warning: No checkpoints found in {ckpt_dir}")
            return model, optim_E, optim_S, optim_G, epoch
        model_ckpt = ckpt_lst[-1]

    # Use stem to avoid issues with potential multiple dots in filename
    parts = model_ckpt.stem.split("_epoch_")
    itr_part = parts[0].split("iter_")[1]
    epoch_part = parts[1]
    itr = int(itr_part)
    epoch = int(epoch_part) - 1  # Epochs usually 0-indexed internally

    print(f"Loading checkpoint: {model_ckpt}")
    print(f"Iteration: {itr}, Epoch: {epoch}")

    dict_model = torch.load(
        model_ckpt, map_location=lambda storage, loc: storage
    )  # Safer loading

    model.netE.load_state_dict(dict_model["netE"])
    model.netS.load_state_dict(dict_model["netS"])
    model.netG.load_state_dict(dict_model["netG"])
    optim_E.load_state_dict(dict_model["optim_E"])
    optim_S.load_state_dict(dict_model["optim_S"])
    optim_G.load_state_dict(dict_model["optim_G"])

    return model, optim_E, optim_S, optim_G, epoch, itr


def test_model_load(checkpoint, model):
    print(f"Loading test checkpoint: {checkpoint}")
    # Ensure loading onto CPU if no GPU available during conversion
    dict_model = torch.load(
        checkpoint, map_location=torch.device("cpu"), weights_only=True
    )
    # Load with strict=True first to catch structural mismatches
    try:
        model.netE.load_state_dict(dict_model["netE"], strict=True)
        model.netS.load_state_dict(dict_model["netS"], strict=True)
        model.netG.load_state_dict(dict_model["netG"], strict=True)
        print("Test model weights loaded successfully (strict=True).")
    except RuntimeError as e:
        print("Strict loading failed. Trying with strict=False. Error was:")
        print(e)
        # If strict loading fails, try non-strict but be aware of potential issues
        model.netE.load_state_dict(dict_model["netE"], strict=False)
        model.netS.load_state_dict(dict_model["netS"], strict=False)
        model.netG.load_state_dict(dict_model["netG"], strict=False)
        print("Test model weights loaded with strict=False. Check warnings carefully.")

    return model


def get_scheduler(optimizer, config):
    if config.lr_policy == "lambda":

        def lambda_rule(epoch):
            # Ensure n_iter_decay exists in config or handle appropriately
            n_iter_decay = getattr(
                config, "n_iter_decay", config.n_iter // 2
            )  # Example default
            lr_l = 1.0 - max(
                0, epoch + getattr(config, "n_epoch", 0) - config.n_iter
            ) / float(n_iter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif config.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=config.lr_decay_iters, gamma=0.1
        )
    elif config.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, threshold=0.01, patience=5
        )
    elif config.lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.n_iter, eta_min=0
        )
    else:
        raise NotImplementedError(  # Use raise for errors
            f"learning rate policy [{config.lr_policy}] is not implemented"
        )
    return scheduler


def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]["lr"]
    print("learning rate = %.7f" % lr)


class Oct_Conv_aftup(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        pad_type,
        alpha_in,
        alpha_out,
    ):
        super(Oct_Conv_aftup, self).__init__()
        lf_in = int(in_channels * alpha_in)
        lf_out = int(out_channels * alpha_out)
        hf_in = in_channels - lf_in
        hf_out = out_channels - lf_out

        self.conv_h = nn.Conv2d(
            in_channels=hf_in,
            out_channels=hf_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=pad_type,
        )
        self.conv_l = nn.Conv2d(
            in_channels=lf_in,
            out_channels=lf_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=pad_type,
        )

    def forward(self, x):
        hf, lf = x
        hf = self.conv_h(hf)
        lf = self.conv_l(lf)
        return hf, lf


class Oct_conv_reLU(nn.ReLU):
    def forward(self, x):
        hf, lf = x
        hf = super(Oct_conv_reLU, self).forward(hf)
        lf = super(Oct_conv_reLU, self).forward(lf)
        return hf, lf


class Oct_conv_lreLU(nn.LeakyReLU):
    def forward(self, x):
        hf, lf = x
        hf = super(Oct_conv_lreLU, self).forward(hf)
        lf = super(Oct_conv_lreLU, self).forward(lf)
        return hf, lf


class Oct_conv_up(nn.Upsample):
    def forward(self, x):
        hf, lf = x
        hf = super(Oct_conv_up, self).forward(hf)
        lf = super(Oct_conv_up, self).forward(lf)
        return hf, lf


############## Encoder ##############
class OctConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        pad_type="reflect",
        alpha_in=0.5,
        alpha_out=0.5,
        type="normal",
        freq_ratio=[1, 1],
    ):
        super(OctConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.type = type
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.freq_ratio = freq_ratio

        hf_ch_in = int(in_channels * (1 - self.alpha_in))
        hf_ch_out = int(out_channels * (1 - self.alpha_out))
        lf_ch_in = in_channels - hf_ch_in
        lf_ch_out = out_channels - hf_ch_out

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.is_dw = groups == in_channels

        if type == "first":
            self.convh = nn.Conv2d(
                in_channels,
                hf_ch_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=pad_type,
                bias=False,
            )
            self.convl = nn.Conv2d(
                in_channels,
                lf_ch_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=pad_type,
                bias=False,
            )
        elif type == "last":
            self.convh = nn.Conv2d(
                hf_ch_in,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=pad_type,
                bias=False,
            )
            self.convl = nn.Conv2d(
                lf_ch_in,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=pad_type,
                bias=False,
            )
        else:
            self.L2L = nn.Conv2d(
                lf_ch_in,
                lf_ch_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=math.ceil(alpha_in * groups),
                padding_mode=pad_type,
                bias=False,
            )
            if self.is_dw:
                self.L2H = None
                self.H2L = None
            else:
                self.L2H = nn.Conv2d(
                    lf_ch_in,
                    hf_ch_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    padding_mode=pad_type,
                    bias=False,
                )
                self.H2L = nn.Conv2d(
                    hf_ch_in,
                    lf_ch_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    padding_mode=pad_type,
                    bias=False,
                )
            self.H2H = nn.Conv2d(
                hf_ch_in,
                hf_ch_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=math.ceil(groups - alpha_in * groups),
                padding_mode=pad_type,
                bias=False,
            )

    def forward(self, x):
        if self.type == "first":
            hf = self.convh(x)
            lf = self.avg_pool(x)
            lf = self.convl(lf)
            return hf, lf
        elif self.type == "last":
            hf, lf = x
            out_h = self.convh(hf)
            out_l = self.convl(self.upsample(lf))
            # Ensure freq_ratio are floats or tensors for multiplication
            output = out_h * float(self.freq_ratio[0]) + out_l * float(
                self.freq_ratio[1]
            )
            return output, out_h, out_l
        else:
            hf, lf = x
            if self.is_dw:
                hf, lf = self.H2H(hf), self.L2L(lf)
            else:
                # Ensure upsample/avg_pool are applied correctly before addition
                lf_upsampled = self.upsample(lf)
                hf_pooled = self.avg_pool(hf)
                hf_new = self.H2H(hf) + self.L2H(lf_upsampled)
                lf_new = self.L2L(lf) + self.H2L(hf_pooled)
                hf, lf = hf_new, lf_new
            return hf, lf


############## Decoder ##############
class AdaOctConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        group_div,
        style_channels,
        kernel_size,
        stride,
        padding,
        oct_groups,
        alpha_in,
        alpha_out,
        type="normal",
    ):
        super(AdaOctConv, self).__init__()
        self.in_channels = in_channels
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.type = type

        h_in = int(in_channels * (1 - self.alpha_in))
        l_in = in_channels - h_in

        # Ensure group_div doesn't lead to zero groups if h_in/l_in is small
        n_groups_h = max(1, h_in // group_div)
        n_groups_l = max(1, l_in // group_div)

        style_channels_h = int(style_channels * (1 - self.alpha_in))
        style_channels_l = int(style_channels - style_channels_h)

        kernel_size_h = kernel_size[0]
        kernel_size_l = kernel_size[1]
        kernel_size_A = kernel_size[2]  # Kernel for the final OctConv

        self.kernelPredictor_h = KernelPredictor(
            in_channels=h_in,
            out_channels=h_in,
            n_groups=n_groups_h,
            style_channels=style_channels_h,
            kernel_size=kernel_size_h,
        )
        self.kernelPredictor_l = KernelPredictor(
            in_channels=l_in,
            out_channels=l_in,
            n_groups=n_groups_l,
            style_channels=style_channels_l,
            kernel_size=kernel_size_l,
        )

        # Pass the actual kernel size used by AdaConv2d's internal conv
        # It seems AdaConv2d uses kernel_size=3 internally, separate from the predictor kernel size
        self.AdaConv_h = AdaConv2d(
            in_channels=h_in, out_channels=h_in, kernel_size=3, n_groups=n_groups_h
        )
        self.AdaConv_l = AdaConv2d(
            in_channels=l_in, out_channels=l_in, kernel_size=3, n_groups=n_groups_l
        )

        self.OctConv = OctConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size_A,  # Use the third kernel size for OctConv
            stride=stride,
            padding=padding,
            groups=oct_groups,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
            type=type,
        )

        self.relu = Oct_conv_lreLU()

    def forward(self, content, style, cond="train"):
        c_hf, c_lf = content
        s_hf, s_lf = style
        h_w_spatial, h_w_pointwise, h_bias = self.kernelPredictor_h(s_hf)
        l_w_spatial, l_w_pointwise, l_bias = self.kernelPredictor_l(s_lf)

        # Always use the AdaConv part, regardless of cond for tracing simplicity
        output_h = self.AdaConv_h(c_hf, h_w_spatial, h_w_pointwise, h_bias)
        output_l = self.AdaConv_l(c_lf, l_w_spatial, l_w_pointwise, l_bias)
        output = output_h, output_l

        output = self.relu(output)

        output = self.OctConv(output)
        if self.type != "last":
            output = self.relu(output)
        # else: output is (final_tensor, high_freq_tensor, low_freq_tensor)
        return output


class KernelPredictor(nn.Module):
    def __init__(
        self, in_channels, out_channels, n_groups, style_channels, kernel_size
    ):
        super(KernelPredictor, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_groups = n_groups
        # self.w_channels = style_channels # Not used directly
        self.kernel_size = kernel_size

        # Ensure kernel_size is odd for symmetric padding
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd for symmetric padding.")
        # Calculate padding as integer
        padding = (kernel_size - 1) // 2

        self.spatial = nn.Conv2d(
            style_channels,
            in_channels
            * out_channels
            // n_groups,  # Make sure n_groups divides evenly or handle remainder
            kernel_size=kernel_size,
            padding=padding,  # Use integer padding
            padding_mode="reflect",
        )
        self.pointwise = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(
                style_channels, out_channels * out_channels // n_groups, kernel_size=1
            ),
        )
        self.bias = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(style_channels, out_channels, kernel_size=1),
        )

    def forward(self, w):
        w_spatial = self.spatial(w)
        # Use tensor.shape[0] instead of len(w)
        batch_size = w.shape[0]
        w_spatial = (
            w_spatial.view(  # Use view instead of reshape for contiguity if needed
                batch_size,
                self.out_channels,
                self.in_channels // self.n_groups,
                self.kernel_size,
                self.kernel_size,
            )
        )

        w_pointwise = self.pointwise(w)
        w_pointwise = w_pointwise.view(
            batch_size, self.out_channels, self.out_channels // self.n_groups, 1, 1
        )
        bias = self.bias(w)
        bias = bias.view(batch_size, self.out_channels)  # Bias is (B, C)
        return w_spatial, w_pointwise, bias


class AdaConv2d(nn.Module):
    # Default kernel_size to 3 as it seems fixed in AdaOctConv usage
    def __init__(self, in_channels, out_channels, kernel_size=3, n_groups=None):
        super(AdaConv2d, self).__init__()
        self.n_groups = in_channels if n_groups is None else n_groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size  # Store kernel size

        # Ensure kernel_size is odd
        if kernel_size % 2 == 0:
            raise ValueError("AdaConv2d internal kernel size must be odd.")
        # Calculate integer padding
        padding = (kernel_size - 1) // 2

        # This convolution is applied *after* the dynamic parts.
        # Restore its original structure (3x3, with bias) to match the checkpoint.
        self.conv = nn.Conv2d(
            in_channels=in_channels,  # Input to this conv is the output of the dynamic part
            out_channels=out_channels,
            kernel_size=(
                kernel_size,
                kernel_size,
            ),  # Restore original kernel size (e.g., 3x3)
            padding=(padding, padding),  # Restore original padding
            padding_mode="reflect",
            bias=True,  # Restore bias=True to match checkpoint keys
        )

    def forward(self, x, w_spatial, w_pointwise, bias):
        # Use shape[0] for batch size
        batch_size = x.shape[0]

        # Apply instance norm
        x_norm = F.instance_norm(x)

        # Processing each item in the batch individually is bad for exporting
        # Need to vectorize this if possible, or accept that export might only work for batch_size=1
        # If batch_size > 1 during export, this might need more complex handling (e.g., torch.vmap)
        # For mobile inference (batch_size=1) should be okay
        if batch_size != 1:
            print(
                "Warning: AdaConv2d ONNX export might only work reliably for batch_size=1 due to dynamic weights."
            )
            # Fallback to loop for non-batch-1 cases if needed during Python execution,
            # but export will likely trace the batch_size=1 path.
            ys = []
            for i in range(batch_size):
                y = self.forward_single(
                    x_norm[i : i + 1],  # Keep batch dim
                    w_spatial[i],  # Remove batch dim for single weights/bias
                    w_pointwise[i],
                    bias[i],
                )
                ys.append(y)
            intermediate_output = torch.cat(ys, dim=0)

        else:
            # Directly call forward_single for batch_size=1
            # Note: Need to keep batch dim for x_norm, but remove for weights/bias
            intermediate_output = self.forward_single(
                x_norm,  # Shape [1, C_in, H, W]
                w_spatial[0],  # Shape [C_out, C_in/G, K, K]
                w_pointwise[0],  # Shape [C_out, C_out/G, 1, 1]
                bias[0],  # Shape [C_out]
            )

        # Apply the final fixed convolution (self.conv)
        output = self.conv(
            intermediate_output
        )  # Apply final conv AFTER dynamic part + bias
        return output

    def forward_single(self, x, w_spatial, w_pointwise, bias):
        # x shape: [1, C_in, H, W]
        # w_spatial shape: [C_out, C_in/G, K, K]
        # w_pointwise shape: [C_out, C_out/G, 1, 1]
        # bias shape: [C_out]

        # Kernel size from spatial weights
        k = w_spatial.size(-1)

        # Calculate padding (assuming k is always odd now)
        padding = (k - 1) // 2
        pad = (padding, padding, padding, padding)  # Symmetric padding

        # 1. Padding
        x_padded = F.pad(x, pad=pad, mode="reflect")

        # 2. Spatial Convolution (Grouped) - NO BIAS
        # Ensure w_spatial has the correct shape [C_out, C_in // groups, k, k]
        x_spatial = F.conv2d(x_padded, w_spatial, groups=self.n_groups, bias=None)

        # 3. Pointwise Convolution (Grouped) - NO BIAS
        # Ensure w_pointwise has the correct shape [C_out, C_out // groups, 1, 1]
        x_pointwise = F.conv2d(x_spatial, w_pointwise, groups=self.n_groups, bias=None)

        # 4. Add Dynamic Bias MANUALLY
        # Reshape dynamic bias from [C_out] to [1, C_out, 1, 1] for broadcasting
        bias_reshaped = bias.view(1, -1, 1, 1)
        output_before_final_conv = (
            x_pointwise + bias_reshaped
        )  # Result before self.conv

        return output_before_final_conv  # Return the intermediate result
