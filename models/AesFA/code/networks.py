import torch
from blocks import AdaOctConv, Oct_Conv_aftup, Oct_conv_lreLU, Oct_conv_up, OctConv
from torch import nn


def define_network(net_type, config=None):
    net = None
    alpha_in = config.alpha_in
    alpha_out = config.alpha_out
    sk = config.style_kernel

    if net_type == "Encoder":
        net = Encoder(
            in_dim=config.input_nc,
            nf=config.nf,
            style_kernel_size=config.style_kernel,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
        )
    elif net_type == "Generator":
        net = Decoder(
            nf=config.nf,
            out_dim=config.output_nc,
            style_channel=256,
            style_kernel=[sk, sk, 3],
            alpha_in=alpha_in,
            freq_ratio=config.freq_ratio,
            alpha_out=alpha_out,
        )
    return net


class Encoder(nn.Module):
    def __init__(self, in_dim, nf=64, style_kernel_size=3, alpha_in=0.5, alpha_out=0.5):
        super(Encoder, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_dim, out_channels=nf, kernel_size=7, stride=1, padding=3
        )

        # Assuming alpha_in=0.5, alpha_out=0.5 for channel calculations below

        # Layer Block 1
        nf_hf_in = nf  # From initial conv
        nf_lf_in = 0  # From initial conv (no low freq yet)
        nf_hf_out1 = int(nf * (1 - alpha_out))  # e.g., 32
        nf_lf_out1 = nf - nf_hf_out1  # e.g., 32
        self.OctConv1_1 = OctConv(
            in_channels=nf,
            out_channels=nf,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=nf,  # Depthwise usually matches input channels
            alpha_in=0.0,
            alpha_out=alpha_out,  # alpha_in is 0 for 'first' type conceptually
            type="first",
        )  # Output: hf=(B, 32, 112, 112), lf=(B, 32, 56, 56)

        nf_hf_in2 = nf_hf_out1  # 32
        nf_lf_in2 = nf_lf_out1  # 32
        nf_hf_out2 = int(2 * nf * (1 - alpha_out))  # e.g., 64
        nf_lf_out2 = 2 * nf - nf_hf_out2  # e.g., 64
        self.OctConv1_2 = OctConv(
            in_channels=nf,
            out_channels=2 * nf,
            kernel_size=1,
            alpha_in=alpha_out,
            alpha_out=alpha_out,  # Use previous alpha_out as current alpha_in
            type="normal",
        )  # Output: hf=(B, 64, 112, 112), lf=(B, 64, 56, 56)

        nf_hf_in3 = nf_hf_out2  # 64
        nf_lf_in3 = nf_lf_out2  # 64
        nf_hf_out3 = int(2 * nf * (1 - alpha_out))  # e.g., 64
        nf_lf_out3 = 2 * nf - nf_hf_out3  # e.g., 64
        self.OctConv1_3 = OctConv(
            in_channels=2 * nf,
            out_channels=2 * nf,
            kernel_size=3,
            stride=1,
            padding=1,
            alpha_in=alpha_out,
            alpha_out=alpha_out,
            type="normal",
        )  # Output: hf=(B, 64, 112, 112), lf=(B, 64, 56, 56)

        # Layer Block 2
        nf_hf_in4 = nf_hf_out3  # 64
        nf_lf_in4 = nf_lf_out3  # 64
        nf_hf_out4 = int(2 * nf * (1 - alpha_out))  # e.g., 64
        nf_lf_out4 = 2 * nf - nf_hf_out4  # e.g., 64
        self.OctConv2_1 = OctConv(
            in_channels=2 * nf,
            out_channels=2 * nf,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=2 * nf,  # Depthwise
            alpha_in=alpha_out,
            alpha_out=alpha_out,
            type="normal",
        )  # Output: hf=(B, 64, 56, 56), lf=(B, 64, 28, 28)

        nf_hf_in5 = nf_hf_out4  # 64
        nf_lf_in5 = nf_lf_out4  # 64
        nf_hf_out5 = int(4 * nf * (1 - alpha_out))  # e.g., 128
        nf_lf_out5 = 4 * nf - nf_hf_out5  # e.g., 128
        self.OctConv2_2 = OctConv(
            in_channels=2 * nf,
            out_channels=4 * nf,
            kernel_size=1,
            alpha_in=alpha_out,
            alpha_out=alpha_out,
            type="normal",
        )  # Output: hf=(B, 128, 56, 56), lf=(B, 128, 28, 28)

        nf_hf_in6 = nf_hf_out5  # 128
        nf_lf_in6 = nf_lf_out5  # 128
        nf_hf_out6 = int(4 * nf * (1 - alpha_out))  # e.g., 128
        nf_lf_out6 = 4 * nf - nf_hf_out6  # e.g., 128
        self.OctConv2_3 = OctConv(
            in_channels=4 * nf,
            out_channels=4 * nf,
            kernel_size=3,
            stride=1,
            padding=1,
            alpha_in=alpha_out,
            alpha_out=alpha_out,
            type="normal",
        )  # Output: hf=(B, 128, 56, 56), lf=(B, 128, 28, 28)

        # replacement for AdaptiveAvgPool2d
        # Calculate kernel/stride based on expected input size (56x56 for hf, 28x28 for lf)
        # and target output size (style_kernel_size x style_kernel_size, e.g., 3x3)

        # For High Frequency Path (Input 56x56 -> Output 3x3)
        in_size_h = 56
        out_size_h = style_kernel_size
        stride_h = in_size_h // out_size_h  # Integer division gives floor
        kernel_h = in_size_h - (out_size_h - 1) * stride_h
        print(
            f"Encoder Style Pool (High Freq): Input={in_size_h}x{in_size_h}, Output={out_size_h}x{out_size_h} -> Using AvgPool2d(kernel={kernel_h}, stride={stride_h})"
        )
        self.pool_h = nn.AvgPool2d(kernel_size=kernel_h, stride=stride_h)

        # For Low Frequency Path (Input 28x28 -> Output 3x3)
        in_size_l = 28
        out_size_l = style_kernel_size
        stride_l = in_size_l // out_size_l  # Integer division gives floor
        kernel_l = in_size_l - (out_size_l - 1) * stride_l
        print(
            f"Encoder Style Pool (Low Freq): Input={in_size_l}x{in_size_l}, Output={out_size_l}x{out_size_l} -> Using AvgPool2d(kernel={kernel_l}, stride={stride_l})"
        )
        self.pool_l = nn.AvgPool2d(kernel_size=kernel_l, stride=stride_l)

        self.relu = Oct_conv_lreLU()

    def forward(self, x):
        enc_feat = []
        out = self.conv(x)

        out = self.OctConv1_1(out)
        out = self.relu(out)
        out = self.OctConv1_2(out)
        out = self.relu(out)
        out = self.OctConv1_3(out)
        out = self.relu(out)
        enc_feat.append(out)  # Appends tuple (hf, lf)

        out = self.OctConv2_1(out)
        out = self.relu(out)
        out = self.OctConv2_2(out)
        out = self.relu(out)
        out = self.OctConv2_3(out)
        out = self.relu(out)
        enc_feat.append(out)  # Appends tuple (hf, lf)

        # Pooling for style features (uses the new AvgPool2d layers)
        out_high, out_low = out
        out_sty_h = self.pool_h(out_high)
        out_sty_l = self.pool_l(out_low)
        out_sty = out_sty_h, out_sty_l  # Tuple of pooled features

        # Returns:
        # 1. Last feature map tuple (hf, lf) before pooling
        # 2. Pooled style feature tuple (pooled_hf, pooled_lf)
        # 3. List of intermediate feature map tuples [(hf1, lf1), (hf2, lf2)]
        return out, out_sty, enc_feat

    def forward_test(self, x, cond):
        # This forward is used during testing and ONNX export
        out = self.conv(x)

        out = self.OctConv1_1(out)
        out = self.relu(out)
        out = self.OctConv1_2(out)
        out = self.relu(out)
        out = self.OctConv1_3(out)
        out = self.relu(out)

        out = self.OctConv2_1(out)
        out = self.relu(out)
        out = self.OctConv2_2(out)
        out = self.relu(out)
        out = self.OctConv2_3(out)
        out = self.relu(out)
        # 'out' is now the tuple (hf, lf) from OctConv2_3

        if cond == "style":
            out_high, out_low = out
            # Apply the NEW fixed AvgPool2d layers
            out_sty_h = self.pool_h(out_high)
            out_sty_l = self.pool_l(out_low)
            return out_sty_h, out_sty_l  # Return tuple of pooled features
        else:  # cond == "content"
            # Return the unpooled feature tuple (hf, lf) from OctConv2_3
            return out


class Decoder(nn.Module):
    def __init__(
        self,
        nf=64,
        out_dim=3,
        style_channel=512,
        style_kernel=[3, 3, 3],
        alpha_in=0.5,
        alpha_out=0.5,
        freq_ratio=[1, 1],
        pad_type="reflect",
    ):
        super(Decoder, self).__init__()

        group_div = [1, 2, 4, 8]
        self.up_oct = Oct_conv_up(scale_factor=2)

        self.AdaOctConv1_1 = AdaOctConv(
            in_channels=4 * nf,
            out_channels=4 * nf,
            group_div=group_div[0],
            style_channels=style_channel,
            kernel_size=style_kernel,
            stride=1,
            padding=1,
            oct_groups=4 * nf,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
            type="normal",
        )
        self.OctConv1_2 = OctConv(
            in_channels=4 * nf,
            out_channels=2 * nf,
            kernel_size=1,
            stride=1,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
            type="normal",
        )
        self.oct_conv_aftup_1 = Oct_Conv_aftup(
            in_channels=2 * nf,
            out_channels=2 * nf,
            kernel_size=3,
            stride=1,
            padding=1,
            pad_type=pad_type,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
        )

        self.AdaOctConv2_1 = AdaOctConv(
            in_channels=2 * nf,
            out_channels=2 * nf,
            group_div=group_div[1],
            style_channels=style_channel,
            kernel_size=style_kernel,
            stride=1,
            padding=1,
            oct_groups=2 * nf,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
            type="normal",
        )
        self.OctConv2_2 = OctConv(
            in_channels=2 * nf,
            out_channels=nf,
            kernel_size=1,
            stride=1,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
            type="normal",
        )
        self.oct_conv_aftup_2 = Oct_Conv_aftup(
            nf, nf, 3, 1, 1, pad_type, alpha_in, alpha_out
        )

        self.AdaOctConv3_1 = AdaOctConv(
            in_channels=nf,
            out_channels=nf,
            group_div=group_div[2],
            style_channels=style_channel,
            kernel_size=style_kernel,
            stride=1,
            padding=1,
            oct_groups=nf,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
            type="normal",
        )
        self.OctConv3_2 = OctConv(
            in_channels=nf,
            out_channels=nf // 2,
            kernel_size=1,
            stride=1,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
            type="last",
            freq_ratio=freq_ratio,
        )

        self.conv4 = nn.Conv2d(in_channels=nf // 2, out_channels=out_dim, kernel_size=1)

    def forward(self, content, style):
        out = self.AdaOctConv1_1(content, style)
        out = self.OctConv1_2(out)
        out = self.up_oct(out)
        out = self.oct_conv_aftup_1(out)

        out = self.AdaOctConv2_1(out, style)
        out = self.OctConv2_2(out)
        out = self.up_oct(out)
        out = self.oct_conv_aftup_2(out)

        out = self.AdaOctConv3_1(out, style)
        out = self.OctConv3_2(out)
        out, out_high, out_low = out

        out = self.conv4(out)
        out_high = self.conv4(out_high)
        out_low = self.conv4(out_low)

        return out, out_high, out_low

    def forward_test(self, content, style):
        out = self.AdaOctConv1_1(content, style, "test")
        out = self.OctConv1_2(out)
        out = self.up_oct(out)
        out = self.oct_conv_aftup_1(out)

        out = self.AdaOctConv2_1(out, style, "test")
        out = self.OctConv2_2(out)
        out = self.up_oct(out)
        out = self.oct_conv_aftup_2(out)

        out = self.AdaOctConv3_1(out, style, "test")
        out = self.OctConv3_2(out)

        out = self.conv4(out[0])
        return out


############## Contrastive Loss function ##############
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def calc_content_loss(input, target):
    assert input.size() == target.size()
    mse_loss = nn.MSELoss()
    return mse_loss(input, target)


def calc_style_loss(input, target):
    assert input.size() == target.size()
    mse_loss = nn.MSELoss()
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)

    loss = mse_loss(input_mean, target_mean) + mse_loss(input_std, target_std)
    return loss


class EFDM_loss(nn.Module):
    def __init__(self):
        super(EFDM_loss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def efdm_single(self, style, trans):
        B, C, W, H = style.size(0), style.size(1), style.size(2), style.size(3)

        value_style, index_style = torch.sort(style.view(B, C, -1))
        value_trans, index_trans = torch.sort(trans.view(B, C, -1))
        inverse_index = index_trans.argsort(-1)

        return self.mse_loss(
            trans.view(B, C, -1), value_style.gather(-1, inverse_index)
        )

    def forward(self, style_E, style_S, translate_E, translate_S, neg_idx):
        loss = 0.0
        batch = style_E[0][0].shape[0]
        for b in range(batch):
            poss_loss = 0.0
            neg_loss = 0.0

            # Positive loss
            for i in range(len(style_E)):
                poss_loss += self.efdm_single(
                    style_E[i][0][b].unsqueeze(0), translate_E[i][0][b].unsqueeze(0)
                ) + self.efdm_single(
                    style_E[i][1][b].unsqueeze(0), translate_E[i][1][b].unsqueeze(0)
                )
            for i in range(len(style_S)):
                poss_loss += self.efdm_single(
                    style_S[i][0][b].unsqueeze(0), translate_S[i][0][b].unsqueeze(0)
                ) + self.efdm_single(
                    style_S[i][1][b].unsqueeze(0), translate_S[i][1][b].unsqueeze(0)
                )

            # Negative loss
            for nb in neg_idx[b]:
                for i in range(len(style_E)):
                    neg_loss += self.efdm_single(
                        style_E[i][0][nb].unsqueeze(0),
                        translate_E[i][0][b].unsqueeze(0),
                    ) + self.efdm_single(
                        style_E[i][1][nb].unsqueeze(0),
                        translate_E[i][1][b].unsqueeze(0),
                    )
                for i in range(len(style_S)):
                    neg_loss += self.efdm_single(
                        style_S[i][0][nb].unsqueeze(0),
                        translate_S[i][0][b].unsqueeze(0),
                    ) + self.efdm_single(
                        style_S[i][1][nb].unsqueeze(0),
                        translate_S[i][1][b].unsqueeze(0),
                    )

            loss += poss_loss / neg_loss

        return loss
