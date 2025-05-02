# Models - Technical Details and Adaptations

This directory contains the Python source code used for developing, adapting, and preparing the Neural Style Transfer (NST) models for the ArtEdge project. While the main project [README](../README.md) provides a high-level overview of the models and their role, this document details the specific technical modifications made to the original implementations or during the development of custom models.

These changes were often necessary to:

1.  Ensure compatibility with updated library versions (e.g., PyTorch, Torchvision).
2.  Facilitate conversion to the Core ML format for on-device deployment.
3.  Improve numerical stability or adherence to expected data ranges.
4.  Optimize components for mobile performance or static graph export.

Refer to the respective subdirectories (`AdaIN/`, `AesFA/`, `FST/`, `StyTR-2/`) for the full adapted source code.

## 1. StyTr² (Transformer-Based)

**Model Overview:** StyTr² utilizes a transformer architecture with separate encoders for content and style features and a decoder to blend them. It leverages self-attention to capture long-range dependencies and introduces Content-Aware Positional Encoding (CAPE) to better handle spatial relationships in images compared to traditional sinusoidal encodings.

**Technical Changes:**

The primary changes involved updating the codebase to handle deprecated functions in `torchvision`, as the required older version conflicted with other dependencies.

- **Replaced Deprecated `torchvision.ops.misc` Functions:**

  - The internal functions `_output_size` and `_new_empty_tensor` (used by `interpolate` in older torchvision versions) were removed or changed. Downgrading `torchvision` caused dependency conflicts.
  - **Solution:** These functions were reimplemented locally within the StyTr² helper files.

  ```python
  # Reimplementation of _output_size
  import math

  def _output_size(dim, input, size, scale_factor):
      if size is not None:
          return size
      assert scale_factor is not None, "Either size or scale_factor must be defined"
      scale_factor = [scale_factor] * dim if isinstance(scale_factor, float) else scale_factor
      input_size = input.shape[-dim:]
      return [int(math.floor(s * f)) for s, f in zip(input_size, scale_factor)]

  # Reimplementation of _new_empty_tensor
  def _new_empty_tensor(input, shape):
      return input.new_empty(shape)
  ```

  - A wrapper `interpolate` function was added to maintain compatibility, using the reimplemented helpers when needed or falling back to the standard `torchvision.ops.misc.interpolate` for newer versions.

  ```python
  # Compatibility wrapper for interpolate
  import torch
  import torchvision

  def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
      # Check torchvision version (example threshold, adjust if needed based on deprecation)
      if float(torchvision.__version__.split('.')[0] + '.' + torchvision.__version__.split('.')[1]) < 0.7: # Or the specific version where the change occurred
          if input.numel() > 0:
              # Use standard interpolate if possible (may depend on exact version/function signature)
               return torch.nn.functional.interpolate(
                   input, size, scale_factor, mode, align_corners
               )
          # Handle empty tensor case using reimplemented helpers
          output_shape = _output_size(2, input, size, scale_factor) # Assumes 2D interpolation
          output_shape = list(input.shape[:-2]) + list(output_shape)
          return _new_empty_tensor(input, output_shape)
      else:
          # Use the current standard torchvision interpolate
          return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)

  ```

- **Replaced Deprecated `torch._six` Import:**
  - Inside `models/ViT_helper.py`, the import `from torch._six import container_abcs` was deprecated.
  - **Solution:** Replaced with `import collections.abc as container_abcs`.

## 2. Fast Style Transfer (FST)

**Model Overview:** FST employs a feed-forward convolutional network trained using perceptual losses for real-time style transfer. Each model is typically trained for a single, specific style. It offers significant speed advantages over optimization-based methods.

**Technical Changes:**

- **Clamped Output Range:**

  - To ensure the output image pixel values remain within the standard normalized range [0, 1], which is crucial for subsequent processing and visualization, the output of the network's forward pass was clamped.
  - **Location:** `transformation_models.py` (within the `models/FST/` directory structure).
  - **Change:**

  _Before:_

  ```python
  def forward(self, content_img):
      return self.layers(content_img)
  ```

  _After:_

  ```python
  def forward(self, content_img):
      output = self.layers(content_img)
      # Clamp the output tensor to the [0, 1] range
      return output.clamp(0, 1)
  ```

## 3. MobileNet AdaIN (Custom)

**Model Overview:** This is a custom architecture designed for this project, prioritizing mobile performance and flexibility. It uses a lightweight MobileNetV2 encoder, an Adaptive Instance Normalization (AdaIN) layer to dynamically apply style features, and a lightweight decoder. Perceptual loss (using a pre-trained VGG19 network _during training only_) guides the learning process.

**Technical Details & Design Choices (No specific code _changes_ to an external base, as it's custom):**

- **Encoder:** MobileNetV2 was chosen for its efficiency (low compute/memory footprint) suitable for mobile deployment, replacing heavier backbones like VGG19 used in original NST papers.
- **Style Transfer Mechanism:** Adaptive Instance Normalization (AdaIN) is used. It aligns the channel-wise mean and standard deviation of content features to match those of the style features, enabling arbitrary style transfer with a single feed-forward pass.
- **Decoder:** A lightweight decoder architecture (Upsample, Conv, ReLU layers) reconstructs the stylized image. A final `Sigmoid` activation ensures the output pixel values are in the [0, 1] range.
- **Loss Function:** A combination of content loss and style loss (perceptual loss) is calculated using features extracted from multiple layers of a _fixed, pre-trained_ VGG19 network.
  - _Content Loss:_ MSE between features (e.g., `relu3_1` or `relu4_1`) of the stylized output and the original content image.
  - _Style Loss:_ MSE between Gram matrices of features (e.g., `relu1_1` to `relu5_1`) of the stylized output and the style image.
  - VGG19 is used for loss calculation due to its proven effectiveness in capturing perceptual similarity, even though MobileNetV2 is used for the main network's encoder.
- **Training Pipeline:** The encoder and decoder are trained end-to-end by minimizing the weighted sum of content and style losses, using content images and a fixed style image per batch/epoch.

## 4. AesFA (Mobile-Optimized)

**Model Overview:** AesFA is designed for efficient style transfer, particularly on mobile devices. It uses frequency-domain disentanglement and an aesthetic contrast loss, aiming to avoid computationally expensive components like VGG networks during inference.

**Technical Changes:**

Modifications were made to optimize the network for mobile deployment, specifically targeting Core ML conversion and inference efficiency.

- **Batch Size 1 Inference in `AdaConv2d`:**

  - The `AdaConv2d` module was refactored to explicitly handle `batch_size=1` during inference or tracing. This involved removing an internal loop over the batch dimension and adjusting how convolution weights/biases are applied.
  - _Reason:_ Simplifies the computation graph for tracing and optimizes for the common single-image inference scenario on mobile.

- **Static Pooling Layer:**

  - The dynamic `AdaptiveAvgPool2d` layer in the encoder was replaced with a fixed-size `AvgPool2d`.
  - _Reason:_ `AdaptiveAvgPool2d` produces an output size dependent on the input, making the computation graph dynamic. A fixed `AvgPool2d` results in a static graph, which is generally required or preferred for Core ML conversion and optimization.

- **Simplified `AdaOctConv`:**
  - The forward pass logic within the `AdaOctConv` module was simplified.
  - _Reason:_ Likely to improve compatibility with Core ML conversion tools, potentially enhance performance, or remove unnecessary complexity.
