import os

import coremltools as ct
import torch
import torch.nn as nn

print(f"Using torch version: {torch.__version__}")
print(f"Using coremltools version: {ct.__version__}")

# Configuration
IMAGE_SIZE = 256
root = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(root, "experiments", "stytr2_scripted.pt")
output_coreml_path = os.path.join(root, "experiments", "StyTr2.mlpackage")

# Normalization Constants
# ImageNet standard
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

print("Loading original TorchScript model...")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"TorchScript model file not found at: {model_path}")

# Load the pre-scripted StyTr2 model
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    original_scripted_model = torch.jit.load(model_path, map_location=device)
    original_scripted_model.eval()
    print("✅ Original TorchScript model loaded successfully.")
except Exception as e:
    print(f"🔴 Error loading original TorchScript model: {e}")
    import traceback

    traceback.print_exc()
    exit()


# Wrapper Definition
class NormalizationWrapper(nn.Module):
    def __init__(self, core_model, mean, std):
        super().__init__()
        self.core_model = core_model
        # Register mean and std as buffers to ensure they move with the model (e.g., .to(device))
        # Shape: (1, C, 1, 1) for broadcasting
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        print("NormalizationWrapper initialized.")
        print(f"  Mean shape: {self.mean.shape}, Std shape: {self.std.shape}")
        print(f"  Mean device: {self.mean.device}, Std device: {self.std.device}")

    def denormalize(self, img_norm):
        # Convert normalized input (from Swift) back to [0, 1] for StyTr2
        # Ensure tensors are on the same device before operations
        if img_norm.device != self.mean.device:
            # This should ideally not happen if model and inputs are moved correctly,
            # but good practice to check or ensure consistency beforehand.
            print(
                f"Warning: DeNormalizing - Device mismatch. Input: {img_norm.device}, Mean/Std: {self.mean.device}. Attempting to use input device."
            )
            mean = self.mean.to(img_norm.device)
            std = self.std.to(img_norm.device)
        else:
            mean = self.mean
            std = self.std

        img_0_1 = (img_norm * std) + mean
        # Clamp to ensure output is strictly within [0, 1] after potential float inaccuracies
        return torch.clamp(img_0_1, 0.0, 1.0)

    def normalize(self, img_0_1):
        # Convert StyTr2's [0, 1] output back to normalized space for Swift
        if img_0_1.device != self.mean.device:
            print(
                f"Warning: Normalizing - Device mismatch. Input: {img_0_1.device}, Mean/Std: {self.mean.device}. Attempting to use input device."
            )
            mean = self.mean.to(img_0_1.device)
            std = self.std.to(img_0_1.device)
        else:
            mean = self.mean
            std = self.std

        return (img_0_1 - mean) / std

    def forward(self, content_normalized, style_normalized):
        # print(f"Wrapper forward called on device: {content_normalized.device}") # Debug device
        # 1. De-normalize inputs for the core StyTr2 model
        content_0_1 = self.denormalize(content_normalized)
        style_0_1 = self.denormalize(style_normalized)
        # print(f"  Content denormalized min: {content_0_1.min()}, max: {content_0_1.max()}") # Debug range
        # print(f"  Style denormalized min: {style_0_1.min()}, max: {style_0_1.max()}")   # Debug range

        # 2. Run the core StyTr2 model
        # Ensure core model is also on the correct device (should be if wrapper is)
        stylized_0_1 = self.core_model(content_0_1, style_0_1)
        # print(f"  Core model output min: {stylized_0_1.min()}, max: {stylized_0_1.max()}") # Debug range

        # 3. Re-normalize the output to match AesFA's output space
        stylized_normalized = self.normalize(stylized_0_1)
        # print(f"  Final normalized output min: {stylized_normalized.min()}, max: {stylized_normalized.max()}") # Debug range
        return stylized_normalized


# Instantiate and Prepare Wrapper
print("Creating and preparing normalization wrapper...")
wrapper_model = NormalizationWrapper(original_scripted_model, MEAN, STD)
wrapper_model.to(
    device
)  # Move wrapper (and its buffers/submodules) to the target device
wrapper_model.eval()
print("Wrapper model created and set to evaluation mode.")

# Prepare Example Inputs (NORMALIZED) for Tracing/Scripting the WRAPPER
# These should represent the data *after* Swift's normalization
# Using random data in the typical normalized range might be slightly better than zeros
example_content_input_normalized = torch.randn(
    1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32
).to(device)
example_style_input_normalized = torch.randn(
    1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32
).to(device)
# example_content_input_normalized = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32).to(device) # Alternative: zeros
# example_style_input_normalized = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32).to(device) # Alternative: zeros
print(
    f"Example normalized content input shape: {example_content_input_normalized.shape}"
)
print(f"Example normalized style input shape: {example_style_input_normalized.shape}")


# Trace or Script the WRAPPER model
print("Performing TorchScript conversion (tracing) on the WRAPPER...")
try:
    # Trace the wrapper model with the *normalized* example inputs
    scripted_wrapper_model = torch.jit.trace(
        wrapper_model,
        (example_content_input_normalized, example_style_input_normalized),
    )
    print("✅ Wrapper model traced successfully.")
except Exception as e_trace:
    print(f"🔴 Error during TorchScript tracing of wrapper: {e_trace}")
    print("Tracing failed. Cannot proceed with conversion.")
    exit()


# Define Input Types for Core ML (using traced wrapper)
content_input_name = "content_image"
style_input_name = "style_image"

# Attempt to get input names from the *wrapper's* graph
try:
    graph_input_names = [
        inp.debugName() for inp in scripted_wrapper_model.graph.inputs()
    ]
    print(f"Detected wrapper graph input names: {graph_input_names}")
    # Use similar logic as before to assign names if possible, otherwise defaults
    if len(graph_input_names) > 1:  # Index 0 is often 'self'
        content_input_name = graph_input_names[1]
    if len(graph_input_names) > 2:
        style_input_name = graph_input_names[2]
    print(
        f"Using input names for Core ML: '{content_input_name}', '{style_input_name}'"
    )
except Exception as e_graph:
    print(
        f"Warning: Could not automatically determine wrapper graph input names: {e_graph}. Using defaults."
    )

inputs = [
    ct.TensorType(
        name=content_input_name,
        shape=example_content_input_normalized.shape,
        dtype=float,
    ),
    ct.TensorType(
        name=style_input_name, shape=example_style_input_normalized.shape, dtype=float
    ),
]

# Define Output Name (wrapper output is normalized)
output_name = "stylized"
outputs = [ct.TensorType(name=output_name, dtype=float)]  # Let Core ML infer shape

# Convert the *Traced Wrapper* Model to Core ML
print("Converting traced WRAPPER model to Core ML...")
try:
    coreml_model = ct.convert(
        scripted_wrapper_model,  # The traced wrapper object
        inputs=inputs,  # Define input shapes/types/names (normalized)
        outputs=outputs,  # Define output name (normalized)
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS18,
    )

    # Add Metadata - CRITICAL: Describe inputs/outputs as NORMALIZED
    coreml_model.author = "ArtEdge"
    coreml_model.license = "MIT"
    coreml_model.short_description = (
        "StyTr2 Style Transfer (Wrapped to accept/output normalized tensors)"
    )
    coreml_model.version = "1.0"

    coreml_model.input_description[content_input_name] = (
        f"{content_input_name} image (1, 3, {IMAGE_SIZE}, {IMAGE_SIZE}), Float32, Normalized CHW"
    )
    coreml_model.input_description[style_input_name] = (
        f"{style_input_name} image (1, 3, {IMAGE_SIZE}, {IMAGE_SIZE}), Float32, Normalized CHW"
    )

    # Get actual output name and set description
    spec = coreml_model.get_spec()
    if len(spec.description.output) > 0:
        actual_output_name = spec.description.output[0].name
        if actual_output_name != output_name:
            print(
                f"Note: Core ML assigned output name '{actual_output_name}'. Using actual name."
            )
            output_name = actual_output_name
        coreml_model.output_description[output_name] = (
            f"{output_name} image (1, 3, {IMAGE_SIZE}, {IMAGE_SIZE}), Float32, Normalized CHW"
        )
    else:
        print(
            "Warning: Could not find any outputs in the converted model specification."
        )

    # Save the Core ML Model
    print(f"Saving Core ML model to: {output_coreml_path}")
    os.makedirs(os.path.dirname(output_coreml_path), exist_ok=True)
    coreml_model.save(output_coreml_path)
    print("✅ Wrapped Core ML model saved successfully.")

    # Inspect saved model details
    print("\n--- Inspecting Saved Wrapped Core ML Model ---")
    saved_mlmodel = ct.models.MLModel(output_coreml_path)
    spec_saved = saved_mlmodel.get_spec()
    print("Model Input Descriptions from Spec:")
    for inp in spec_saved.description.input:
        print(
            f"  Name: {inp.name}, Type: {inp.type.WhichOneof('Type')}, Shape: {inp.type.multiArrayType.shape if inp.type.WhichOneof('Type') == 'multiArrayType' else 'N/A'}"
        )
    print("Model Output Descriptions from Spec:")
    for outp in spec_saved.description.output:
        print(
            f"  Name: {outp.name}, Type: {outp.type.WhichOneof('Type')}, Shape: {outp.type.multiArrayType.shape if outp.type.WhichOneof('Type') == 'multiArrayType' else 'N/A'}"
        )
    print("---------------------------------")


except Exception as e:
    print(f"🔴 Core ML conversion failed: {e}")
    import traceback

    traceback.print_exc()
    print(
        "Check the error message. Ensure the wrapper and core model ops are Core ML compatible."
    )
