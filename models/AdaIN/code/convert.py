import os

import coremltools as ct
import torch
import torch.nn as nn
from mobilenet_adain import MobileNetAdaINNet

print(f"Using torch version: {torch.__version__}")
print(f"Using coremltools version: {ct.__version__}")

# Configuration
IMAGE_SIZE = 256

# Choose the specific style model you want to convert
model_style_name = "brushstrokes"
# model_style_name = "mondrian"
# model_style_name = "starrynight"
ROOT = os.path.dirname(os.path.abspath(__file__))  # Get the directory of this script
CHECKPOINT_PATH = os.path.join(
    ROOT, "..", "models", f"adain_mobilenet_model_{model_style_name}.pth"
)
OUTPUT_COREML_PATH = os.path.join(
    ROOT, "..", "models", f"AdaIN-{model_style_name}.mlpackage"
)

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(OUTPUT_COREML_PATH), exist_ok=True)

# Load PyTorch Model
print("Loading PyTorch AdaIN MobileNet model...")
if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint file not found at: {CHECKPOINT_PATH}")

# Instantiate the AdaIN model
pytorch_model = MobileNetAdaINNet()

# Load the state dictionary
state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
print("Loaded state dict with weights_only=True.")
pytorch_model.load_state_dict(state_dict)
pytorch_model.eval()
print("PyTorch model loaded and set to evaluation mode.")


# Wrapper for Simplified Inference
# The MobileNetAdaINNet already has a simple forward, but wrapper ensures eval mode
class AdaINWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()  # Ensure model is in eval mode

    def forward(self, content_image, style_image):
        # Direct call to the model's forward method
        stylized_image = self.model(content_image, style_image)
        # Clamp output similar to inference script - this might be better done in Swift
        # or ensured by the model's final layer (Sigmoid already does [0,1])
        # stylized_image = torch.clamp(stylized_image, 0, 1)
        return stylized_image


wrapper_model = AdaINWrapper(pytorch_model)
wrapper_model.eval()
print("Wrapper model created.")

# Prepare Example Inputs for Tracing
# Shape: (Batch Size, Channels, Height, Width)
example_content_input = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
example_style_input = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
print(f"Example content input shape: {example_content_input.shape}")
print(f"Example style input shape: {example_style_input.shape}")

# TorchScript Conversion
print("Performing TorchScript conversion (tracing)...")
try:
    # Use torch.jit.trace for models like this
    scripted_model = torch.jit.trace(
        wrapper_model, (example_content_input, example_style_input)
    )
    print("✅ Model traced successfully.")
except Exception as e_trace:
    print(f"🔴 Error during TorchScript tracing: {e_trace}")
    print("Tracing failed. Cannot proceed with conversion.")
    exit()

# Define Input/Output Types and Names for Core ML

# Define desired input names (can be overridden by graph inspection below)
content_input_name = "content_image"
style_input_name = "style_image"

# Attempt to get names from the traced graph (more robust)
try:
    graph_input_names = [inp.debugName() for inp in scripted_model.graph.inputs()]
    print(f"Detected graph input names: {graph_input_names}")
    # The first input in the graph is often 'self', so we skip it.
    if len(graph_input_names) > 1:
        content_input_name = graph_input_names[1]  # Usually the first *actual* input
    if len(graph_input_names) > 2:
        style_input_name = graph_input_names[2]  # Usually the second *actual* input
    print(
        f"Using input names for Core ML: '{content_input_name}', '{style_input_name}'"
    )
except Exception as e_graph:
    print(
        f"Warning: Could not automatically determine graph input names: {e_graph}. Using default names '{content_input_name}', '{style_input_name}'."
    )

# Define input types as Tensors.
# The Swift code handles preprocessing (resizing, normalization to [0,1], CHW)
inputs = [
    ct.TensorType(
        name=content_input_name,
        shape=example_content_input.shape,
        dtype=float,  # float is Float32
    ),
    ct.TensorType(name=style_input_name, shape=example_style_input.shape, dtype=float),
]

# Define Output Name and Type
output_name = "stylized"
# Output is also a tensor. Core ML will infer the shape.
# The model's decoder ends with Sigmoid, so output is Float32 in [0, 1] range.
outputs = [ct.TensorType(name=output_name, dtype=float)]

# Convert the TorchScript Model to Core ML
print("Converting TorchScript model to Core ML...")
try:
    # Convert using ct.convert
    coreml_model = ct.convert(
        scripted_model,  # The traced TorchScript model
        inputs=inputs,  # Defined inputs
        outputs=outputs,  # Defined outputs
        convert_to="mlprogram",  # Recommended format
        compute_units=ct.ComputeUnit.ALL,  # Use CPU, GPU, and Neural Engine
        minimum_deployment_target=ct.target.iOS18,
    )

    # Add Metadata
    coreml_model.author = "ArtEdge"
    coreml_model.license = "MIT"
    coreml_model.short_description = f"AdaIN Style Transfer using MobileNetV2 Encoder/Decoder ({model_style_name} style)"
    coreml_model.version = "1.0"

    # Add Input/Output Descriptions
    # Input descriptions using the determined names
    coreml_model.input_description[content_input_name] = (
        f"Content image as Float32 tensor, shape (1, 3, {IMAGE_SIZE}, {IMAGE_SIZE}). "
        f"Expected format: CHW, pixel values scaled to [0, 1]."
    )
    coreml_model.input_description[style_input_name] = (
        f"Style image as Float32 tensor, shape (1, 3, {IMAGE_SIZE}, {IMAGE_SIZE}). "
        f"Expected format: CHW, pixel values scaled to [0, 1]."
    )

    # Output description
    # The output shape should match the content input shape
    coreml_model.output_description[output_name] = (
        f"Stylized image as Float32 tensor, shape (1, 3, {IMAGE_SIZE}, {IMAGE_SIZE}). "
        f"Format: CHW, pixel values are in the [0, 1] range."
    )

    # Save the Core ML Model
    print(f"Saving Core ML model to: {OUTPUT_COREML_PATH}")
    coreml_model.save(OUTPUT_COREML_PATH)
    print("✅ Core ML model saved successfully.")

    # Inspect Saved Model (Optional)
    print("\n*** Inspecting Saved Core ML Model")
    try:
        saved_mlmodel = ct.models.MLModel(OUTPUT_COREML_PATH)
        spec = saved_mlmodel.get_spec()
        print("Model Input Descriptions:")
        for input_desc in spec.description.input:
            print(f"  Name: {input_desc.name}")
            print(f"  Type: {input_desc.type.WhichOneof('Type')}")
            if input_desc.type.WhichOneof("Type") == "multiArrayType":
                print(
                    f"  Shape: {[dim for dim in input_desc.type.multiArrayType.shape]}"
                )
                print(f"  Data Type: {input_desc.type.multiArrayType.dataType}")
            print(f"  Description: {saved_mlmodel.input_description[input_desc.name]}")

        print("\nModel Output Descriptions:")
        for output_desc in spec.description.output:
            print(f"  Name: {output_desc.name}")
            print(f"  Type: {output_desc.type.WhichOneof('Type')}")
            if output_desc.type.WhichOneof("Type") == "multiArrayType":
                # Output shape might be flexible/symbolic, print if available
                if output_desc.type.multiArrayType.shape:
                    print(
                        f"  Shape: {[dim for dim in output_desc.type.multiArrayType.shape]}"
                    )
                else:
                    print("  Shape: (Inferred by Core ML)")
                print(f"  Data Type: {output_desc.type.multiArrayType.dataType}")
            print(
                f"  Description: {saved_mlmodel.output_description[output_desc.name]}"
            )
        print("*********************************")
    except Exception as e_inspect:
        print(f"Warning: Could not inspect the saved model: {e_inspect}")


except Exception as e:
    print(f"🔴 Core ML conversion failed: {e}")
    import traceback

    traceback.print_exc()
    print(
        "\nCheck the error message. Common issues include:"
        "\n- Unsupported PyTorch operators."
        "\n- Incorrect input shapes or types during tracing."
        "\n- Issues with the loaded checkpoint file."
        "\n- Compatibility problems between torch, coremltools, and python versions."
    )
