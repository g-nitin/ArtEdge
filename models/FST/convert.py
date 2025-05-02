import os

import coremltools as ct
import torch
from modeling import transformation_models

print(f"Using torch version: {torch.__version__}")
print(f"Using coremltools version: {ct.__version__}")

# Configuration
# Define the input size the model expects (based on training or benchmark script)
CONTENT_SIZE = 256

# Select the specific pretrained FST model to convert
ROOT = os.path.dirname(os.path.abspath(__file__))  # Get the directory of this script
MODEL_NAME = "starry_night"
# MODEL_NAME = "rain_princess"
# MODEL_NAME = "abstract"
# MODEL_NAME = "mosaic"
CHECKPOINT_PATH = os.path.join(
    ROOT, "models", f"{MODEL_NAME}_pretrained.pth"
)  # Path to the pretrained model checkpoint

# Define the output path for the Core ML model
OUTPUT_COREML_PATH = os.path.join(ROOT, "models", f"FST-{MODEL_NAME}.mlpackage")

# Model Loading
print("Loading PyTorch FST model...")
if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint file not found at: {CHECKPOINT_PATH}")

# Instantiate the TransformationModel
pytorch_model = transformation_models.TransformationModel()

# Load the state dict from the checkpoint
# The benchmark script loads the full checkpoint first
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # Determine device
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)

# Check if the checkpoint contains 'model_state_dict' key
if "model_state_dict" in checkpoint:
    pytorch_model.load_state_dict(checkpoint["model_state_dict"])
    print("Loaded state dict from 'model_state_dict' key.")
else:
    # Attempt to load the entire checkpoint as the state dict (less common)
    try:
        pytorch_model.load_state_dict(checkpoint)
        print("Loaded state dict directly from checkpoint object.")
    except Exception as e:
        print(f"🔴 Error loading state dict: {e}")
        print("Checkpoint keys:", checkpoint.keys())
        raise RuntimeError("Could not load model state_dict from checkpoint.")


pytorch_model.to(device)  # Move model to appropriate device (important for tracing)
pytorch_model.eval()  # Set to evaluation mode
pytorch_model.requires_grad_(False)  # Disable gradients
print(f"PyTorch model '{MODEL_NAME}' loaded, set to eval mode, and gradients disabled.")

# No Wrapper Needed
# The TransformationModel's forward method already takes the single content image
# and returns the single stylized image. No wrapper is required.

# Prepare Example Input for Tracing
# Create a dummy tensor with the expected input shape [Batch, Channels, Height, Width]
# The model expects normalized input (like VGG).
# We don't need actual preprocessing here, just the correct shape and type.
example_content_input = torch.zeros(
    1, 3, CONTENT_SIZE, CONTENT_SIZE, dtype=torch.float32
).to(device)  # Ensure example input is on the same device as the model
print(f"Example content input shape: {example_content_input.shape}")

# TorchScript Conversion
print("Performing TorchScript conversion (tracing)...")
try:
    # Trace the original model directly
    scripted_model = torch.jit.trace(
        pytorch_model,
        example_content_input,  # Only pass the single content input
    )
    print("✅ Model traced successfully.")
except Exception as e_trace:
    print(f"🔴 Error during TorchScript tracing: {e_trace}")
    print("Tracing failed. Cannot proceed with conversion.")
    exit()

# Define Input/Output Types for Core ML
# Define the *single* input
content_input_name = "content_image"
inputs = [
    ct.TensorType(
        name=content_input_name,
        shape=example_content_input.shape,  # Use the shape of the example input
        dtype=float,  # Corresponds to torch.float32
    ),
    # No style input needed for FST models
]
print(
    f"Defined Core ML input: '{content_input_name}' with shape {example_content_input.shape}"
)

# Define the *single* output
output_name = "stylized"
# Let Core ML infer the output shape, but it should be the same as the input
# The model output is clamped between 0 and 1 (Float32)
outputs = [ct.TensorType(name=output_name, dtype=float)]
print(f"Defined Core ML output: '{output_name}' (shape will be inferred)")


# Convert the TorchScript Model to Core ML
print("Converting TorchScript model to Core ML...")
try:
    coreml_model = ct.convert(
        scripted_model,  # The TorchScript model object
        inputs=inputs,  # Define input shapes/types/names
        outputs=outputs,  # Define output name(s)
        convert_to="mlprogram",  # Recommended format
        compute_units=ct.ComputeUnit.ALL,  # Use CPU, GPU, and Neural Engine
        minimum_deployment_target=ct.target.iOS18,
    )

    # Add Metadata
    coreml_model.author = "ArtEdge"
    coreml_model.license = "MIT"
    coreml_model.short_description = (
        f"Fast Style Transfer ({MODEL_NAME.replace('_', ' ').title()})"
    )
    coreml_model.version = "1.0"

    # Input description
    # Note: The model expects input normalized using VGG mean/std.
    # The Swift code handles this normalization.
    coreml_model.input_description[content_input_name] = (
        f"Input content image (1, 3, {CONTENT_SIZE}, {CONTENT_SIZE}), Float32, Normalized CHW (VGG Mean/Std)"
    )

    # Output description
    # The model outputs an image clamped to [0, 1]. The Swift code handles de-normalization.
    # We need to check the actual output name Core ML assigned if we let it infer.
    try:
        # Get the actual output name from the spec after conversion
        spec = coreml_model.get_spec()
        actual_output_name = spec.description.output[0].name
        print(f"Inferred output name: '{actual_output_name}'")
        if actual_output_name != output_name:
            print(
                f"Warning: Core ML used output name '{actual_output_name}' instead of requested '{output_name}'. Using actual name for description."
            )
            output_name = actual_output_name  # Use the actual name

        coreml_model.output_description[output_name] = (
            f"Output stylized image (1, 3, {CONTENT_SIZE}, {CONTENT_SIZE}), Float32, Clamped [0,1], CHW"
        )
    except IndexError:
        print(
            "🔴 Error: Could not find any output description in the converted model spec."
        )
    except Exception as e:
        print(f"🔴 Error accessing output description: {e}")

    # Save the Core ML Model
    print(f"Saving Core ML model to: {OUTPUT_COREML_PATH}")
    coreml_model.save(OUTPUT_COREML_PATH)
    print("✅ Core ML model saved successfully.")

    # Inspect Saved Core ML Model (Optional)
    print("\nInspecting Saved Core ML Model")
    try:
        saved_mlmodel = ct.models.MLModel(OUTPUT_COREML_PATH)
        spec = saved_mlmodel.get_spec()
        print("Model Input Descriptions:")
        for inp in spec.description.input:
            print(f"  Name: {inp.name}, Type: {inp.type.WhichOneof('Type')}")
            if inp.type.WhichOneof("Type") == "multiArrayType":
                print(f"    Shape: {[dim for dim in inp.type.multiArrayType.shape]}")
                print(f"    Data Type: {inp.type.multiArrayType.dataType}")
        print("Model Output Descriptions:")
        for outp in spec.description.output:
            print(f"  Name: {outp.name}, Type: {outp.type.WhichOneof('Type')}")
            if outp.type.WhichOneof("Type") == "multiArrayType":
                # Shape might not be fully defined if inferred
                print(f"    Shape: {[dim for dim in outp.type.multiArrayType.shape]}")
                print(f"    Data Type: {outp.type.multiArrayType.dataType}")

    except Exception as e:
        print(f"🔴 Error inspecting saved model: {e}")
    print("---------------------------------")


except Exception as e:
    print(f"🔴 Core ML conversion failed: {e}")
    import traceback

    traceback.print_exc()
    print(
        "Check the error message. Ensure the TorchScript model is valid and coremltools supports all operators."
    )
