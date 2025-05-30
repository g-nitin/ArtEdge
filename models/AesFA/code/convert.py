import os

import coremltools as ct
import torch
import torch.nn as nn
from blocks import test_model_load
from Config import Config
from model import AesFA_test

print(f"Using torch version: {torch.__version__}")
print(f"Using coremltools version: {ct.__version__}")

config = Config()
CONTENT_SIZE = config.test_content_size
STYLE_SIZE = config.test_style_size
CHECKPOINT_PATH = config.model
OUTPUT_COREML_PATH = "models/AesFA.mlpackage"

print("Loading PyTorch model...")
if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint file not found at: {CHECKPOINT_PATH}")

pytorch_model = AesFA_test(config)
pytorch_model = test_model_load(checkpoint=CHECKPOINT_PATH, model=pytorch_model)
pytorch_model.eval()
print("PyTorch model loaded and set to evaluation mode.")


# Wrapper for Simplified Inference
class StyleTransferWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.netE.eval()
        self.model.netS.eval()
        self.model.netG.eval()

    def forward(self, content_image, style_image):
        content_features = self.model.netE.forward_test(content_image, "content")
        style_features = self.model.netS.forward_test(style_image, "style")
        stylized_image = self.model.netG.forward_test(content_features, style_features)
        return stylized_image


wrapper_model = StyleTransferWrapper(pytorch_model)
wrapper_model.eval()
print("Wrapper model created.")

# *** Prepare Example Inputs for Tracing/Scripting ***
example_content_input = torch.zeros(
    1, 3, CONTENT_SIZE, CONTENT_SIZE, dtype=torch.float32
)
example_style_input = torch.zeros(1, 3, STYLE_SIZE, STYLE_SIZE, dtype=torch.float32)
print(f"Example content input shape: {example_content_input.shape}")
print(f"Example style input shape: {example_style_input.shape}")

print("Performing TorchScript conversion...")
try:
    # Use torch.jit.trace as fallback
    scripted_model = torch.jit.trace(
        wrapper_model, (example_content_input, example_style_input)
    )
    print("✅ Model traced successfully.")
except Exception as e_trace:
    print(f"🔴 Error during TorchScript tracing: {e_trace}")
    print("Both scripting and tracing failed. Cannot proceed with direct conversion.")
    exit()

# Define Input Types for Core ML
content_input_name = "content"
style_input_name = "style"

# Verify input names from the scripted model
try:
    graph_input_names = [inp.debugName() for inp in scripted_model.graph.inputs()]
    print(f"Detected graph input names: {graph_input_names}")
    if len(graph_input_names) > 1:
        content_input_name = graph_input_names[1]
    if len(graph_input_names) > 2:
        style_input_name = graph_input_names[2]
    print(
        f"Using input names for Core ML: '{content_input_name}', '{style_input_name}'"
    )
except Exception as e_graph:
    print(
        f"Warning: Could not automatically determine graph input names: {e_graph}. Using default names."
    )

inputs = [
    ct.TensorType(
        name=content_input_name, shape=example_content_input.shape, dtype=float
    ),
    ct.TensorType(name=style_input_name, shape=example_style_input.shape, dtype=float),
]

# Define Output Name
output_name = "stylized"
outputs = [ct.TensorType(name=output_name, dtype=float)]  # Let Core ML infer shape

# Convert the TorchScript Model to Core ML
print("Converting TorchScript model to Core ML...")
try:
    coreml_model = ct.convert(
        scripted_model,  # The TorchScript model object
        inputs=inputs,  # Define input shapes/types/names
        outputs=outputs,  # Define output name (optional)
        # source="pytorch",         # Can explicitly state source, but often inferred
        convert_to="mlprogram",  # Recommended format (default usually)
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS18,
    )

    # Add Metadata
    coreml_model.author = "ArtEdge"
    coreml_model.license = "MIT"
    coreml_model.short_description = "AesFA Neural Style Transfer Model"
    coreml_model.version = "1.0"

    # Input descriptions using the determined names
    coreml_model.input_description[content_input_name] = (
        f"{content_input_name} image (1, 3, {CONTENT_SIZE}, {CONTENT_SIZE}), Float32, Normalized CHW"
    )
    coreml_model.input_description[style_input_name] = (
        f"{style_input_name} image (1, 3, {STYLE_SIZE}, {STYLE_SIZE}), Float32, Normalized CHW"
    )

    # Output description
    if coreml_model.output_description[output_name]:
        coreml_model.output_description[output_name] = (
            f"{output_name} image (1, 3, {CONTENT_SIZE}, {CONTENT_SIZE}), Float32, Normalized CHW"
        )
    else:
        print(
            f"Warning: Could not find output named '{output_name}' in description. Check conversion."
        )

    # Save the Core ML Model
    print(f"Saving Core ML model to: {OUTPUT_COREML_PATH}")
    coreml_model.save(OUTPUT_COREML_PATH)
    print("✅ Core ML model saved successfully.")

    print("\n*** Inspecting Saved Core ML Model ***")
    # Load the saved model
    saved_mlmodel = ct.models.MLModel(OUTPUT_COREML_PATH)
    spec = saved_mlmodel.get_spec()
    print("Model Output Descriptions:")
    for output in spec.description.output:
        print(f"  Name: {output.name}, Type: {output.type}")
    print("*********************************")

except Exception as e:
    print(f"🔴 Core ML conversion failed: {e}")
    import traceback

    traceback.print_exc()
    print(
        "Check the error message. Ensure the TorchScript model is valid and coremltools supports all operators."
    )
