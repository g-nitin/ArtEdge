import Accelerate
import Combine
import CoreML
import SwiftUI
import VideoToolbox
import Vision

class StyleTransferService: ObservableObject {

    @Published var styledImage: Image?
    @Published var processingTime: Double = 0.0
    @Published var error: Error?
    @Published var isModelLoaded: Bool = false
    @Published var isProcessing: Bool = false  // Track processing state

    private var mlModel: MLModel?
    // Store the dynamically determined names
    private var determinedContentInputName: String?
    private var determinedStyleInputName: String?
    private var determinedOutputName: String?

    private let processingQueue = DispatchQueue(
        label: "com.artedge.processingQueue", qos: .userInitiated)

    // Enum to represent the expected input type
    enum ModelInputType {
        case unknown
        case multiArray
        case image  // Represents CVPixelBuffer
    }

    private var expectedInputType: ModelInputType = .unknown

    // TODO: Target size might ALSO need to be dynamic if models expect different input dimensions.
    // For now, we'll assume 224x224, but this is a potential future enhancement.
    // We could inspect the input MLFeatureDescription's imageConstraint or multiArrayConstraint.
    private let targetSize = CGSize(width: 224, height: 224)

    private var styleMultiArray: MLMultiArray?  // Keep for models needing MLMultiArray
    private var stylePixelBuffer: CVPixelBuffer?  // Add for models needing CVPixelBuffer
    private var currentStyleName: String?

    // Initialize without loading a default model immediately, or load a default one passed from outside
    init(modelName: String? = nil) {  // Allow optional initial model name
        if let initialModel = modelName {
            print("Initializing StyleTransferService with model: \(initialModel)")
            // Use a background thread for initial load to avoid blocking UI thread if complex
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                self?.loadModel(named: initialModel)
            }
        } else {
            print("Initializing StyleTransferService without an initial model.")
            // isModelLoaded will remain false until loadModel or switchModel is called
        }
    }

    private func loadModel(named modelName: String) {
        // Make sure we are on a background thread potentially? Or ensure caller handles it.
        // For simplicity, let's assume it's called appropriately.
        print("Attempting to load model: \(modelName)")
        guard let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc")
        else {
            print(
                "🔴 Error: Could not find compiled model '\(modelName).mlmodelc'. Ensure it's added to the Xcode project target and the name is correct."
            )
            DispatchQueue.main.async {
                self.error = StyleTransferError.modelFileNotFound(modelName)  // More specific error
                self.isModelLoaded = false
                self.clearModelData()  // Helper to clear internal state
            }
            return
        }

        do {
            let config = MLModelConfiguration()
            // config.computeUnits = .all // Default: Use Neural Engine if available
            // config.computeUnits = .cpuAndGPU // Alternative
            // config.computeUnits = .cpuOnly // For debugging

            let loadedModel = try MLModel(contentsOf: modelURL, configuration: config)
            let description = loadedModel.modelDescription
            print("🟢 Successfully loaded model: \(modelName). Inspecting description...")

            // --- Dynamic Input Name & Type Determination ---
            var foundContentName: String? = nil
            var foundStyleName: String? = nil
            var detectedInputType: ModelInputType = .unknown  // Local variable for detection

            let multiArrayInputs = description.inputDescriptionsByName.filter {
                $1.type == .multiArray
            }
            let imageInputs = description.inputDescriptionsByName.filter { $1.type == .image }

            print("ℹ️ Found \(multiArrayInputs.count) MLMultiArray inputs: \(multiArrayInputs.keys)")
            print("ℹ️ Found \(imageInputs.count) Image (CVPixelBuffer) inputs: \(imageInputs.keys)")

            // --- Check for MLMultiArray Inputs First (Current Pipeline Preference) ---
            if multiArrayInputs.count == 2 {
                detectedInputType = .multiArray
                print("✅ Detected requirement for MLMultiArray inputs.")
                // Try specific names first
                if let contentDesc = multiArrayInputs["content_image"] {
                    foundContentName = contentDesc.name
                }
                if let styleDesc = multiArrayInputs["style_image"] {
                    foundStyleName = styleDesc.name
                }

                // If specific names not found, infer based on sorted order
                if foundContentName == nil || foundStyleName == nil {
                    print(
                        "⚠️ Specific MLMultiArray names ('content_image', 'style_image') not found. Inferring based on order."
                    )
                    let sortedNames = multiArrayInputs.keys.sorted()
                    foundContentName = sortedNames[0]
                    foundStyleName = sortedNames[1]
                    print(
                        "⚠️ Inferred MLMultiArray inputs: Content='\(foundContentName!)', Style='\(foundStyleName!)'. Verify."
                    )
                }
            }
            // --- Else, Check for Image (CVPixelBuffer) Inputs ---
            else if imageInputs.count == 2 {
                detectedInputType = .image
                print("✅ Detected requirement for Image (CVPixelBuffer) inputs.")
                // Try specific names first (allow different names like 'content', 'style')
                if let contentDesc = imageInputs["content"] {
                    foundContentName = contentDesc.name
                }  // Common alternative name
                else if let contentDesc = imageInputs["content_image"] {
                    foundContentName = contentDesc.name
                }

                if let styleDesc = imageInputs["style"] {
                    foundStyleName = styleDesc.name
                }  // Common alternative name
                else if let styleDesc = imageInputs["style_image"] {
                    foundStyleName = styleDesc.name
                }

                // If specific names not found, infer based on sorted order
                if foundContentName == nil || foundStyleName == nil {
                    print(
                        "⚠️ Specific Image names ('content'/'content_image', 'style'/'style_image') not found. Inferring based on order."
                    )
                    let sortedNames = imageInputs.keys.sorted()
                    // Be careful with inference - maybe 'content' should come first?
                    // Let's assume alphabetical for now, but this might need adjustment based on models.
                    foundContentName = sortedNames[0]
                    foundStyleName = sortedNames[1]
                    print(
                        "⚠️ Inferred Image inputs: Content='\(foundContentName!)', Style='\(foundStyleName!)'. Verify."
                    )
                }
            }
            // --- Else, Input configuration is unexpected ---
            else {
                print(
                    "🔴 Error: Unexpected input configuration. Expected 2 MLMultiArray inputs OR 2 Image inputs."
                )
                print(
                    "🔴 Found \(multiArrayInputs.count) MLMultiArray and \(imageInputs.count) Image inputs."
                )
                DispatchQueue.main.async {
                    self.error = StyleTransferError.inputMismatch(
                        "Expected 2 MLMultiArray OR 2 Image inputs. Found M:\(multiArrayInputs.count), I:\(imageInputs.count)"
                    )
                    self.isModelLoaded = false
                    self.clearModelData()
                }
                return
            }

            // Ensure both names were found
            guard let finalContentName = foundContentName, let finalStyleName = foundStyleName
            else {
                print(
                    "🔴 Error: Failed to assign both content and style input names for detected type \(detectedInputType)."
                )
                DispatchQueue.main.async {
                    self.error = StyleTransferError.inputNameNotDetermined(
                        "Could not resolve both input names for type \(detectedInputType)")
                    self.isModelLoaded = false
                    self.clearModelData()
                }
                return
            }

            print("ℹ️ Using Content Input: '\(finalContentName)' (Type: \(detectedInputType))")
            print("ℹ️ Using Style Input: '\(finalStyleName)' (Type: \(detectedInputType))")

            // --- Dynamic Output Name Determination (Keep as is, assuming MLMultiArray output) ---
            let multiArrayOutputs = description.outputDescriptionsByName.filter {
                $1.type == .multiArray
            }
            print(
                "ℹ️ Found \(multiArrayOutputs.count) MLMultiArray outputs: \(multiArrayOutputs.keys)"
            )

            var foundOutputName: String? = nil
            if multiArrayOutputs.count == 1 {
                foundOutputName = multiArrayOutputs.first!.value.name
                print("ℹ️ Using Output: '\(foundOutputName!)'")
            } else {
                // If the model outputs an Image instead, adapt this section too
                let imageOutputs = description.outputDescriptionsByName.filter { $1.type == .image }
                if imageOutputs.count == 1 && multiArrayOutputs.isEmpty {
                    // TODO: Handle Image output - requires different post-processing
                    print(
                        "🔴 Error: Model outputs an Image (CVPixelBuffer), but post-processing expects MLMultiArray. Output handling adaptation needed."
                    )
                    DispatchQueue.main.async {
                        self.error = StyleTransferError.outputTypeMismatch(
                            "Expected MLMultiArray output, found Image output.")
                        self.isModelLoaded = false
                        self.clearModelData()
                    }
                    return
                } else {
                    print(
                        "🔴 Error: Expected exactly 1 MLMultiArray output, but found \(multiArrayOutputs.count) (and \(imageOutputs.count) Image outputs)."
                    )
                    DispatchQueue.main.async {
                        self.error = StyleTransferError.outputNameNotDetermined(
                            "Expected 1 MLMultiArray output, found \(multiArrayOutputs.count)")
                        self.isModelLoaded = false
                        self.clearModelData()
                    }
                    return
                }
            }

            // --- Success: Assign properties ---
            DispatchQueue.main.async {
                self.mlModel = loadedModel
                self.determinedContentInputName = finalContentName
                self.determinedStyleInputName = finalStyleName
                self.determinedOutputName = foundOutputName
                self.expectedInputType = detectedInputType  // Store the detected type
                self.isModelLoaded = true
                self.error = nil
                print("✅ Model '\(modelName)' is ready. Expecting \(detectedInputType) inputs.")
            }

        } catch {
            // ... (error handling remains the same) ...
            print("🔴 Error loading Core ML model '\(modelName)': \(error)")
            DispatchQueue.main.async {
                self.error = error
                self.isModelLoaded = false
                self.clearModelData()
            }
        }
    }

    // Helper to clear model-specific data
    private func clearModelData() {
        self.mlModel = nil
        self.determinedContentInputName = nil
        self.determinedStyleInputName = nil
        self.determinedOutputName = nil
        self.expectedInputType = .unknown  // Reset expected type
        // Decide whether to clear the style image too, as it might be incompatible
        // self.styleMultiArray = nil
        // self.currentStyleName = nil
    }

    // Load and Preprocess Style Image - Called when user selects a style
    func loadStyleImage(named imageName: String) {
        guard imageName != currentStyleName else {
            print("ℹ️ Style '\(imageName)' is already loaded.")
            // Ensure error state is consistent if style was previously missing
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                if let currentError = self.error as? StyleTransferError,
                    currentError == .styleImageNotSet
                {
                    self.error = nil
                }
            }
            return
        }

        print("🔄 Loading style image: \(imageName)...")
        guard let uiImage = UIImage(named: imageName) else {
            print("🔴 Error: Could not load style image named '\(imageName)'.")
            DispatchQueue.main.async { [weak self] in
                self?.error = StyleTransferError.styleImageNotFound
                self?.styleMultiArray = nil
                self?.stylePixelBuffer = nil  // Clear pixel buffer too
                self?.currentStyleName = nil
            }
            return
        }

        // 1. Preprocess to CVPixelBuffer (This is the common step)
        // Use the *same* preprocessing as the content image for consistency
        guard
            let buffer = preprocessImagePyTorchStyle(
                image: uiImage, targetSize: Int(targetSize.width))
        else {
            print("🔴 Error: Failed to preprocess style image '\(imageName)' to CVPixelBuffer.")
            DispatchQueue.main.async { [weak self] in
                self?.error = StyleTransferError.styleImageProcessingFailed
                self?.styleMultiArray = nil
                self?.stylePixelBuffer = nil  // Clear pixel buffer too
                self?.currentStyleName = nil
            }
            return
        }

        // Store the CVPixelBuffer
        self.stylePixelBuffer = buffer
        print("🟢 Style image '\(imageName)' preprocessed to CVPixelBuffer.")

        // 2. Convert CVPixelBuffer to MLMultiArray (for models that need it)
        // This conversion includes normalization.
        guard let multiArray = mlMultiArray(from: buffer) else {
            print("🔴 Error: Failed to convert style CVPixelBuffer to MLMultiArray.")
            DispatchQueue.main.async { [weak self] in
                // This might be acceptable if no models need MLMultiArray, but log it.
                // Consider if this should be a fatal error for style loading.
                // For now, let's allow loading to succeed if buffer is okay, but log error.
                print(
                    "⚠️ Warning: Could not create MLMultiArray for style. Models needing MLMultiArray input will fail."
                )
                // self?.error = StyleTransferError.styleImageProcessingFailed
                self?.styleMultiArray = nil  // Ensure it's nil
                // Don't clear currentStyleName or pixelBuffer here if buffer is valid
            }
            // Continue even if multi-array fails, as pixel buffer might be usable
            return
        }

        // Store the MLMultiArray (if conversion succeeded)
        if multiArray != nil {
            self.styleMultiArray = multiArray
            print("🟢 Style CVPixelBuffer also converted to MLMultiArray.")
        }

        // Update state
        self.currentStyleName = imageName
        DispatchQueue.main.async { [weak self] in
            // Clear error only if it was styleImageNotSet previously
            if let currentError = self?.error as? StyleTransferError,
                currentError == .styleImageNotSet
            {
                self?.error = nil
            }
        }
    }

    func process(contentImage: UIImage) {
        guard let model = mlModel, isModelLoaded,
            let contentInputName = self.determinedContentInputName,
            let styleInputName = self.determinedStyleInputName,
            let outputName = self.determinedOutputName,
            expectedInputType != .unknown  // Ensure type was determined
        else {
            print("🔴 Error: Model not loaded or input/output names/type not determined.")
            DispatchQueue.main.async {
                // Provide more specific error
                if !self.isModelLoaded {
                    self.error = StyleTransferError.modelNotLoaded
                } else if self.expectedInputType == .unknown {
                    self.error = StyleTransferError.inputMismatch(
                        "Input type could not be determined during load.")
                } else {
                    self.error = StyleTransferError.modelNotLoaded
                }  // Fallback
                self.isProcessing = false
            }
            return
        }

        // --- Style Input Handling ---
        // Style image is ALWAYS preprocessed to MLMultiArray currently via loadStyleImage.
        // If a model needs CVPixelBuffer for style, loadStyleImage needs adaptation too.
        // For now, assume style is MLMultiArray OR we adapt loadStyleImage later.
        // Let's assume for THIS specific AdaIn model, style is ALSO CVPixelBuffer.
        // We need to preprocess the style image to CVPixelBuffer ON DEMAND here,
        // OR change loadStyleImage to store the CVPixelBuffer instead of MLMultiArray.
        // Let's modify loadStyleImage slightly first.

        // --- Content Input Handling ---
        // Preprocess content image to CVPixelBuffer (needed for both paths)
        guard
            let contentPixelBuffer = self.preprocessImagePyTorchStyle(
                image: contentImage, targetSize: Int(self.targetSize.width))
        else {
            print("🔴 Failed to preprocess content UIImage to CVPixelBuffer.")
            DispatchQueue.main.async {
                self.error = StyleTransferError.inputResizeFailed
                self.isProcessing = false
            }
            return
        }

        // Indicate processing started
        DispatchQueue.main.async {
            self.isProcessing = true
            self.styledImage = nil
            self.error = nil  // Clear previous errors before starting
        }

        let startTime = Date()

        processingQueue.async { [weak self] in
            guard let self = self else { return }

            // --- Create Input Provider based on Expected Type ---
            var inputFeatures: [String: Any] = [:]
            var inputProviderCreationFailed = false

            switch self.expectedInputType {
            case .multiArray:
                // Convert content CVPixelBuffer to MLMultiArray
                guard let contentInputArray = self.mlMultiArray(from: contentPixelBuffer) else {
                    print("🔴 Failed to convert content CVPixelBuffer to MLMultiArray.")
                    DispatchQueue.main.async {
                        self.error = StyleTransferError.inputConversionFailed
                        self.isProcessing = false
                    }
                    return  // Exit async block
                }
                // Get pre-converted style MLMultiArray
                guard let styleInputArray = self.styleMultiArray else {
                    print("🔴 Style MLMultiArray not available.")
                    DispatchQueue.main.async {
                        self.error = StyleTransferError.styleImageNotSet  // Or a different error
                        self.isProcessing = false
                    }
                    return  // Exit async block
                }
                inputFeatures[contentInputName] = contentInputArray
                inputFeatures[styleInputName] = styleInputArray
                print("ℹ️ Providing MLMultiArray inputs to model.")

            case .image:
                // Use CVPixelBuffer directly for content
                inputFeatures[contentInputName] = contentPixelBuffer

                // --- Style Input for Image type ---
                // We need the style image as CVPixelBuffer too.
                // Let's retrieve/create it here. Requires stylePixelBuffer property.
                guard let styleBuffer = self.stylePixelBuffer else {  // Assumes stylePixelBuffer exists
                    print("🔴 Style CVPixelBuffer not available.")
                    DispatchQueue.main.async {
                        self.error = StyleTransferError.styleImageNotSet  // Or a different error
                        self.isProcessing = false
                    }
                    return  // Exit async block
                }
                inputFeatures[styleInputName] = styleBuffer
                print("ℹ️ Providing CVPixelBuffer inputs to model.")

            case .unknown:
                // Should have been caught by the initial guard, but handle defensively
                print("🔴 Error: Expected input type is unknown during processing.")
                inputProviderCreationFailed = true  // Mark failure
            }

            // Check if input preparation failed
            guard !inputProviderCreationFailed else {
                DispatchQueue.main.async {
                    if self.error == nil {  // Set error if not already set
                        self.error = StyleTransferError.inputProviderCreationFailed
                    }
                    self.isProcessing = false
                }
                return  // Exit async block
            }

            // Create the actual feature provider
            guard let inputProvider = try? MLDictionaryFeatureProvider(dictionary: inputFeatures)
            else {
                print(
                    "🔴 Failed to create MLDictionaryFeatureProvider. Check input names and types.")
                DispatchQueue.main.async {
                    self.error = StyleTransferError.inputProviderCreationFailed
                    self.isProcessing = false
                }
                return
            }

            // --- Perform Prediction (Remains the same, assuming MLMultiArray output) ---
            do {
                let prediction = try model.prediction(from: inputProvider)

                // Extract Output (Still assuming MLMultiArray output)
                guard
                    let multiArrayOutput = prediction.featureValue(for: outputName)?.multiArrayValue
                else {
                    print(
                        "🔴 Failed to get MLMultiArray output using determined name '\(outputName)'."
                    )
                    // Check if the output was actually an Image
                    if let imageOutput = prediction.featureValue(for: outputName)?.imageBufferValue
                    {
                        print(
                            "ℹ️ Model actually outputted an Image (CVPixelBuffer). Post-processing needs update."
                        )
                        // TODO: Implement CVPixelBuffer to UIImage conversion here
                        DispatchQueue.main.async {
                            self.error = StyleTransferError.outputTypeMismatch(
                                "Expected MLMultiArray output, got Image.")
                            self.isProcessing = false
                        }
                    } else {
                        DispatchQueue.main.async {
                            self.error = StyleTransferError.unexpectedResultType
                            self.isProcessing = false
                        }
                    }
                    return
                }

                // Convert MLMultiArray output to Image
                let convertedImage = self.imageFromMultiArray(multiArray: multiArrayOutput)
                let endTime = Date()
                let timeInterval = endTime.timeIntervalSince(startTime) * 1000

                // Update UI
                DispatchQueue.main.async {
                    if let img = convertedImage {
                        self.styledImage = Image(uiImage: img)
                        self.error = nil
                    } else {
                        print("🔴 Failed to convert MultiArray output to image.")
                        self.styledImage = nil
                        self.error = StyleTransferError.multiArrayConversionFailed
                    }
                    self.processingTime = timeInterval
                    self.isProcessing = false
                }

            } catch {
                print("🔴 Failed to perform prediction: \(error)")
                DispatchQueue.main.async {
                    self.error = error
                    self.styledImage = nil
                    self.isProcessing = false
                }
            }
        }  // End of processingQueue.async
    }  // End of process function

    func switchModel(to modelName: String) {
        print("🔄 Request received to switch model to: \(modelName)")
        // Reset state before loading the new model
        // Perform UI updates on the main thread
        DispatchQueue.main.async {
            self.isModelLoaded = false
            self.styledImage = nil
            self.error = StyleTransferError.modelLoading  // Indicate loading
            self.isProcessing = false
            // self.processingTime = 0.0 // Optional reset
        }

        // Clear internal model-specific properties immediately
        self.clearModelData()  // Use the helper function

        // Load the new model asynchronously
        // Wrap loadModel call in a background queue if it's not already async
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.loadModel(named: modelName)
        }
    }

    // Preprocessing Function (Mimics PyTorch Resize(shorter_edge) + CenterCrop)
    private func preprocessImagePyTorchStyle(image: UIImage, targetSize: Int) -> CVPixelBuffer? {
        let originalSize = image.size
        // Define target size in PIXELS
        let targetPixelCGSize = CGSize(width: targetSize, height: targetSize)  // Target pixel size object
        print(
            "➡️ Preprocessing: Original size \(originalSize), Target Pixel Size \(targetPixelCGSize)"
        )

        // 1. Calculate intermediate size (ensure integer dimensions)
        // This calculation should resize based on the target *pixel* size
        let intermediateSize: CGSize
        let scaleFactor: CGFloat  // Use a different name to avoid confusion with UIImage.scale
        if originalSize.width < originalSize.height {
            // Width is shorter edge (in points)
            // Calculate scale needed to make pixel width equal targetSize
            scaleFactor = CGFloat(targetSize) / (originalSize.width * image.scale)  // Scale based on original pixels
            // Intermediate size in points
            intermediateSize = CGSize(
                width: originalSize.width * scaleFactor, height: originalSize.height * scaleFactor)

        } else {
            // Height is shorter or equal edge (in points)
            // Calculate scale needed to make pixel height equal targetSize
            scaleFactor = CGFloat(targetSize) / (originalSize.height * image.scale)  // Scale based on original pixels
            // Intermediate size in points
            intermediateSize = CGSize(
                width: originalSize.width * scaleFactor, height: originalSize.height * scaleFactor)
        }
        // Round the intermediate size for the renderer
        let roundedIntermediateSize = CGSize(
            width: round(intermediateSize.width), height: round(intermediateSize.height))
        print("➡️ Preprocessing: Calculated intermediate size (points) \(roundedIntermediateSize)")

        // 2. Resize to intermediate size (in points)
        guard let resizedImage = image.resize(to: roundedIntermediateSize) else {
            print(
                "🔴 Preprocessing: Failed to resize image to intermediate size \(roundedIntermediateSize)"
            )
            return nil
        }
        // Log the size of the resized image (points) and its underlying pixel dimensions
        if let resizedCG = resizedImage.cgImage {
            print(
                "➡️ Preprocessing: Resized image size \(resizedImage.size) (points), Scale \(resizedImage.scale), Pixels \(resizedCG.width)x\(resizedCG.height)"
            )
        } else {
            print(
                "➡️ Preprocessing: Resized image size \(resizedImage.size) (points), Scale \(resizedImage.scale), Could not get CGImage"
            )
        }

        // 3. Center Crop to target PIXEL size
        guard let croppedImage = resizedImage.centerCrop(to: targetPixelCGSize) else {  // Pass targetPixelCGSize
            print(
                "🔴 Preprocessing: Failed to center crop image to target pixel size \(targetPixelCGSize)"
            )
            return nil
        }
        // Log the size immediately after cropping (should be 224x224 due to scale: 1.0)
        print("➡️ Preprocessing: Cropped image size \(croppedImage.size)")

        // Check if cropped size is correct BEFORE pixel buffer conversion
        guard
            abs(croppedImage.size.width - targetPixelCGSize.width) < 0.1
                && abs(croppedImage.size.height - targetPixelCGSize.height) < 0.1
        else {
            print(
                "🔴 Preprocessing: Cropped image size \(croppedImage.size) is NOT the target pixel size \(targetPixelCGSize) before pixel buffer conversion."
            )
            return nil
        }

        // 4. Convert the final cropped image to CVPixelBuffer
        guard
            let pixelBuffer = pixelBufferFromImage(
                image: croppedImage, expectedSize: targetPixelCGSize)
        else {  // Pass targetPixelCGSize
            print("🔴 Preprocessing: Failed to convert final cropped image to CVPixelBuffer.")
            return nil
        }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        print("➡️ Preprocessing: Final pixel buffer size \(width)x\(height)")

        return pixelBuffer
    }

    // Convert UIImage to CVPixelBuffer (kCVPixelFormatType_32BGRA)
    // Added expectedSize check for safety.
    private func pixelBufferFromImage(image: UIImage, expectedSize: CGSize) -> CVPixelBuffer? {
        // Relax the check slightly to handle potential floating point inaccuracies in UIImage.size
        guard
            abs(image.size.width - expectedSize.width) < 0.1
                && abs(image.size.height - expectedSize.height) < 0.1
        else {
            print(
                "🔴 pixelBufferFromImage: Input image size \(image.size) does not match expected size \(expectedSize) (within tolerance)."
            )
            return nil
        }

        // Ensure we create the buffer with INTEGER dimensions
        let bufferWidth = Int(round(expectedSize.width))
        let bufferHeight = Int(round(expectedSize.height))

        let attrs =
            [
                kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue,
                kCVPixelBufferMetalCompatibilityKey: kCFBooleanTrue,
            ] as CFDictionary
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            bufferWidth,  // Use integer dimension
            bufferHeight,  // Use integer dimension
            kCVPixelFormatType_32BGRA,
            attrs,
            &pixelBuffer)

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            print("🔴 Failed CVPixelBufferCreate, status: \(status)")
            return nil
        }

        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        defer { CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0)) }

        guard let pixelData = CVPixelBufferGetBaseAddress(buffer) else {
            print("🔴 Failed to get base address of pixel buffer.")
            return nil
        }
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        guard
            let context = CGContext(
                data: pixelData,
                width: bufferWidth,  // Use integer dimension
                height: bufferHeight,  // Use integer dimension
                bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: rgbColorSpace,
                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
                    | CGBitmapInfo.byteOrder32Little.rawValue
            )
        else {
            print("🔴 Failed to create CGContext for pixel buffer conversion.")
            return nil
        }

        guard let cgImage = image.cgImage else {
            print("🔴 Failed to get CGImage from UIImage.")
            return nil
        }

        // Draw the image into the context, ensuring it fills the integer dimensions
        context.draw(
            cgImage, in: CGRect(x: 0, y: 0, width: bufferWidth, height: bufferHeight))

        return buffer
    }

    // MLMultiArray to UIImage Conversion
    private func imageFromMultiArray(multiArray: MLMultiArray) -> UIImage? {
        print("ℹ️ MLMultiArray - Shape: \(multiArray.shape), Data Type: \(multiArray.dataType)")

        guard multiArray.dataType == .float32 else {
            print("🔴 Error: Expected Float32 MLMultiArray, got \(multiArray.dataType)")
            return nil
        }
        // Expecting [Batch, Channels, Height, Width] or similar (at least 3 dims)
        guard multiArray.shape.count >= 3 else {
            print(
                "🔴 MultiArray shape count is less than 3 (\(multiArray.shape.count)). Cannot determine dimensions."
            )
            return nil
        }

        // Determine dimensions assuming [Batch, Channels, Height, Width] layout
        // This matches typical PyTorch -> Core ML conversion for CNNs
        let channels = multiArray.shape[multiArray.shape.count - 3].intValue
        let height = multiArray.shape[multiArray.shape.count - 2].intValue
        let width = multiArray.shape[multiArray.shape.count - 1].intValue

        print(
            "ℹ️ Interpreted Dimensions - Channels: \(channels), Height: \(height), Width: \(width)")

        // Check if dimensions are reasonable
        guard channels == 3 && height > 0 && width > 0 else {
            print(
                "🔴 MultiArray dimensions are invalid (Channels: \(channels), Height: \(height), Width: \(width)). Cannot create RGB image."
            )
            return nil
        }

        let dataPointer = multiArray.dataPointer.bindMemory(
            to: Float32.self, capacity: channels * height * width)

        // Define normalization constants (must match Python input normalization)
        let mean: [Float] = [0.485, 0.456, 0.406]
        let std: [Float] = [0.229, 0.224, 0.225]

        // Prepare buffer for RGBA output image data
        let bytesPerRow = width * 4  // 4 bytes per pixel (RGBA)
        var pixelData = [UInt8](repeating: 0, count: height * bytesPerRow)

        print("--- Debug Pixel Values (Denormalized Float 0-1) ---")
        logPixelValue(
            x: 0, y: 0, width: width, height: height, channels: channels, dataPointer: dataPointer,
            std: std, mean: mean)  // Top-Left
        logPixelValue(
            x: width - 1, y: 0, width: width, height: height, channels: channels,
            dataPointer: dataPointer, std: std, mean: mean)  // Top-Right
        logPixelValue(
            x: 0, y: height - 1, width: width, height: height, channels: channels,
            dataPointer: dataPointer, std: std, mean: mean)  // Bottom-Left
        logPixelValue(
            x: width - 1, y: height - 1, width: width, height: height, channels: channels,
            dataPointer: dataPointer, std: std, mean: mean)  // Bottom-Right
        logPixelValue(
            x: width / 2, y: height / 2, width: width, height: height, channels: channels,
            dataPointer: dataPointer, std: std, mean: mean)  // Center
        print("----------------------------------------------------")

        // --- Image Creation Loop (Keep the channel swap active for now) ---
        for y in 0..<height {
            for x in 0..<width {
                // Calculate indices for R, G, B channels in the flat dataPointer
                // Assumes CHW layout: Index = (channel * height * width) + (row * width) + column
                let r_index = (0 * height * width) + (y * width) + x
                let g_index = (1 * height * width) + (y * width) + x
                let b_index = (2 * height * width) + (y * width) + x

                // Get normalized float values from the model output
                let r_norm_float = dataPointer[r_index]
                let g_norm_float = dataPointer[g_index]
                let b_norm_float = dataPointer[b_index]

                // Denormalize: value = (normalized_value * std) + mean
                let r_denorm_float = (r_norm_float * std[0]) + mean[0]
                let g_denorm_float = (g_norm_float * std[1]) + mean[1]
                let b_denorm_float = (b_norm_float * std[2]) + mean[2]

                // Clip to [0, 1] range
                let r_clipped_float = max(0.0, min(1.0, r_denorm_float))
                let g_clipped_float = max(0.0, min(1.0, g_denorm_float))
                let b_clipped_float = max(0.0, min(1.0, b_denorm_float))

                // Scale to [0, 255] and convert to UInt8
                let r_uint8 = UInt8(r_clipped_float * 255.0)
                let g_uint8 = UInt8(g_clipped_float * 255.0)
                let b_uint8 = UInt8(b_clipped_float * 255.0)

                // Calculate the index in the output pixelData buffer (RGBA)
                let pixelIndex = (y * bytesPerRow) + (x * 4)

                // SWAP R and B channels here
                pixelData[pixelIndex + 0] = b_uint8  // Assign Blue to Red position
                pixelData[pixelIndex + 1] = g_uint8  // Green remains Green
                pixelData[pixelIndex + 2] = r_uint8  // Assign Red to Blue position
                pixelData[pixelIndex + 3] = 255  // Alpha

                // Original assignment (commented out):
                // pixelData[pixelIndex + 0] = r_uint8  // R
                // pixelData[pixelIndex + 1] = g_uint8  // G
                // pixelData[pixelIndex + 2] = b_uint8  // B
                // pixelData[pixelIndex + 3] = 255      // A
            }
        }

        // Create CGImage from the denormalized and clipped pixel data
        guard let providerRef = CGDataProvider(data: Data(pixelData) as CFData) else {
            print("🔴 Failed to create CGDataProvider.")
            return nil
        }

        guard
            let cgImage = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 32,  // RGBA
                bytesPerRow: bytesPerRow,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),  // RGBA format
                provider: providerRef,
                decode: nil,
                shouldInterpolate: true,
                intent: .defaultIntent
            )
        else {
            print("🔴 Failed to create CGImage from pixel data.")
            return nil
        }

        // Create UIImage from CGImage
        return UIImage(cgImage: cgImage)
    }

    // Helper function for logging pixel values
    private func logPixelValue(
        x: Int, y: Int, width: Int, height: Int, channels: Int,
        dataPointer: UnsafeMutablePointer<Float32>, std: [Float], mean: [Float]
    ) {
        // Assuming CHW layout
        let r_index = (0 * height * width) + (y * width) + x
        let g_index = (1 * height * width) + (y * width) + x
        let b_index = (2 * height * width) + (y * width) + x

        // Check bounds (simple check)
        guard r_index >= 0 && r_index < channels * height * width,
            g_index >= 0 && g_index < channels * height * width,
            b_index >= 0 && b_index < channels * height * width
        else {
            print("Pixel (\(x), \(y)): Index out of bounds")
            return
        }

        let r_norm_float = dataPointer[r_index]
        let g_norm_float = dataPointer[g_index]
        let b_norm_float = dataPointer[b_index]

        // Denormalize
        let r_denorm_float = (r_norm_float * std[0]) + mean[0]
        let g_denorm_float = (g_norm_float * std[1]) + mean[1]
        let b_denorm_float = (b_norm_float * std[2]) + mean[2]

        // Clip
        let r_clipped_float = max(0.0, min(1.0, r_denorm_float))
        let g_clipped_float = max(0.0, min(1.0, g_denorm_float))
        let b_clipped_float = max(0.0, min(1.0, b_denorm_float))

        // Format for printing
        let r_str = String(format: "%.4f", r_clipped_float)
        let g_str = String(format: "%.4f", g_clipped_float)
        let b_str = String(format: "%.4f", b_clipped_float)

        print("Pixel (\(x), \(y)): RGB(\(r_str), \(g_str), \(b_str))")
    }

    // Converts CVPixelBuffer (BGRA) to MLMultiArray (Float32, CHW, Normalized)
    private func mlMultiArray(from pixelBuffer: CVPixelBuffer) -> MLMultiArray? {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        guard width == Int(targetSize.width) && height == Int(targetSize.height) else {
            print(
                "🔴 mlMultiArray: Input pixel buffer size \(width)x\(height) does not match target \(targetSize)"
            )
            return nil
        }

        // Create MLMultiArray
        guard
            let multiArray = try? MLMultiArray(
                shape: [1, 3, NSNumber(value: height), NSNumber(value: width)], dataType: .float32)
        else {
            print("🔴 mlMultiArray: Failed to create MLMultiArray")
            return nil
        }

        // Lock buffer for reading
        guard CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly) == kCVReturnSuccess else {
            print("🔴 mlMultiArray: Failed to lock pixel buffer base address")
            return nil
        }
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            print("🔴 mlMultiArray: Failed to get pixel buffer base address")
            return nil
        }

        // Get buffer details
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)

        // Get pointer to MLMultiArray data
        let multiArrayPointer = multiArray.dataPointer.bindMemory(
            to: Float32.self, capacity: 3 * height * width)

        // Normalization constants (MUST match Python preprocessing)
        let mean: [Float] = [0.485, 0.456, 0.406]  // RGB
        let std: [Float] = [0.229, 0.224, 0.225]  // RGB

        // Iterate through pixels and fill the MLMultiArray
        for y in 0..<height {
            for x in 0..<width {
                let pixelOffset = y * bytesPerRow + x * 4  // BGRA format

                // Read B, G, R values (UInt8)
                let b_uint8 = buffer[pixelOffset + 0]
                let g_uint8 = buffer[pixelOffset + 1]
                let r_uint8 = buffer[pixelOffset + 2]
                // let a_uint8 = buffer[pixelOffset + 3] // Alpha - ignored

                // Convert to Float32 [0.0, 1.0]
                let r_float = Float(r_uint8) / 255.0
                let g_float = Float(g_uint8) / 255.0
                let b_float = Float(b_uint8) / 255.0

                // Normalize: (value - mean) / std
                let r_norm = (r_float - mean[0]) / std[0]
                let g_norm = (g_float - mean[1]) / std[1]
                let b_norm = (b_float - mean[2]) / std[2]

                // Write to MLMultiArray (CHW layout)
                // Index = (channel * height * width) + (row * width) + column
                let r_index = (0 * height * width) + (y * width) + x
                let g_index = (1 * height * width) + (y * width) + x
                let b_index = (2 * height * width) + (y * width) + x

                multiArrayPointer[r_index] = r_norm
                multiArrayPointer[g_index] = g_norm
                multiArrayPointer[b_index] = b_norm
            }
        }

        return multiArray
    }

    // Define potential errors
    enum StyleTransferError: Error, LocalizedError, Equatable {
        case modelFileNotFound(String)  // Include filename
        case modelLoading
        case modelNotLoaded
        case styleImageNotFound
        case styleImageProcessingFailed
        case styleImageNotSet
        case inputResizeFailed
        case inputConversionFailed
        case inputProviderCreationFailed
        case unexpectedResultType
        case multiArrayConversionFailed
        case contentImageLoadFailed
        case outputNameNotDetermined(String)  // Include details
        case inputNameNotDetermined(String)  // Include details
        case inputMismatch(String)  // For cases where type is wrong (e.g., expected MultiArray, got Image)
        case outputTypeMismatch(String) // Added for Image output case

        var errorDescription: String? {
            switch self {
            case .modelFileNotFound(let name):
                return "Model file '\(name).mlmodelc' not found. Ensure it's added to the target."
            case .modelLoading:
                return "Loading selected model..."
            case .modelNotLoaded:
                return "Style transfer model could not be loaded or is not selected."
            case .styleImageNotFound: return "Style image file not found in assets."
            case .styleImageProcessingFailed: return "Could not resize/convert the style image."
            case .styleImageNotSet: return "Please select a style image first."
            case .inputResizeFailed: return "Failed to resize content image for model input."
            case .inputConversionFailed: return "Failed to convert content image to MLMultiArray."
            case .inputProviderCreationFailed:
                return "Failed to create input features for the model."
            case .unexpectedResultType:
                return "Model output was not the expected MLMultiArray format."
            case .multiArrayConversionFailed:
                return "Could not convert model output tensor to an image."
            case .contentImageLoadFailed: return "Failed to load the selected content image."
            case .outputNameNotDetermined(let details):
                return "Could not determine model output name. \(details)."
            case .inputNameNotDetermined(let details):
                return "Could not determine model input names. \(details)."
            case .inputMismatch(let details): return "Model input type mismatch. \(details)."
            case .outputTypeMismatch(let details): return "Model output type mismatch. \(details)."
            }
        }
    }
}

// UIImage Extensions for Resizing and Cropping
extension UIImage {
    // Resizes the image to a specified size (in points).
    // UIGraphicsImageRenderer handles scale correctly.
    func resize(to size: CGSize) -> UIImage? {
        // Ensure target size dimensions are at least 1x1
        guard size.width > 0 && size.height > 0 else {
            print("🔴 resize: Invalid target size \(size)")
            return nil
        }
        // Use UIGraphicsImageRenderer for high-quality resizing
        // It automatically handles scale, resulting in a UIImage whose size
        // is in points, but underlying CGImage has the target pixel dimensions.
        let renderer = UIGraphicsImageRenderer(size: size)
        return renderer.image { (context) in
            self.draw(in: CGRect(origin: .zero, size: size))
        }
    }

    // Crops the center of the image to the specified PIXEL size.
    func centerCrop(to pixelSize: CGSize) -> UIImage? {  // Parameter is target pixel size
        guard let cgImage = self.cgImage else {
            print("🔴 centerCrop: Failed to get cgImage.")
            return nil
        }

        // Get actual pixel dimensions of the source CGImage
        let sourcePixelWidth = CGFloat(cgImage.width)
        let sourcePixelHeight = CGFloat(cgImage.height)

        // Target crop size IS the pixelSize passed in
        let targetPixelWidth = pixelSize.width
        let targetPixelHeight = pixelSize.height

        // Ensure crop size is not larger than the source image in pixels and positive
        // Also ensure target size is positive
        guard targetPixelWidth > 0 && targetPixelHeight > 0 else {
            print("🔴 centerCrop: Invalid target pixel size \(pixelSize)")
            return nil
        }
        let cropPixelWidth = min(targetPixelWidth, sourcePixelWidth)
        let cropPixelHeight = min(targetPixelHeight, sourcePixelHeight)

        // Calculate origin in pixels, centering the targetPixelWidth/Height within the sourcePixelWidth/Height
        let x = floor((sourcePixelWidth - cropPixelWidth) / 2.0)
        let y = floor((sourcePixelHeight - cropPixelHeight) / 2.0)

        // Define the crop rectangle in the source image's pixel coordinates
        let cropRect = CGRect(x: x, y: y, width: cropPixelWidth, height: cropPixelHeight)
        print(
            "➡️ centerCrop: Source Pixels \(sourcePixelWidth)x\(sourcePixelHeight), Target Pixels \(targetPixelWidth)x\(targetPixelHeight), CropRect \(cropRect)"
        )

        // Perform the crop on the CGImage
        guard let croppedCGImage = cgImage.cropping(to: cropRect) else {
            print("🔴 centerCrop: cgImage.cropping failed for rect \(cropRect)")
            return nil
        }
        print("➡️ centerCrop: Cropped CGImage size \(croppedCGImage.width)x\(croppedCGImage.height)")  // Should be targetPixelWidth x targetPixelHeight

        // Create a new UIImage from the cropped CGImage, setting scale to 1.0
        // so that UIImage.size reports the correct pixel dimensions.
        let croppedUIImage = UIImage(
            cgImage: croppedCGImage, scale: 1.0, orientation: self.imageOrientation)
        print("➡️ centerCrop: Final Cropped UIImage size \(croppedUIImage.size)")  // Should now be targetPixelWidth x targetPixelHeight
        return croppedUIImage
    }
}

// Helper for CIContext
extension CIContext {
    static let shared = CIContext()
}
