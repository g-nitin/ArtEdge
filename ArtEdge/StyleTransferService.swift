import Accelerate
import Combine
import CoreImage
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
    @Published var isSingleInputModel: Bool = false  // Track if the loaded model is single-input

    private var mlModel: MLModel?
    // Store the dynamically determined names
    private var determinedContentInputName: String?
    private var determinedStyleInputName: String?  // Will be nil for single-input models
    private var determinedOutputName: String?

    // Enum to represent the expected input type for content
    enum ModelInputType {
        case unknown
        case multiArray
        case image  // CVPixelBuffer
    }

    private var expectedInputType: ModelInputType = .unknown

    // Enum to represent the model structure regarding inputs
    enum ModelStructureType {
        case unknown
        case singleInput
        case dualInput
    }
    private var currentModelStructure: ModelStructureType = .unknown

    // Output Type Tracking
    enum ModelOutputType {
        case unknown
        case multiArray
        case image  // CVPixelBuffer
    }
    private var expectedOutputType: ModelOutputType = .unknown  // Track expected output type

    // Stores the expected input dimensions (width, height) required by the loaded model.
    // Optional because it's determined during model loading.
    private var modelInputSize: CGSize?

    private let processingQueue = DispatchQueue(
        label: "com.artedge.processingQueue", qos: .userInitiated)

    // Style data (only relevant for dual-input models)
    private var styleMultiArray: MLMultiArray?
    private var stylePixelBuffer: CVPixelBuffer?
    private var currentStyleName: String?

    // Shared CIContext for CVPixelBuffer to UIImage conversion
    private let ciContext = CIContext()

    init(modelName: String? = nil) {
        if let initialModel = modelName {
            print("Initializing StyleTransferService with model: \(initialModel)")
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                self?.loadModel(named: initialModel)
            }
        } else {
            print("Initializing StyleTransferService without an initial model.")
        }
    }

    private func loadModel(named modelName: String) {
        print("Attempting to load model: \(modelName)")
        guard let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc")
        else {
            print("🔴 Error: Could not find compiled model '\(modelName).mlmodelc'.")
            DispatchQueue.main.async {
                self.error = StyleTransferError.modelFileNotFound(modelName)
                self.isModelLoaded = false
                self.clearModelData()
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

            // Dynamic Input Determination
            var foundContentName: String? = nil
            var foundStyleName: String? = nil  // Will remain nil for single-input
            var detectedInputType: ModelInputType = .unknown
            var detectedModelStructure: ModelStructureType = .unknown

            let multiArrayInputs = description.inputDescriptionsByName.filter {
                $1.type == .multiArray
            }
            let imageInputs = description.inputDescriptionsByName.filter { $1.type == .image }

            print("ℹ️ Found \(multiArrayInputs.count) MLMultiArray inputs: \(multiArrayInputs.keys)")
            print("ℹ️ Found \(imageInputs.count) Image (CVPixelBuffer) inputs: \(imageInputs.keys)")

            // Case 1: Single Input Model
            if multiArrayInputs.count == 1 && imageInputs.isEmpty {
                detectedModelStructure = .singleInput
                detectedInputType = .multiArray
                foundContentName = multiArrayInputs.first!.key  // Assume the single input is content
                print("✅ Detected SINGLE input model requiring MLMultiArray.")
            } else if imageInputs.count == 1 && multiArrayInputs.isEmpty {
                detectedModelStructure = .singleInput
                detectedInputType = .image
                foundContentName = imageInputs.first!.key  // Assume the single input is content
                print("✅ Detected SINGLE input model requiring Image (CVPixelBuffer).")
            }
            // Case 2: Dual Input Model (MLMultiArray)
            else if multiArrayInputs.count == 2 && imageInputs.isEmpty {
                detectedModelStructure = .dualInput
                detectedInputType = .multiArray
                print("✅ Detected DUAL input model requiring MLMultiArray.")
                // Infer names (same logic as before)
                foundContentName =
                    multiArrayInputs["content_image"]?.name ?? multiArrayInputs["content"]?.name
                foundStyleName =
                    multiArrayInputs["style_image"]?.name ?? multiArrayInputs["style"]?.name
                if foundContentName == nil || foundStyleName == nil {
                    print(
                        "⚠️ Specific MLMultiArray names not found. Inferring based on sorted order.")
                    let sortedNames = multiArrayInputs.keys.sorted()
                    foundContentName = sortedNames[0]
                    foundStyleName = sortedNames[1]
                    print(
                        "⚠️ Inferred MLMultiArray inputs: Content='\(foundContentName!)', Style='\(foundStyleName!)'. Verify."
                    )
                }
            }
            // Case 3: Dual Input Model (Image/CVPixelBuffer)
            else if imageInputs.count == 2 && multiArrayInputs.isEmpty {
                detectedModelStructure = .dualInput
                detectedInputType = .image
                print("✅ Detected DUAL input model requiring Image (CVPixelBuffer).")
                // Infer names
                foundContentName =
                    imageInputs["content_image"]?.name ?? imageInputs["content"]?.name
                foundStyleName = imageInputs["style_image"]?.name ?? imageInputs["style"]?.name
                if foundContentName == nil || foundStyleName == nil {
                    print("⚠️ Specific Image names not found. Inferring based on sorted order.")
                    let sortedNames = imageInputs.keys.sorted()
                    foundContentName = sortedNames[0]
                    foundStyleName = sortedNames[1]
                    print(
                        "⚠️ Inferred Image inputs: Content='\(foundContentName!)', Style='\(foundStyleName!)'. Verify."
                    )
                }
            }
            // Case 4: Unsupported Input Configuration
            else {
                print(
                    "🔴 Error: Unsupported input configuration. Expected 1 OR 2 inputs (either MLMultiArray or Image type)."
                )
                print(
                    "🔴 Found \(multiArrayInputs.count) MLMultiArray and \(imageInputs.count) Image inputs."
                )
                DispatchQueue.main.async {
                    self.error = StyleTransferError.inputMismatch(
                        "Expected 1 or 2 inputs (MLMultiArray or Image). Found M:\(multiArrayInputs.count), I:\(imageInputs.count)"
                    )
                    self.isModelLoaded = false
                    self.clearModelData()
                }
                return
            }

            // Validate Found Names
            guard let finalContentName = foundContentName else {
                print(
                    "🔴 Error: Failed to assign content input name for detected structure \(detectedModelStructure)."
                )
                DispatchQueue.main.async {
                    self.error = StyleTransferError.inputNameNotDetermined(
                        "Could not resolve content input name.")
                    self.isModelLoaded = false
                    self.clearModelData()
                }
                return
            }
            // Style name is only required for dual input models
            if detectedModelStructure == .dualInput && foundStyleName == nil {
                print("🔴 Error: Failed to assign style input name for DUAL input model.")
                DispatchQueue.main.async {
                    self.error = StyleTransferError.inputNameNotDetermined(
                        "Could not resolve style input name for dual-input model.")
                    self.isModelLoaded = false
                    self.clearModelData()
                }
                return
            }

            print("ℹ️ Using Content Input: '\(finalContentName)' (Type: \(detectedInputType))")
            if let finalStyleName = foundStyleName {  // Only print if it exists
                print("ℹ️ Using Style Input: '\(finalStyleName)' (Type: \(detectedInputType))")
            } else {
                print("ℹ️ No Style Input required (Single Input Model).")
            }

            var determinedInputSize: CGSize? = nil
            if let inputDesc = description.inputDescriptionsByName[finalContentName] {
                switch detectedInputType {
                case .image:
                    let constraint = inputDesc.imageConstraint
                    if constraint != nil && constraint!.pixelsWide > 0 && constraint!.pixelsHigh > 0
                    {
                        determinedInputSize = CGSize(
                            width: constraint!.pixelsWide, height: constraint!.pixelsHigh)
                        print(
                            "✅ Detected Image Input Size: \(determinedInputSize!.width) x \(determinedInputSize!.height)"
                        )
                    } else {
                        print(
                            "⚠️ Could not determine input size from Image constraint for '\(finalContentName)'."
                        )
                    }
                case .multiArray:
                    let constraint = inputDesc.multiArrayConstraint
                    // Assuming shape is [Batch, Channels, Height, Width] or similar (at least 4 dims)
                    if constraint != nil && constraint!.shape.count >= 4 {
                        let heightIndex = constraint!.shape.count - 2
                        let widthIndex = constraint!.shape.count - 1
                        let height = constraint!.shape[heightIndex].intValue
                        let width = constraint!.shape[widthIndex].intValue
                        if width > 0 && height > 0 {
                            determinedInputSize = CGSize(width: width, height: height)
                            print("✅ Detected MLMultiArray Input Size (HxW): \(height) x \(width)")
                        } else {
                            print(
                                "⚠️ Could not determine valid input size from MLMultiArray shape constraint for '\(finalContentName)': \(constraint!.shape)"
                            )
                        }
                    } else {
                        print(
                            "⚠️ Could not determine input size from MLMultiArray constraint (shape invalid or missing) for '\(finalContentName)'. Shape: \(constraint?.shape ?? [])"
                        )
                    }
                case .unknown:
                    print("⚠️ Cannot determine input size: Input type is unknown.")
                }
            } else {
                print(
                    "🔴 Error: Could not find input description for determined content input name '\(finalContentName)'."
                )
                // Handle error - maybe set a default size with a warning, or fail loading? Let's fail for now.
                DispatchQueue.main.async {
                    self.error = StyleTransferError.inputNameNotDetermined(
                        "Could not find description for input '\(finalContentName)' to determine size."
                    )
                    self.isModelLoaded = false
                    self.clearModelData()
                }
                return
            }

            // Ensure we determined a size
            guard let finalInputSize = determinedInputSize else {
                print("🔴 Error: Failed to determine model input size. Cannot proceed.")
                DispatchQueue.main.async {
                    self.error = StyleTransferError.inputMismatch(
                        "Failed to determine required input dimensions from model.")
                    self.isModelLoaded = false
                    self.clearModelData()
                }
                return
            }

            // Dynamic Output Name & Type Determination
            var foundOutputName: String? = nil
            var detectedOutputType: ModelOutputType = .unknown  // Local variable for detection

            let multiArrayOutputs = description.outputDescriptionsByName.filter {
                $1.type == .multiArray
            }
            let imageOutputs = description.outputDescriptionsByName.filter { $1.type == .image }

            print(
                "ℹ️ Found \(multiArrayOutputs.count) MLMultiArray outputs: \(multiArrayOutputs.keys)"
            )
            print(
                "ℹ️ Found \(imageOutputs.count) Image (CVPixelBuffer) outputs: \(imageOutputs.keys)")

            if multiArrayOutputs.count == 1 && imageOutputs.isEmpty {
                // Case 1: Single MLMultiArray output (original expectation)
                detectedOutputType = .multiArray
                foundOutputName = multiArrayOutputs.first!.key  // Use key (name)
                print("✅ Detected SINGLE MLMultiArray output: '\(foundOutputName!)'")
            } else if imageOutputs.count == 1 && multiArrayOutputs.isEmpty {
                // Case 2: Single Image (CVPixelBuffer) output
                detectedOutputType = .image
                foundOutputName = imageOutputs.first!.key  // Use key (name)
                print("✅ Detected SINGLE Image (CVPixelBuffer) output: '\(foundOutputName!)'")
            } else {
                // Case 3: Unsupported output configuration
                print(
                    "🔴 Error: Unsupported output configuration. Expected exactly 1 output (either MLMultiArray OR Image)."
                )
                print(
                    "🔴 Found \(multiArrayOutputs.count) MLMultiArray and \(imageOutputs.count) Image outputs."
                )
                DispatchQueue.main.async {
                    self.error = StyleTransferError.outputNameNotDetermined(
                        "Expected 1 output (MLMultiArray or Image). Found M:\(multiArrayOutputs.count), I:\(imageOutputs.count)"
                    )
                    self.isModelLoaded = false
                    self.clearModelData()
                }
                return
            }

            // Success: Assign properties
            DispatchQueue.main.async {
                self.mlModel = loadedModel
                self.determinedContentInputName = finalContentName
                self.determinedStyleInputName = foundStyleName
                self.determinedOutputName = foundOutputName
                self.expectedInputType = detectedInputType
                self.currentModelStructure = detectedModelStructure
                self.expectedOutputType = detectedOutputType
                self.modelInputSize = finalInputSize
                self.isSingleInputModel = (detectedModelStructure == .singleInput)
                self.isModelLoaded = true
                self.error = nil
                print(
                    "✅ Model '\(modelName)' ready. Input Size: \(finalInputSize.width)x\(finalInputSize.height), Structure: \(detectedModelStructure), Input: \(detectedInputType), Output: \(detectedOutputType)."
                )
            }

        } catch {
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
        self.expectedInputType = .unknown
        self.currentModelStructure = .unknown
        self.expectedOutputType = .unknown
        self.modelInputSize = nil
        DispatchQueue.main.async {
            self.isSingleInputModel = false
        }
        // Clear style data too
        self.styleMultiArray = nil
        self.stylePixelBuffer = nil
        self.currentStyleName = nil
    }

    // Load and Preprocess Style Image
    func loadStyleImage(named imageName: String) {
        // Add a check: Don't load style if it's a single-input model
        guard currentModelStructure != .singleInput else {
            print("ℹ️ Style loading skipped: Current model is single-input.")
            // Optionally clear any existing style data if switching from dual to single
            // self.styleMultiArray = nil
            // self.stylePixelBuffer = nil
            // self.currentStyleName = nil
            // DispatchQueue.main.async { self.error = nil } // Clear potential style errors
            return
        }

        guard imageName != currentStyleName else {
            print("ℹ️ Style '\(imageName)' is already loaded.")
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

        guard let requiredSize = self.modelInputSize else {
            print("🔴 Error loading style: Model input size not determined yet.")
            DispatchQueue.main.async { self.error = StyleTransferError.modelNotLoaded }  // Or a more specific error
            return
        }

        print(
            "🔄 Loading style image: \(imageName)... (Target Size: \(requiredSize.width)x\(requiredSize.height))"
        )
        guard let uiImage = UIImage(named: imageName) else {
            print("🔴 Error: Could not load style image named '\(imageName)'.")
            DispatchQueue.main.async { [weak self] in
                self?.error = StyleTransferError.styleImageNotFound
                self?.styleMultiArray = nil
                self?.stylePixelBuffer = nil
                self?.currentStyleName = nil
            }
            return
        }

        // Preprocess using the dynamic size
        guard let buffer = preprocessImagePyTorchStyle(image: uiImage, targetSize: requiredSize)
        else {  // Pass CGSize
            print("🔴 Error: Failed to preprocess style image '\(imageName)' to CVPixelBuffer.")
            DispatchQueue.main.async { [weak self] in
                self?.error = StyleTransferError.styleImageProcessingFailed
                self?.styleMultiArray = nil
                self?.stylePixelBuffer = nil
                self?.currentStyleName = nil
            }
            return
        }
        self.stylePixelBuffer = buffer
        print("🟢 Style image '\(imageName)' preprocessed to CVPixelBuffer.")

        // Convert CVPixelBuffer to MLMultiArray (if needed by a dual-input model)
        // mlMultiArray(from:) now uses modelInputSize internally for validation
        if let multiArray = mlMultiArray(from: buffer) {
            self.styleMultiArray = multiArray
            print("🟢 Style CVPixelBuffer also converted to MLMultiArray.")
        } else {
            print(
                "⚠️ Warning: Could not create MLMultiArray for style. Models needing MLMultiArray input might fail if this style is used."
            )
            self.styleMultiArray = nil  // Ensure it's nil
        }

        // Update state
        self.currentStyleName = imageName
        DispatchQueue.main.async { [weak self] in
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
            let outputName = self.determinedOutputName,
            let requiredInputSize = self.modelInputSize,
            expectedInputType != .unknown,
            currentModelStructure != .unknown,
            expectedOutputType != .unknown
        else {
            print(
                "🔴 Error: Model not loaded or required properties (names, types, size) not determined."
            )
            DispatchQueue.main.async {
                // Provide more specific error based on what's missing
                if !self.isModelLoaded {
                    self.error = StyleTransferError.modelNotLoaded
                } else if self.modelInputSize == nil {
                    self.error = StyleTransferError.inputMismatch(
                        "Model input size not determined.")
                } else if self.expectedInputType == .unknown {
                    self.error = StyleTransferError.inputMismatch("Input type unknown.")
                } else if self.currentModelStructure == .unknown {
                    self.error = StyleTransferError.inputMismatch("Model structure unknown.")
                } else if self.determinedOutputName == nil {
                    self.error = StyleTransferError.outputNameNotDetermined("Output name unknown.")
                } else if self.expectedOutputType == .unknown {
                    self.error = StyleTransferError.outputTypeMismatch("Output type unknown.")
                } else {
                    self.error = StyleTransferError.modelNotLoaded
                }  // Fallback
                self.isProcessing = false
            }
            return
        }

        // Content Input Handling (Common Step)
        print(
            "ℹ️ Preprocessing content image to size: \(requiredInputSize.width)x\(requiredInputSize.height)"
        )
        guard
            let contentPixelBuffer = self.preprocessImagePyTorchStyle(
                image: contentImage, targetSize: requiredInputSize)
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
            self.error = nil
        }

        let startTime = Date()

        processingQueue.async { [weak self] in
            guard let self = self else { return }

            // Create Input Provider based on Model Structure and Expected Type
            var inputFeatures: [String: Any] = [:]
            var inputProviderCreationFailed = false
            var errorMessage: String? = nil  // For specific errors during input prep

            switch self.currentModelStructure {

            case .singleInput:
                print("ℹ️ Preparing input for SINGLE-INPUT model.")
                switch self.expectedInputType {
                case .multiArray:
                    guard let contentInputArray = self.mlMultiArray(from: contentPixelBuffer) else {
                        errorMessage = "Failed to convert content CVPixelBuffer to MLMultiArray."
                        inputProviderCreationFailed = true
                        break  // Break from inner switch
                    }
                    inputFeatures[contentInputName] = contentInputArray
                    print("ℹ️ Providing MLMultiArray content input.")
                case .image:
                    inputFeatures[contentInputName] = contentPixelBuffer
                    print("ℹ️ Providing CVPixelBuffer content input.")
                case .unknown:
                    errorMessage = "Expected input type is unknown for single-input model."
                    inputProviderCreationFailed = true
                }

            case .dualInput:
                print("ℹ️ Preparing input for DUAL-INPUT model.")
                guard let styleInputName = self.determinedStyleInputName else {
                    // This should have been caught during loadModel, but double-check
                    errorMessage = "Style input name missing for dual-input model."
                    inputProviderCreationFailed = true
                    break  // Break from outer switch
                }

                switch self.expectedInputType {
                case .multiArray:
                    guard let contentInputArray = self.mlMultiArray(from: contentPixelBuffer) else {
                        errorMessage = "Failed to convert content CVPixelBuffer to MLMultiArray."
                        inputProviderCreationFailed = true
                        break  // Break from inner switch
                    }
                    guard let styleInputArray = self.styleMultiArray else {
                        errorMessage = "Style MLMultiArray not available or not loaded."
                        inputProviderCreationFailed = true
                        break  // Break from inner switch
                    }
                    inputFeatures[contentInputName] = contentInputArray
                    inputFeatures[styleInputName] = styleInputArray
                    print("ℹ️ Providing MLMultiArray content and style inputs.")

                case .image:
                    guard let styleBuffer = self.stylePixelBuffer else {
                        errorMessage = "Style CVPixelBuffer not available or not loaded."
                        inputProviderCreationFailed = true
                        break  // Break from inner switch
                    }
                    inputFeatures[contentInputName] = contentPixelBuffer
                    inputFeatures[styleInputName] = styleBuffer
                    print("ℹ️ Providing CVPixelBuffer content and style inputs.")

                case .unknown:
                    errorMessage = "Expected input type is unknown for dual-input model."
                    inputProviderCreationFailed = true
                }

            case .unknown:
                errorMessage = "Model structure is unknown during processing."
                inputProviderCreationFailed = true
            }

            // Check if input preparation failed
            guard !inputProviderCreationFailed else {
                print("🔴 Input preparation failed: \(errorMessage ?? "Unknown reason")")
                DispatchQueue.main.async {
                    // Use a more specific error if available
                    if errorMessage?.contains("Style") ?? false {
                        self.error = StyleTransferError.styleImageNotSet  // Or a more specific style error
                    } else if errorMessage?.contains("convert content") ?? false {
                        self.error = StyleTransferError.inputConversionFailed
                    } else {
                        self.error = StyleTransferError.inputProviderCreationFailed  // Generic fallback
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

            // Perform Prediction
            do {
                let prediction = try model.prediction(from: inputProvider)
                var resultUIImage: UIImage? = nil  // Variable to hold the final UIImage

                // Extract and Convert Output
                switch self.expectedOutputType {
                case .multiArray:
                    print("ℹ️ Handling MLMultiArray output...")
                    guard
                        let multiArrayOutput = prediction.featureValue(for: outputName)?
                            .multiArrayValue
                    else {
                        print("🔴 Failed to get MLMultiArray output using name '\(outputName)'.")
                        throw StyleTransferError.unexpectedResultType  // Throw specific error
                    }
                    // Convert MLMultiArray output to Image using existing function
                    resultUIImage = self.imageFromMultiArray(multiArray: multiArrayOutput)
                    if resultUIImage == nil {
                        print("🔴 Failed to convert MultiArray output to image.")
                        throw StyleTransferError.multiArrayConversionFailed
                    }

                case .image:
                    print("ℹ️ Handling Image (CVPixelBuffer) output...")
                    guard
                        let pixelBufferOutput = prediction.featureValue(for: outputName)?
                            .imageBufferValue
                    else {
                        print(
                            "🔴 Failed to get Image (CVPixelBuffer) output using name '\(outputName)'."
                        )
                        throw StyleTransferError.unexpectedResultType  // Throw specific error
                    }
                    // Convert CVPixelBuffer output to Image using NEW function
                    resultUIImage = self.imageFromPixelBuffer(pixelBuffer: pixelBufferOutput)
                    if resultUIImage == nil {
                        print("🔴 Failed to convert CVPixelBuffer output to image.")
                        throw StyleTransferError.pixelBufferConversionFailed  // Add new error case
                    }

                case .unknown:
                    print("🔴 Cannot process output: Expected output type is unknown.")
                    throw StyleTransferError.outputTypeMismatch(
                        "Output type was not determined during model load.")
                }

                // --- Update UI (Common logic) ---
                let endTime = Date()
                let timeInterval = endTime.timeIntervalSince(startTime) * 1000

                // Update UI
                DispatchQueue.main.async {
                    self.styledImage = Image(uiImage: resultUIImage!)  // Force unwrap safe due to checks above
                    self.error = nil
                    self.processingTime = timeInterval
                    self.isProcessing = false
                    print("✅ Processing successful. Time: \(timeInterval) ms")
                }

            } catch let processingError {  // Catch errors from prediction or conversion
                print("🔴 Failed during prediction or output conversion: \(processingError)")
                // Ensure error is of expected type or wrap it
                let finalError: Error
                if processingError is StyleTransferError {
                    finalError = processingError
                } else {
                    // Wrap unexpected CoreML errors if needed
                    finalError = StyleTransferError.predictionFailed(
                        processingError.localizedDescription)  // Add new error case
                }
                DispatchQueue.main.async {
                    self.error = finalError
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
            self.error = StyleTransferError.modelLoading
            self.isProcessing = false
            // isSingleInputModel will be updated by loadModel
        }

        self.clearModelData()  // Clear internal properties

        // Load the new model asynchronously
        // Wrap loadModel call in a background queue if it's not already async
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.loadModel(named: modelName)
        }
    }

    // Preprocessing Function (Mimics PyTorch Resize(shorter_edge) + CenterCrop)
    private func preprocessImagePyTorchStyle(image: UIImage, targetSize: CGSize) -> CVPixelBuffer? {
        // Use integer dimensions from the CGSize
        let targetWidth = Int(targetSize.width)
        let targetHeight = Int(targetSize.height)
        guard targetWidth > 0 && targetHeight > 0 else {
            print("🔴 Preprocessing: Invalid target size \(targetSize)")
            return nil
        }
        let targetPixelCGSize = CGSize(width: targetWidth, height: targetHeight)

        let originalPixelWidth = image.size.width * image.scale
        let originalPixelHeight = image.size.height * image.scale
        let originalPixelSize = CGSize(width: originalPixelWidth, height: originalPixelHeight)

        print(
            "➡️ Preprocessing: Original Size \(image.size) (points), Scale \(image.scale), Pixels \(originalPixelWidth)x\(originalPixelHeight)"
        )

        // 1. Resize directly to target size
        guard let resizedImage = image.resize(to: targetPixelCGSize) else {
            print("🔴 Preprocessing: Failed to resize image to \(targetPixelCGSize)")
            return nil
        }
        // Log the size of the resized image (points) and its underlying pixel dimensions
        if let cg = resizedImage.cgImage {
            print(
                "➡️ Preprocessing: Resized image intermediate size \(resizedImage.size) (points), Scale \(resizedImage.scale), Pixels \(cg.width)x\(cg.height)"
            )
        }

        // 2. Convert to CVPixelBuffer using the exact target size
        guard
            let pixelBuffer = pixelBufferFromImage(
                image: resizedImage, expectedSize: targetPixelCGSize)
        else {
            print("🔴 Preprocessing: Failed to convert resized image to CVPixelBuffer.")
            return nil
        }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        print("➡️ Preprocessing: Final pixel buffer size \(width)x\(height)")

        // Final check (should always pass if pixelBufferFromImage works correctly)
        guard width == targetWidth && height == targetHeight else {
            print(
                "🔴 Preprocessing: Final pixel buffer size \(width)x\(height) does not match target \(targetWidth)x\(targetHeight)."
            )
            return nil
        }

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

    // Converts CVPixelBuffer (BGRA) to MLMultiArray (Float32, CHW, Normalized)
    private func mlMultiArray(from pixelBuffer: CVPixelBuffer) -> MLMultiArray? {
        guard let requiredSize = self.modelInputSize else {
            print("🔴 mlMultiArray: Cannot create array, model input size not determined.")
            return nil
        }
        let targetWidth = Int(requiredSize.width)
        let targetHeight = Int(requiredSize.height)

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        // Validate against the dynamic modelInputSize
        guard width == targetWidth && height == targetHeight else {
            print(
                "🔴 mlMultiArray: Input pixel buffer size \(width)x\(height) does not match required model size \(targetWidth)x\(targetHeight)"
            )
            return nil
        }

        // Create MLMultiArray using dynamic dimensions
        guard
            let multiArray = try? MLMultiArray(
                shape: [1, 3, NSNumber(value: targetHeight), NSNumber(value: targetWidth)],  // Use dynamic H/W
                dataType: .float32)
        else {
            print(
                "🔴 mlMultiArray: Failed to create MLMultiArray with shape [1, 3, \(targetHeight), \(targetWidth)]"
            )
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
            to: Float32.self, capacity: 3 * targetHeight * targetWidth)  // Use dynamic H/W

        // Normalization constants (MUST match Python preprocessing)
        let mean: [Float] = [0.485, 0.456, 0.406]  // RGB
        let std: [Float] = [0.229, 0.224, 0.225]  // RGB

        // Iterate through pixels and fill the MLMultiArray
        for y in 0..<targetHeight {
            for x in 0..<targetWidth {
                let pixelOffset = y * bytesPerRow + x * 4
                let b = buffer[pixelOffset + 0]
                let g = buffer[pixelOffset + 1]
                let r = buffer[pixelOffset + 2]
                let r_float = Float(r) / 255.0
                let g_float = Float(g) / 255.0
                let b_float = Float(b) / 255.0
                let r_norm = (r_float - mean[0]) / std[0]
                let g_norm = (g_float - mean[1]) / std[1]
                let b_norm = (b_float - mean[2]) / std[2]

                // Write to MLMultiArray (use dynamic height/width for index calculation)
                let r_index = (0 * targetHeight * targetWidth) + (y * targetWidth) + x
                let g_index = (1 * targetHeight * targetWidth) + (y * targetWidth) + x
                let b_index = (2 * targetHeight * targetWidth) + (y * targetWidth) + x

                multiArrayPointer[r_index] = r_norm
                multiArrayPointer[g_index] = g_norm
                multiArrayPointer[b_index] = b_norm
            }
        }

        return multiArray
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

        // Image Creation Loop
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

                // Assign in RGBA order to match CGImage parameters
                pixelData[pixelIndex + 0] = r_uint8  // Red
                pixelData[pixelIndex + 1] = g_uint8  // Green
                pixelData[pixelIndex + 2] = b_uint8  // Blue
                pixelData[pixelIndex + 3] = 255      // Alpha

                // SWAP R and B channels here
                // pixelData[pixelIndex + 0] = b_uint8  // Assign Blue to Red position
                // pixelData[pixelIndex + 1] = g_uint8  // Green remains Green
                // pixelData[pixelIndex + 2] = r_uint8  // Assign Red to Blue position
                // pixelData[pixelIndex + 3] = 255  // Alpha

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

    private func imageFromPixelBuffer(pixelBuffer: CVPixelBuffer) -> UIImage? {
        // Use CIImage for straightforward conversion
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)

        // Create a CGImage from the CIImage using the shared CIContext
        // Specify the frame rectangle of the CIImage to ensure correct dimensions
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
            print("🔴 imageFromPixelBuffer: Failed to create CGImage from CIImage.")
            return nil
        }

        // Create a UIImage from the CGImage
        return UIImage(cgImage: cgImage)
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
        case predictionFailed(String)  // New: For CoreML prediction errors
        case unexpectedResultType  // Model output feature was nil or wrong type (e.g., expected image, got multiarray)
        case multiArrayConversionFailed  // Failed converting MLMultiArray output to UIImage
        case pixelBufferConversionFailed  // New: Failed converting CVPixelBuffer output to UIImage
        case contentImageLoadFailed
        case outputNameNotDetermined(String)
        case inputNameNotDetermined(String)
        case inputMismatch(String)
        case outputTypeMismatch(String)  // Can be used if type determined at load doesn't match at runtime, or if unknown

        var errorDescription: String? {
            switch self {
            case .modelFileNotFound(let name): return "Model file '\(name).mlmodelc' not found."
            case .modelLoading: return "Loading selected model..."
            case .modelNotLoaded:
                return "Style transfer model could not be loaded or is not selected."
            case .styleImageNotFound: return "Style image file not found in assets."
            case .styleImageProcessingFailed: return "Could not resize/convert the style image."
            case .styleImageNotSet: return "Please select a style image (required for this model)."
            case .inputResizeFailed: return "Failed to resize content image for model input."
            case .inputConversionFailed:
                return "Failed to convert content image to model input format."
            case .inputProviderCreationFailed:
                return "Failed to create input features for the model."
            case .predictionFailed(let details): return "Model prediction failed: \(details)"  // New
            case .unexpectedResultType:
                return "Model output was missing or not the expected type."  // Modified
            case .multiArrayConversionFailed:
                return "Could not convert MLMultiArray output to an image."
            case .pixelBufferConversionFailed:
                return "Could not convert PixelBuffer output to an image."  // New
            case .contentImageLoadFailed: return "Failed to load the selected content image."
            case .outputNameNotDetermined(let details):
                return "Could not determine model output name. \(details)."
            case .inputNameNotDetermined(let details):
                return "Could not determine model input names. \(details)."
            case .inputMismatch(let details):
                return "Model input configuration mismatch. \(details)."
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
