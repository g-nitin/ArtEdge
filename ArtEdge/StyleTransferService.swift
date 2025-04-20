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
    private var outputFeatureName: String?
    private let processingQueue = DispatchQueue(
        label: "com.artedge.processingQueue", qos: .userInitiated)

    // Use the exact size expected by the model
    private let targetSize = CGSize(width: 224, height: 224)

    private var stylePixelBuffer: CVPixelBuffer?  // Hold the preprocessed style image
    private var currentStyleName: String?  // Keep track of the loaded style

    // Initialize without loading a default style immediately, or load a default one
    init(modelName: String = "AesFA") {  // Default model name
        loadModel(named: modelName)
    }

    private func loadModel(named modelName: String) {
        guard let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc")
        else {
            print(
                "🔴 Error: Could not find compiled model '\(modelName).mlmodelc'. Ensure it's added to the Xcode project target."
            )
            DispatchQueue.main.async {
                self.error = StyleTransferError.modelNotFound
                self.isModelLoaded = false
            }
            return
        }

        do {
            // Load MLModel directly
            // Use configuration to potentially force CPU for debugging
            let config = MLModelConfiguration()
            // config.computeUnits = .cpuOnly // Uncomment to force CPU for debugging GPU/ANE issues

            let loadedModel = try MLModel(contentsOf: modelURL, configuration: config)  // Load into a temporary variable
            self.mlModel = loadedModel  // Assign to the class property
            print("🟢 Successfully loaded model: \(modelName)")

            // Determine Output Feature Name
            // Check if the output name is known from conversion (often 'var_...' or similar)
            // Let's try to find the first MLMultiArray output dynamically as before,
            // but add more robust checking.
            let multiArrayOutputs = loadedModel.modelDescription.outputDescriptionsByName.values
                .filter { $0.type == .multiArray }

            if multiArrayOutputs.count == 1 {
                self.outputFeatureName = multiArrayOutputs.first!.name
                print(
                    "ℹ️ Automatically determined model output feature name: \(self.outputFeatureName!)"
                )
            } else if multiArrayOutputs.count > 1 {
                // If multiple multiarray outputs, we need a way to identify the correct one.
                // For now, let's print a warning and maybe pick the first one or require manual setting.
                print(
                    "⚠️ Warning: Model has multiple MLMultiArray outputs: \(multiArrayOutputs.map { $0.name }). Using the first one found: \(multiArrayOutputs.first!.name). Verify this is correct."
                )
                self.outputFeatureName = multiArrayOutputs.first!.name  // Or handle error / require specific name
            } else {
                // Fallback or error if no suitable output found
                print(
                    "🔴 Error: Could not find any MLMultiArray output feature for model '\(modelName)'."
                )
                self.outputFeatureName = nil  // Ensure it's nil if not found
                DispatchQueue.main.async {
                    self.error = StyleTransferError.outputNameNotDetermined
                    self.isModelLoaded = false  // Mark as not fully loaded/usable
                    self.mlModel = nil
                }
                return  // Stop further processing if output name is crucial and not found
            }

            DispatchQueue.main.async {
                self.isModelLoaded = true
                self.error = nil  // Clear previous errors
            }
        } catch {
            print("🔴 Error loading Core ML model: \(error)")
            DispatchQueue.main.async {
                self.error = error
                self.isModelLoaded = false
                self.mlModel = nil
                self.outputFeatureName = nil  // Clear output name on error too
            }
        }
    }

    // Load and Preprocess Style Image - Called when user selects a style
    func loadStyleImage(named imageName: String) {
        // Avoid reloading the same style unnecessarily
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

        print("🔄 Loading style image: \(imageName)...")
        guard let uiImage = UIImage(named: imageName) else {
            print("🔴 Error: Could not load style image named '\(imageName)'.")
            DispatchQueue.main.async { [weak self] in
                self?.error = StyleTransferError.styleImageNotFound
                self?.stylePixelBuffer = nil
                self?.currentStyleName = nil
            }
            return
        }

        // Preprocess using the updated function
        guard
            let buffer = preprocessImagePyTorchStyle(
                image: uiImage, targetSize: Int(targetSize.width))
        else {  // Pass Int size
            print("🔴 Error: Failed to preprocess style image '\(imageName)'.")
            DispatchQueue.main.async { [weak self] in
                self?.error = StyleTransferError.styleImageProcessingFailed
                self?.stylePixelBuffer = nil
                self?.currentStyleName = nil
            }
            return
        }

        self.stylePixelBuffer = buffer
        self.currentStyleName = imageName
        print("🟢 Style image '\(imageName)' loaded and preprocessed.")
        DispatchQueue.main.async { [weak self] in
            self?.error = nil
        }
    }

    // Process a SINGLE content image (UIImage)
    func process(contentImage: UIImage) {
        guard let model = mlModel, isModelLoaded else {
            print("🔴 Model not loaded.")
            DispatchQueue.main.async {
                self.error = StyleTransferError.modelNotLoaded
                self.isProcessing = false
            }
            return
        }
        guard let styleBuffer = stylePixelBuffer else {
            print("🔴 Style image not loaded or processed.")
            DispatchQueue.main.async {
                self.error = StyleTransferError.styleImageNotSet
                self.isProcessing = false
            }
            return
        }

        guard let outputName = self.outputFeatureName else {
            print("🔴 Error: Output feature name was not determined during model loading.")
            DispatchQueue.main.async {
                self.error = StyleTransferError.outputNameNotDetermined
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

            // 1. Preprocess content image using the updated function
            guard
                let contentPixelBuffer = self.preprocessImagePyTorchStyle(
                    image: contentImage, targetSize: Int(self.targetSize.width))  // Pass Int size
            else {
                print("🔴 Failed to preprocess content UIImage to CVPixelBuffer.")
                DispatchQueue.main.async {
                    self.error = StyleTransferError.inputResizeFailed
                    self.isProcessing = false
                }
                return
            }

            // 2. Create MLFeatureProvider
            guard
                let inputProvider = try? MLDictionaryFeatureProvider(
                    dictionary: [
                        "content": contentPixelBuffer,
                        "style": styleBuffer,
                    ]
                )
            else {
                print("🔴 Failed to create input provider. Check input names ('content', 'style').")
                DispatchQueue.main.async {
                    self.error = StyleTransferError.inputProviderCreationFailed
                    self.isProcessing = false
                }
                return
            }

            // 3. Perform Prediction
            do {
                let prediction = try model.prediction(from: inputProvider)
                // 4. Extract Output using the determined name
                guard
                    let multiArrayOutput = prediction.featureValue(for: outputName)?.multiArrayValue
                else {
                    print(
                        "🔴 Failed to get MLMultiArray output using determined name '\(outputName)'. Check model output."
                    )
                    DispatchQueue.main.async {
                        self.error = StyleTransferError.unexpectedResultType
                        self.isProcessing = false
                    }
                    return
                }

                // 5. Convert MLMultiArray to Image (with added logging)
                let convertedImage = self.imageFromMultiArray(multiArray: multiArrayOutput)

                let endTime = Date()
                let timeInterval = endTime.timeIntervalSince(startTime) * 1000  // ms

                // 6. Update UI on Main Thread
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
                    self.isProcessing = false  // Mark processing finished
                }

            } catch {
                print("🔴 Failed to perform prediction: \(error)")
                DispatchQueue.main.async {
                    self.error = error
                    self.styledImage = nil
                    self.isProcessing = false  // Mark processing finished on error
                }
            }
        }
    }

    func switchModel(to modelName: String) {
        print("🔄 Switching model to: \(modelName)")
        // Reset state before loading new model
        DispatchQueue.main.async {
            self.isModelLoaded = false
            self.styledImage = nil
            self.error = nil
            self.isProcessing = false  // Cancel any ongoing processing if needed
        }
        // Clear internal state related to the old model
        self.mlModel = nil
        self.outputFeatureName = nil
        // Load the new model (this will update isModelLoaded, error, outputFeatureName)
        loadModel(named: modelName)
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

    // Define potential errors
    enum StyleTransferError: Error, LocalizedError {
        case modelNotFound
        case modelNotLoaded
        case styleImageNotFound
        case styleImageProcessingFailed
        case styleImageNotSet
        case inputResizeFailed
        case inputProviderCreationFailed
        case unexpectedResultType
        case multiArrayConversionFailed
        case contentImageLoadFailed
        case outputNameNotDetermined

        var errorDescription: String? {
            switch self {
            case .modelNotFound:
                return
                    "Style transfer model file (\(Bundle.main.bundleIdentifier ?? "App")/*.mlmodelc) not found. Ensure it's added to the target."
            case .modelNotLoaded:
                return "Style transfer model could not be loaded. Check console logs."
            case .styleImageNotFound: return "Style image file not found in assets."
            case .styleImageProcessingFailed: return "Could not resize/convert the style image."
            case .styleImageNotSet: return "Please select a style image first."
            case .inputResizeFailed:
                return "Failed to resize/convert content image for model input."
            case .inputProviderCreationFailed:
                return "Failed to create input features for the model. Check input names."
            case .unexpectedResultType:
                return "Model output was not the expected format (MLMultiArray)."
            case .multiArrayConversionFailed:
                return "Could not convert model output tensor to an image."
            case .contentImageLoadFailed: return "Failed to load the selected content image."
            case .outputNameNotDetermined:
                return
                    "Could not determine the required MLMultiArray output name from the loaded model."
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
