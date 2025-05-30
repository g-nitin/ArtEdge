import PhotosUI
import SwiftUI
import UIKit

enum ModelFamily: String, CaseIterable, Identifiable {
    case adain = "AdaIN"  // Single-style models, BUT require style image input as per new req.
    case fst = "FST"  // Single-style models, loaded via style selection (truly single input)
    case aesfa = "AesFA"  // Arbitrary model, loaded on family selection
    case stytr2 = "StyTR2"  // Arbitrary model, loaded on family selection

    var id: String { self.rawValue }

    // Helper to get the primary model filename for arbitrary types
    // Returns nil for families where the model is loaded via style selection
    var primaryModelFilename: String? {
        switch self {
        // AdaIN models are loaded via style selection, so no primary model here.
        case .adain: return nil
        case .fst: return nil  // Model determined by style selection
        case .aesfa: return "AesFA"  // This model is loaded when family is selected
        case .stytr2: return "StyTr2"  // This model is loaded when family is selected
        }
    }

    // Check if this family uses models loaded via style tap
    // Both AdaIN and FST models are loaded this way.
    var isSingleStyleFamily: Bool {
        switch self {
        case .adain, .fst: return true
        case .aesfa, .stytr2: return false
        }
    }

    // Check if this family represents an arbitrary style transfer model
    // (can take any style image as input)
    var isArbitraryStyleFamily: Bool {
        switch self {
        // AdaIN is NOT arbitrary, even though it takes 2 inputs now.
        // FST is NOT arbitrary.
        case .adain, .fst: return false
        case .aesfa, .stytr2: return true
        }
    }

    // NEW: Check if this family requires a style IMAGE input during processing
    // This is true for arbitrary models AND the modified AdaIN flow.
    var requiresStyleImageInput: Bool {
        switch self {
        case .adain: return true  // AdaIN models now require the style image input
        case .fst: return false  // FST models are truly single-input
        case .aesfa, .stytr2: return true  // Arbitrary models require style image input
        }
    }
}

// ContentView conform to NSObject to handle the save callback
class ContentViewCoordinator: NSObject {
    var parent: ContentView

    init(_ parent: ContentView) {
        self.parent = parent
    }

    // Callback function for UIImageWriteToSavedPhotosAlbum
    @objc func image(
        _ image: UIImage, didFinishSavingWithError error: Error?, contextInfo: UnsafeRawPointer
    ) {
        DispatchQueue.main.async {  // Ensure UI updates on main thread
            if let error = error {
                // We got back an error!
                self.parent.userMessage = "Save error: \(error.localizedDescription)"
                print("🔴 Save error: \(error.localizedDescription)")
            } else {
                // Saved successfully
                self.parent.userMessage = "Result saved to Photos!"
                print("✅ Image saved successfully.")
                // Optionally clear the message after a delay
                DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
                    if self.parent.userMessage == "Result saved to Photos!" {
                        self.parent.userMessage = nil
                    }
                }
            }
        }
    }
}

struct ContentView: View {
    // State for UI elements and selections
    @State private var contentImage: UIImage?
    @State private var selectedPhotoItem: PhotosPickerItem?
    @State private var selectedStyle: StyleInfo?  // Represents the *selected style image* or *selected single-style model*
    @State private var resultImage: Image?
    @State var userMessage: String? {
        didSet {
            // Optional: Print message changes for debugging
            // print("ℹ️ User Message Updated: \(userMessage ?? "nil")")
        }
    }

    // Default to an arbitrary model family on launch
    @State private var selectedModelFamily: ModelFamily = .aesfa

    // State to control camera picker presentation
    @State private var showCameraPicker = false

    // Initialize StyleTransferService - Load the initial family's model if applicable
    @StateObject private var styleTransferService: StyleTransferService

    // Coordinator for save callback
    @State private var coordinator: ContentViewCoordinator?

    // Computed property to check if processing can start
    private var canProcess: Bool {
        let hasContent = contentImage != nil
        // A model must be loaded (specific single-style or arbitrary)
        let modelReady = styleTransferService.isModelLoaded
        let serviceIdle = !styleTransferService.isProcessing
        // A style must have been selected (either to load the model or provide input)
        let styleSelected = selectedStyle != nil

        // Additional check: If the model requires a style image input (AdaIN, AesFA, StyTR2),
        // ensure the style image loading hasn't failed. We infer this by checking
        // that the error state isn't specifically a style-loading error.
        var styleInputRequirementMet = true
        if selectedModelFamily.requiresStyleImageInput && modelReady {
            if let currentError = styleTransferService.error
                as? StyleTransferService.StyleTransferError
            {
                switch currentError {
                case .styleImageNotFound, .styleImageProcessingFailed, .styleImageNotSet:
                    styleInputRequirementMet = false  // Style input is required but failed/missing
                default:
                    styleInputRequirementMet = true  // Other errors don't block based on style input state
                }
            } else {
                // No error, assume style input is loaded or will be loaded correctly if needed.
                // Note: This assumes loadStyleImage was called correctly after model load for AdaIN.
                styleInputRequirementMet = true
            }
        }

        // Combine all conditions
        let allMet =
            hasContent && modelReady && serviceIdle && styleSelected && styleInputRequirementMet

        // Debugging log for canProcess state changes
        // print("ℹ️ canProcess check: hasContent=\(hasContent), modelReady=\(modelReady), serviceIdle=\(serviceIdle), styleSelected=\(styleSelected), styleInputReqMet=\(styleInputRequirementMet) -> \(allMet)")

        return allMet
    }

    // Custom Initializer
    init() {
        // Start with an arbitrary family (e.g., AesFA) so a model loads initially
        let initialFamily: ModelFamily = .adain
        _selectedModelFamily = State(initialValue: initialFamily)

        // Initialize the service with the initial family's primary model filename
        // This will be "AesFA" in this case. If we defaulted to AdaIN/FST, it would be "" or nil.
        let initialModelName = initialFamily.primaryModelFilename ?? ""
        _styleTransferService = StateObject(
            wrappedValue: StyleTransferService(modelName: initialModelName)
        )
        print(
            "ContentView init: Initializing with model family '\(initialFamily.rawValue)' (Model: \(initialModelName.isEmpty ? "None - Select Style" : initialModelName))"
        )
    }

    var body: some View {
        ZStack {
            Color(red: 252 / 255, green: 247 / 255, blue: 243 / 255)
                .ignoresSafeArea()
            NavigationView {
                VStack(spacing: 0) {
                    Image("ArtEdgeLogoBanner")
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: .infinity)
                        .clipped()
                        .padding(.bottom, 10)

                    ScrollView {
                        VStack(spacing: 15) {
                            modelFamilySelectionSection
                            Divider()
                            contentImageSection
                            Divider()
                            styleSelectionSection  // Dynamically shows correct styles
                            Divider()
                            actionButtonSection
                            userMessageSection
                            resultSection
                            Spacer()  // Pushes content up
                        }
                        .padding(.top)  // Add padding above the first section
                    }
                }
                // .navigationTitle("ArtEdge Style Transfer") // Give it a title
                .navigationBarTitleDisplayMode(.inline)
                .background(Color(red: 252 / 255, green: 247 / 255, blue: 243 / 255))  // Apply background to VStack
            }
            .navigationViewStyle(.stack)  // Recommended for avoiding sidebar issues on iPad if needed
        }
        .onAppear {
            // Initialize the coordinator when the view appears
            if coordinator == nil {
                coordinator = ContentViewCoordinator(self)
            }
        }
        // Load UIImage when photo item changes
        .onChange(of: selectedPhotoItem) { newItem in
            handlePhotoSelection(newItem: newItem)
        }
        // Clear photo item if image is set via camera
        .onChange(of: contentImage) { newImage in
            if newImage != nil && selectedPhotoItem != nil { selectedPhotoItem = nil }
        }
        .onChange(of: selectedModelFamily) { newFamily in
            handleModelFamilyChange(newFamily: newFamily)
        }
        .onReceive(styleTransferService.$styledImage) { newStyledImage in
            handleStyledImageUpdate(newStyledImage: newStyledImage)
        }
        .onReceive(styleTransferService.$error) { error in handleServiceError(error: error) }
        .onChange(of: styleTransferService.isModelLoaded) { loaded in
            handleModelLoadingStatus(loaded: loaded)  // Existing handler
            // NEW: Trigger style image loading for AdaIN *after* its model loads
            triggerAdaINStyleImageLoadIfNeeded(modelIsLoaded: loaded)
        }
        .onReceive(styleTransferService.$isProcessing) { processing in
            handleProcessingStatus(processing: processing)
        }
        .onReceive(styleTransferService.$isStyleInputDataLoaded) { isDataLoaded in
            handleStyleInputDataLoaded(isDataLoaded)
        }
        .fullScreenCover(isPresented: $showCameraPicker) {
            ImagePicker(image: $contentImage, isPresented: $showCameraPicker, sourceType: .camera)
        }
    }

    private var modelFamilySelectionSection: some View {
        VStack(alignment: .leading) {
            Text("1. Select Model Type").font(.headline)
                .padding(.horizontal)

            Picker("Model Type", selection: $selectedModelFamily) {
                ForEach(ModelFamily.allCases) { family in
                    Text(family.rawValue).tag(family)
                }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal)
            .disabled(styleTransferService.isProcessing)
        }
    }

    private var contentImageSection: some View {
        VStack {
            Text("2. Choose Content Image").font(.headline)
            if let contentImage = contentImage {
                Image(uiImage: contentImage)
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 200)
                    .cornerRadius(8)
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.gray, lineWidth: 1))
            } else {
                Rectangle()
                    .fill(Color.secondary.opacity(0.1))
                    .frame(height: 150)
                    .cornerRadius(8)
                    .overlay(Text("Select or take a photo").foregroundColor(.secondary))
            }

            HStack(spacing: 15) {
                PhotosPicker(
                    selection: $selectedPhotoItem, matching: .images, photoLibrary: .shared()
                ) {
                    Label("Library", systemImage: "photo.on.rectangle")
                }
                .buttonStyle(.bordered)
                .disabled(styleTransferService.isProcessing)  // Disable while processing

                Button {
                    showCameraPicker = true
                } label: {
                    Label("Photo", systemImage: "camera.fill")
                }
                .buttonStyle(.bordered)
                .disabled(
                    !UIImagePickerController.isSourceTypeAvailable(.camera)
                        || styleTransferService.isProcessing)  // Disable while processing
            }
            .padding(.top, 5)
        }
        .padding(.horizontal)
    }

    private var styleSelectionSection: some View {
        VStack {
            Text(styleSectionTitle).font(.headline)  // Dynamic title

            // Show selected style name clearly
            if let style = selectedStyle {
                Text("Selected: \(style.name)")
                    .font(.caption)
                    .foregroundColor(.blue)
                    .padding(.bottom, 2)
            } else {
                Text(styleSectionPrompt)  // Dynamic prompt
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding(.bottom, 2)
            }

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 15) {  // Add spacing between items
                    // Use the computed property to get the correct list
                    ForEach(currentStyleList) { style in
                        VStack {
                            Image(style.assetName)  // Assumes assetName matches imageset
                                .resizable()
                                .scaledToFill()
                                .frame(width: 80, height: 80)
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 8)
                                        .stroke(
                                            selectedStyle == style
                                                ? Color.blue : Color.gray.opacity(0.5),  // Use opacity for unselected
                                            lineWidth: selectedStyle == style ? 3 : 1
                                        )
                                )
                                .onTapGesture {
                                    handleStyleTap(style)  // Centralized tap logic
                                }
                            Text(style.name).font(.caption).lineLimit(1)
                        }
                    }
                }
                .padding(.horizontal)  // Add padding so items aren't flush with edges
                .padding(.vertical, 5)  // Add vertical padding
            }
            .frame(height: 110)  // Fixed height for the scroll view
            // Disable while processing OR if a model is currently loading (indicated by isModelLoaded=false AND a style was selected to trigger it)
            .disabled(
                styleTransferService.isProcessing
                    || (!styleTransferService.isModelLoaded && selectedStyle != nil
                        && selectedModelFamily.isSingleStyleFamily)
            )
        }
    }

    // Helper computed properties for style section UI
    private var styleSectionTitle: String {
        switch selectedModelFamily {
        // Both load via tap, but AdaIN now also needs the image *input*
        case .adain: return "3. Select Style"
        case .fst: return "3. Select Style"
        case .aesfa, .stytr2: return "3. Select Style"
        }
    }

    private var styleSectionPrompt: String {
        switch selectedModelFamily {
        case .adain: return "Tap an AdaIN style below to load its model and style data"
        case .fst: return "Tap an FST style below to load its model"
        case .aesfa, .stytr2: return "Tap a style to use as input"
        }
    }

    // Dynamically returns the correct list of StyleInfo based on the selected family
    private var currentStyleList: [StyleInfo] {
        switch selectedModelFamily {
        case .adain: return availableAdaINStyles
        case .fst: return availableFSTStyles
        // Use the same list of style inputs for both arbitrary and AdaIN families now
        case .aesfa, .stytr2: return availableArbitraryStyleInputs
        }
    }

    private var actionButtonSection: some View {
        Button {
            applyStyle()
        } label: {
            if styleTransferService.isProcessing {
                HStack {
                    ProgressView().tint(.white)
                    Text("Applying Style...")
                }  // Ensure progress view is visible
                .frame(maxWidth: .infinity)  // Make label take full width for consistent button size
            } else {
                Label("Apply Style", systemImage: "paintbrush.pointed.fill")
                    .frame(maxWidth: .infinity)  // Make label take full width
            }
        }
        .buttonStyle(.borderedProminent)
        .disabled(!canProcess)  // Use updated computed property
        .padding(.top)
        .padding(.horizontal)  // Add horizontal padding to button
    }

    private var userMessageSection: some View {
        Group {
            // Only show message if it's not nil and not empty
            if let msg = userMessage, !msg.isEmpty {
                Text(msg)
                    // Determine color based on message content
                    .foregroundColor(messageColor(msg))
                    .font(.caption)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
                    .padding(.vertical, 5)  // Add vertical padding
                    .transition(.opacity.animation(.easeInOut(duration: 0.3)))  // Animate appearance/disappearance
                    .id(msg)  // Use message content as ID to force transition on change
            } else {
                // Placeholder to maintain layout stability when no message
                Text(" ")  // Use a space instead of empty string
                    .font(.caption)
                    .padding(.horizontal)
                    .padding(.vertical, 5)
            }
        }
        .frame(minHeight: 30)  // Ensure minimum height for message area
    }

    // Helper to determine message color
    private func messageColor(_ msg: String) -> Color {
        let lowerMsg = msg.lowercased()
        if lowerMsg.contains("error") || lowerMsg.contains("failed")
            || lowerMsg.contains("could not") || lowerMsg.contains("mismatch")
        {
            return .red
        } else if lowerMsg.contains("success") || lowerMsg.contains("saved")
            || lowerMsg.contains("loaded")
        {
            return .green  // Use green for success messages
        } else {
            return .secondary  // Default gray for informational messages (loading, prompts)
        }
    }

    private var resultSection: some View {
        VStack {  // Use VStack to group result elements
            // Show progress only if processing AND no result image yet
            if styleTransferService.isProcessing && resultImage == nil {
                ProgressView().padding(.top)
            } else if let result = resultImage {
                Divider().padding(.vertical, 10)
                Text("Result").font(.headline)
                result
                    .resizable()
                    .scaledToFit()
                    .cornerRadius(8)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8).stroke(
                            Color.gray.opacity(0.5), lineWidth: 1)
                    )  // Added border
                    .padding(.vertical, 5)
                    .contextMenu {  // Allow saving the result
                        Button {
                            saveImage(image: result)
                        } label: {
                            Label("Save Image", systemImage: "square.and.arrow.down")
                        }
                        Button {
                            shareImage(image: result)
                        } label: {
                            Label("Share", systemImage: "square.and.arrow.up")
                        }
                    }

                Text(
                    "Processing Time: \(String(format: "%.1f", styleTransferService.processingTime)) ms"
                )
                .font(.caption)
                .foregroundColor(.secondary)
                .padding(.bottom)
            }
            // No need for an else clause if neither processing nor result exists
        }
        .padding(.horizontal)
    }

    func handleModelFamilyChange(newFamily: ModelFamily) {
        print("ContentView: Detected model family change to '\(newFamily.rawValue)'")
        resultImage = nil  // Clear previous result
        selectedStyle = nil  // Clear previous style selection
        userMessage = nil  // Clear previous message first

        if let modelToLoad = newFamily.primaryModelFilename {
            // This is an arbitrary family (AesFA, StyTR2)
            // Set loading message *before* calling switchModel
            userMessage = "Switching to \(newFamily.rawValue) model..."
            styleTransferService.switchModel(to: modelToLoad)  // Load the main model
        } else {
            // This is a single-style family (AdaIN, FST)
            // Reset the service, unloading any previous model.
            styleTransferService.resetService()
            // Set the prompt *after* resetting
            userMessage = styleSectionPrompt  // Use the prompt for this family
        }
    }

    func handleStyleTap(_ style: StyleInfo) {
        guard !styleTransferService.isProcessing else { return }

        print(
            "ContentView: Tapped style '\(style.name)' for family '\(selectedModelFamily.rawValue)'"
        )
        selectedStyle = style  // Update selection regardless of family

        switch selectedModelFamily {
        case .adain:
            // 1. Load the specific AdaIN *model* associated with this style
            if let modelToLoad = style.associatedModelFilename {
                print("-> Switching to AdaIN model: '\(modelToLoad)'")
                // Clear previous general message before setting specific loading message
                if userMessage != nil && !(userMessage?.lowercased().contains("error") ?? false) {
                    userMessage = nil
                }
                userMessage = "Loading \(style.name) (AdaIN) model..."
                styleTransferService.switchModel(to: modelToLoad)
                // 2. Style *image* loading will be triggered by onChange(of: isModelLoaded) once the model load completes successfully.
            } else {
                handleMissingModelError(style: style)
            }

        case .fst:
            // Load the specific single-style *model* associated with this style
            if let modelToLoad = style.associatedModelFilename {
                print("-> Switching to FST model: '\(modelToLoad)'")
                if userMessage != nil && !(userMessage?.lowercased().contains("error") ?? false) {
                    userMessage = nil
                }
                userMessage = "Loading \(style.name) (FST) model..."
                styleTransferService.switchModel(to: modelToLoad)
            } else {
                handleMissingModelError(style: style)
            }

        case .aesfa, .stytr2:
            // Check if this style's asset is already the current one in the service AND its data is loaded
            if styleTransferService.currentStyleName == style.assetName
                && styleTransferService.isStyleInputDataLoaded
            {
                print("-> Style '\(style.name)' (\(style.assetName)) is already loaded and active.")

                let activeMessage = "Style '\(style.name)' is active."
                if userMessage != activeMessage {  // Avoid redundant sets
                    userMessage = activeMessage
                }

                DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                    if self.userMessage == activeMessage {
                        self.userMessage = nil
                    }
                }
                return  // Exit: Don't re-issue load command
            }

            // If not already loaded, proceed to load
            print(
                "-> Attempting to load style asset '\(style.assetName)' as input for \(selectedModelFamily.rawValue)."
            )
            if userMessage != nil && !(userMessage?.lowercased().contains("error") ?? false) {
                userMessage = nil
            }
            userMessage = "Loading style input: \(style.name)..."
            styleTransferService.loadStyleImage(named: style.assetName)
        }
    }

    // Helper for missing model error
    private func handleMissingModelError(style: StyleInfo) {
        let errorMsg = "Error: Configuration issue - model not defined for style \(style.name)."
        print("🔴 \(errorMsg)")
        userMessage = errorMsg
        styleTransferService.error = StyleTransferService.StyleTransferError
            .modelFileNotFound("Associated with \(style.name)")
    }

    // Function to trigger AdaIN style image loading
    private func triggerAdaINStyleImageLoadIfNeeded(modelIsLoaded: Bool) {
        // Check conditions: model just loaded, family is AdaIN, a style is selected,
        // and the loaded model requires style input (is dual-input).
        guard modelIsLoaded,
            selectedModelFamily == .adain,
            let currentSelectedStyle = selectedStyle,
            styleTransferService.currentModelStructure == .dualInput  // Check the actual loaded model structure
        else {
            // Conditions not met, do nothing.
            // Log why if needed for debugging:
            // if modelIsLoaded && selectedModelFamily == .adain {
            //     print("ℹ️ AdaIN style load trigger skipped: selectedStyle=\(selectedStyle != nil), modelStructure=\(styleTransferService.currentModelStructure)")
            // }
            return
        }

        // Check if the style image for this specific style is already loaded or being loaded
        // Avoid redundant calls if the user taps the same style quickly.
        // We infer this by checking if an error related to style loading exists for *this* style.
        // A more robust check might involve tracking the state in StyleTransferService.
        if let currentError = styleTransferService.error as? StyleTransferService.StyleTransferError
        {
            switch currentError {
            case .styleImageNotFound, .styleImageProcessingFailed, .styleImageNotSet:
                // If there's already a style error, maybe don't retry automatically,
                // or only retry if the error is specifically .styleImageNotSet.
                // For now, let's proceed to allow retrying.
                print(
                    "⚠️ Proceeding with AdaIN style image load despite existing style error: \(currentError)"
                )
            default:
                break  // Other errors don't prevent style loading attempt
            }
        }

        // All checks passed, load the style image associated with the selected AdaIN style
        print(
            "✅ AdaIN model '\(currentSelectedStyle.associatedModelFilename ?? "N/A")' loaded. Now loading style input asset '\(currentSelectedStyle.assetName)'..."
        )
        userMessage = "Loading \(currentSelectedStyle.name) style input..."
        styleTransferService.loadStyleImage(named: currentSelectedStyle.assetName)
    }

    // Handler for isStyleInputDataLoaded changes
    private func handleStyleInputDataLoaded(_ isDataLoaded: Bool) {
        // Try to get the display name of the style that the service reports as current
        // Note: styleTransferService.currentStyleName is the asset name
        let serviceAssetName = styleTransferService.currentStyleName

        // Find the StyleInfo corresponding to the service's current asset name
        // This covers both arbitrary style inputs and AdaIN style inputs.
        let activeStyleInfo =
            (availableArbitraryStyleInputs.first { $0.assetName == serviceAssetName }
                ?? availableAdaINStyles.first { $0.assetName == serviceAssetName })

        if isDataLoaded, let style = activeStyleInfo {
            // Check if the userMessage was for loading *this specific* style
            let loadingArbitraryMsg = "Loading style input: \(style.name)..."
            let loadingAdaINMsg = "Loading \(style.name) style input..."  // Used by triggerAdaINStyleImageLoadIfNeeded

            if userMessage == loadingArbitraryMsg || userMessage == loadingAdaINMsg {
                let successMsg = "Style input '\(style.name)' loaded."
                userMessage = successMsg
                print("✅ \(successMsg)")
                DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                    if self.userMessage == successMsg {
                        self.userMessage = nil
                    }
                }
            }
        } else if !isDataLoaded {
            // Style input data became unloaded (e.g., model switched to single-input, or load failed implicitly)
            // If a "Style input 'X' loaded." message is showing, clear it,
            // but only if there isn't a superseding error message.
            if let msg = userMessage, msg.contains("Style input") && msg.contains("loaded."),
                styleTransferService.error == nil
            {  // Only clear if no active error
                userMessage = nil
            }
            // If a "Loading style input..." message was showing and isDataLoaded became false,
            // it implies a failure. handleServiceError should ideally pick up the error message.
            // If it doesn't (e.g. error was set then cleared, but isDataLoaded remains false),
            // this branch might need to be more robust. For now, assume error handling path is primary for failures.
        }
    }

    func applyStyle() {
        guard let content = contentImage else {
            userMessage = "Please select a content image first."
            return
        }

        // Use canProcess to simplify the guard
        guard canProcess else {
            // Provide more specific feedback why it's disabled
            if contentImage == nil {
                userMessage = "Please select a content image."
            } else if !styleTransferService.isModelLoaded {
                userMessage = "Model is not loaded. Please select a style/family."
            } else if selectedStyle == nil {
                userMessage = "Please select a style."
            } else if selectedModelFamily.requiresStyleImageInput
                && !styleInputRequirementMetInCanProcess()
            {
                // Check the specific style input requirement again for a clearer message
                userMessage =
                    "Style input required for \(selectedModelFamily.rawValue) is missing or failed to load."
                print("🔴 Apply blocked: Style input requirement not met.")
            } else if styleTransferService.isProcessing {
                userMessage = "Already processing..."  // Should be handled by button state
            } else {
                userMessage = "Cannot apply style. Check selections and wait for loading."  // Generic fallback
            }
            print("🚫 Apply Style blocked. canProcess = false.")
            return
        }

        // If all checks pass
        print(
            "🚀 Triggering style transfer. Family: \(selectedModelFamily.rawValue), Style: \(selectedStyle!.name), Model Loaded: \(styleTransferService.isModelLoaded), Requires Style Input: \(selectedModelFamily.requiresStyleImageInput)"
        )
        // Set processing message *before* calling process
        userMessage = "Processing..."
        styleTransferService.process(contentImage: content)
    }

    // Helper function mirroring the logic within canProcess for style input check
    private func styleInputRequirementMetInCanProcess() -> Bool {
        if selectedModelFamily.requiresStyleImageInput && styleTransferService.isModelLoaded {
            if let currentError = styleTransferService.error
                as? StyleTransferService.StyleTransferError
            {
                switch currentError {
                case .styleImageNotFound, .styleImageProcessingFailed, .styleImageNotSet:
                    return false
                default:
                    return true
                }
            } else {
                return true
            }
        }
        return true  // Not required or model not loaded yet
    }

    func handlePhotoSelection(newItem: PhotosPickerItem?) {
        // Clear previous result immediately
        resultImage = nil

        // Set loading message immediately
        var isLoadingSet = false
        if newItem != nil {
            userMessage = "Loading content image..."
            isLoadingSet = true
        } else if contentImage == nil {
            // Only clear message if nothing was selected AND no image was previously loaded
            userMessage = nil
        }
        // If newItem is nil but contentImage exists, keep the existing image and message

        Task {
            guard let item = newItem else {
                // No item selected, message handled above
                return
            }
            do {
                print("⏳ Starting to load image data...")
                if let data = try await item.loadTransferable(type: Data.self) {
                    print("⏳ Data loaded (\(data.count) bytes), creating UIImage...")
                    if let uiImage = UIImage(data: data) {
                        // Update on main thread
                        await MainActor.run {
                            contentImage = uiImage
                            // Clear the "Loading..." message only if it was set for this operation
                            if isLoadingSet && userMessage == "Loading content image..." {
                                userMessage = nil
                            }
                            print("✅ Content image loaded from library.")
                        }
                    } else {
                        print("🔴 Failed to create UIImage from data.")
                        throw StyleTransferService.StyleTransferError.contentImageLoadFailed
                    }
                } else {
                    print("🔴 loadTransferable returned nil data.")
                    throw StyleTransferService.StyleTransferError.contentImageLoadFailed
                }
            } catch {
                print("🔴 Error loading content image: \(error)")
                // Update UI on main thread
                await MainActor.run {
                    contentImage = nil  // Clear image on failure
                    // Let handleServiceError display the error message
                    handleServiceError(error: error)
                }
            }
        }
    }

    func handleStyledImageUpdate(newStyledImage: Image?) {
        self.resultImage = newStyledImage
        // Clear processing message *only if* successful and the message *is* "Processing..."
        if newStyledImage != nil && userMessage == "Processing..." {
            userMessage = nil  // Success, clear processing message
        }
        // If newStyledImage is nil, it might be due to an error (handled elsewhere)
        // or the start of a new process (handled by handleProcessingStatus)
    }

    func handleModelLoadingStatus(loaded: Bool) {
        let familyName = selectedModelFamily.rawValue
        let styleName = selectedStyle?.name  // Get name if a style triggered the load

        // Only update message if a loading operation was likely in progress
        let wasLoadingModel =
            userMessage?.lowercased().contains("loading \(styleName ?? familyName)") ?? false
        let wasSwitchingFamily =
            userMessage?.lowercased().contains("switching to \(familyName)") ?? false
        let wasLoading = wasLoadingModel || wasSwitchingFamily

        if loaded {
            if wasLoading {
                // Determine if it's single or dual input based on the *service's* detection
                let modelType =
                    styleTransferService.currentModelStructure == .singleInput
                    ? "(Single-Input)" : "(Dual-Input)"
                let specificName = styleName ?? familyName  // Use style name if available, else family name
                var successMsg = "'\(specificName)' \(modelType) model loaded."

                // If it's AdaIN, the style input loading starts next (handled by triggerAdaINStyleImageLoadIfNeeded)
                // Avoid clearing the message immediately for AdaIN.
                let isAdaINLoad = selectedModelFamily == .adain && styleName != nil

                if !isAdaINLoad {
                    userMessage = successMsg + " Ready."
                    print("✅ \(successMsg)")
                    // Optionally clear the message after a delay
                    DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
                        if userMessage == successMsg + " Ready." {
                            userMessage = nil
                        }
                    }
                } else {
                    // For AdaIN, just confirm model load, next step is style input load
                    userMessage = successMsg  // Message will be updated again by triggerAdaIN...
                    print("✅ \(successMsg) - Awaiting style input load.")
                }
            }
            // If an arbitrary model loaded successfully after selecting the family, prompt for style input
            else if selectedModelFamily.isArbitraryStyleFamily && selectedStyle == nil
                && styleTransferService.error == nil
            {
                // Only set prompt if no other message (like an error) is present
                if userMessage == nil {
                    userMessage = styleSectionPrompt  // e.g., "Tap a style to use as input"
                }
            }

        } else {
            // Model is not loaded.
            // If it was loading, an error likely occurred (handled by handleServiceError).
            // If it was reset intentionally (e.g., switching to FST family), the message should be the prompt.
            if styleTransferService.error == nil {
                // If switching *to* a family that requires selection (FST, AdaIN)
                if selectedModelFamily.isSingleStyleFamily {
                    // Ensure the prompt is set if no error and model unloaded
                    if userMessage != styleSectionPrompt {
                        userMessage = styleSectionPrompt
                    }
                }
                // Don't automatically set a message for arbitrary family if model unloads without error,
                // as it might be mid-switch. handleModelFamilyChange sets the initial message.
            }
            // If there's an error, handleServiceError will display it.
        }
    }

    func handleServiceError(error: Error?) {
        if let error = error {
            let description = error.localizedDescription
            var displayMessage = "Error: \(description)"

            if let styleError = error as? StyleTransferService.StyleTransferError {
                switch styleError {
                case .modelLoading:
                    // Keep specific loading messages if already set
                    if !(userMessage?.lowercased().contains("loading") ?? false
                        || userMessage?.lowercased().contains("switching") ?? false)
                    {
                        displayMessage = description  // Use the "Loading selected model..." message
                    } else {
                        displayMessage = userMessage ?? description  // Keep existing specific message
                    }
                case .modelFileNotFound(let name) where name.isEmpty:
                    // Model unloaded intentionally (e.g., reset), don't show as error
                    print("ℹ️ Model unloaded or cleared.")
                    // Keep the prompt message set by handleModelFamilyChange or handleModelLoadingStatus
                    displayMessage = userMessage ?? ""  // Avoid overwriting prompt
                    if displayMessage.isEmpty { return }  // Don't display empty message

                // NEW: Handle style loading errors more gracefully
                case .styleImageNotFound, .styleImageProcessingFailed, .styleImageNotSet:
                    // These errors occur *after* model load for AdaIN/Arbitrary.
                    // Display the error, but don't clear selectedStyle.
                    displayMessage = "Style Input Error: \(description)"
                    print("🔴 \(displayMessage)")

                default:
                    // Use default "Error: description"
                    print("🔴 Service Error Received: \(description)")
                }
            } else {
                // Non-StyleTransferError
                print("🔴 Service Error Received (Non-StyleTransferError): \(description)")
            }

            // Update message only if it's different to avoid redundant UI updates/flicker
            if userMessage != displayMessage {
                userMessage = displayMessage
            }

        } else {
            // Error is nil. Clear the message *only if* it was previously an error message
            // or a transient loading/processing message that wasn't cleared by success.
            if let msg = userMessage,
                messageColor(msg) == .red || msg.lowercased().contains("loading")
                    || msg.lowercased().contains("processing")
                    || msg.lowercased().contains("switching")
            {
                // Check if it's the AdaIN model loaded message, waiting for style input
                let isAdaINModelLoadedMsg = msg.contains("model loaded.") && msg.contains("AdaIN")
                if !isAdaINModelLoadedMsg {
                    userMessage = nil  // Clear the message
                } else {
                    // Keep the "AdaIN model loaded" message until style input loads or fails
                    print("ℹ️ Keeping AdaIN model loaded message while waiting for style input.")
                }
            }
            // Don't clear prompts like "Tap a style..."
        }
    }

    func handleProcessingStatus(processing: Bool) {
        if processing {
            // Message should already be set to "Processing..." by applyStyle()
            // Ensure it is, in case applyStyle logic changes
            if userMessage != "Processing..." {
                userMessage = "Processing..."
            }
            resultImage = nil  // Clear old result when starting new process
        } else {
            // Processing finished.
            // If the message is still "Processing...", it means neither success (handleStyledImageUpdate)
            // nor failure (handleServiceError) updated the message. This indicates an unexpected state
            // or simply that processing finished without error but produced no image (less likely).
            if userMessage == "Processing..." {
                if styleTransferService.error == nil && resultImage == nil {
                    userMessage = "Processing finished."  // Clearer than "unexpectedly"
                    print("⚠️ Processing finished but no result or error was reported.")
                }
                // If error is not nil, handleServiceError should have updated the message.
                // If resultImage is not nil, handleStyledImageUpdate should have cleared the message.
            }
        }
    }

    func saveImage(image: Image) {  // `image` here is the SwiftUI Image from resultImage
        // Ensure coordinator is available
        guard let coordinator = coordinator else {
            userMessage = "Error: Cannot save image (Internal state error)."
            print("🔴 Cannot save: Coordinator not initialized.")
            return
        }

        if #available(iOS 16.0, *) {
            // *** Main Thread ***
            // 1. Set initial user message
            userMessage = "Rendering image for saving..."
            print("ℹ️ Preparing ImageRenderer and rendering UIImage on main thread...")

            // 2. Create the ImageRenderer on the main thread
            let renderer = ImageRenderer(content: image.aspectRatio(contentMode: .fit))
            renderer.scale = UIScreen.main.scale  // Configure on main thread

            // 3. Perform the rendering AND get the UIImage ON THE MAIN THREAD
            guard let uiImage = renderer.uiImage else {
                // Failed to render (still on main thread)
                userMessage = "Error: Could not render image for saving."
                print("🔴 ImageRenderer failed to produce UIImage on main thread.")
                return  // Exit if rendering fails
            }

            // *** Saving ***
            // 4. Call UIImageWriteToSavedPhotosAlbum.
            //    This function handles its own background processing for I/O.
            //    Our callback (`ContentViewCoordinator.image(...)`) ensures UI updates
            //    (like the success/error message) happen back on the main thread.
            //    Running this call from the main thread is generally safe and simpler here.
            print(
                "ℹ️ Calling UIImageWriteToSavedPhotosAlbum from main thread (callback handles UI updates)..."
            )
            UIImageWriteToSavedPhotosAlbum(
                uiImage,
                coordinator,  // Target object for the callback
                #selector(ContentViewCoordinator.image(_:didFinishSavingWithError:contextInfo:)),  // The callback selector
                nil  // Optional context info
            )

        } else {
            // Fallback for older iOS versions
            userMessage = "Error: Saving requires iOS 16 or later."
            print("⚠️ Image saving requires iOS 16+ ImageRenderer.")
        }
    }

    // Share Functionality
    func shareImage(image: Image) {
        if #available(iOS 16.0, *) {
            let renderer = ImageRenderer(content: image.aspectRatio(contentMode: .fit))
            renderer.scale = UIScreen.main.scale
            guard let uiImage = renderer.uiImage else {
                userMessage = "Error: Could not render image for sharing."
                return
            }

            let activityViewController = UIActivityViewController(
                activityItems: [uiImage], applicationActivities: nil)

            // Find the current scene and present the share sheet
            guard let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
                let rootViewController = windowScene.windows.first?.rootViewController
            else {
                userMessage = "Error: Could not find view controller to present share sheet."
                return
            }
            // Ensure presentation on main thread
            DispatchQueue.main.async {
                rootViewController.present(activityViewController, animated: true, completion: nil)
                userMessage = nil  // Clear message after initiating share
            }

        } else {
            userMessage = "Error: Sharing requires iOS 16 or later."
        }
    }
}

// Previews
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
