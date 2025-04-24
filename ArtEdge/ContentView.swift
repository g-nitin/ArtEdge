import PhotosUI
import SwiftUI
import UIKit

enum ModelFamily: String, CaseIterable, Identifiable {
    case adain = "AdaIN"  // Single-style models, loaded via style selection
    case fst = "FST"  // Single-style models, loaded via style selection
    case aesfa = "AesFA"  // Arbitrary model, loaded on family selection
    case stytr2 = "StyTR2"  // Arbitrary model, loaded on family selection

    var id: String { self.rawValue }

    // Helper to get the primary model filename for arbitrary types
    // Returns nil for families where the model is loaded via style selection
    var primaryModelFilename: String? {
        switch self {
        case .adain: return nil  // Model determined by style selection
        case .fst: return nil  // Model determined by style selection
        case .aesfa: return "AesFA"  // This model is loaded when family is selected
        case .stytr2: return "StyTr2"  // This model is loaded when family is selected (Corrected filename)
        }
    }

    // Check if this family uses single-style models loaded via style tap
    var isSingleStyleFamily: Bool {
        switch self {
        case .adain, .fst: return true
        case .aesfa, .stytr2: return false
        }
    }

    // Check if this family represents an arbitrary style transfer model
    var isArbitraryStyleFamily: Bool {
        switch self {
        case .adain, .fst: return false
        case .aesfa, .stytr2: return true
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
        let modelReady = styleTransferService.isModelLoaded  // A model (either arbitrary or specific single-style) must be loaded
        let serviceIdle = !styleTransferService.isProcessing
        var styleRequirementMet = false

        if selectedModelFamily.isArbitraryStyleFamily {
            // Arbitrary models require a style *input* to be selected
            styleRequirementMet = (selectedStyle != nil)
            // Also ensure the loaded model is NOT single-input (sanity check)
            styleRequirementMet = styleRequirementMet && !styleTransferService.isSingleInputModel
        } else if selectedModelFamily.isSingleStyleFamily {
            // Single-style families require a style to have been tapped (which loads the model)
            // The modelReady check covers that the model is loaded.
            // We also need selectedStyle to be non-nil because tapping sets it.
            styleRequirementMet = (selectedStyle != nil)
            // Also ensure the loaded model IS single-input (sanity check)
            styleRequirementMet = styleRequirementMet && styleTransferService.isSingleInputModel
        }

        // Combine all conditions
        let allMet = hasContent && modelReady && serviceIdle && styleRequirementMet

        // Debugging log for canProcess state changes
        // print("ℹ️ canProcess check: hasContent=\(hasContent), modelReady=\(modelReady), serviceIdle=\(serviceIdle), styleReqMet=\(styleRequirementMet) -> \(allMet)")

        return allMet
    }

    // Custom Initializer
    init() {
        // Start with an arbitrary family (e.g., AesFA) so a model loads initially
        let initialFamily: ModelFamily = .aesfa
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
        .onReceive(styleTransferService.$isModelLoaded) { loaded in
            handleModelLoadingStatus(loaded: loaded)
        }
        .onReceive(styleTransferService.$isProcessing) { processing in
            handleProcessingStatus(processing: processing)
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
            // Disable while processing OR if a single-style model is currently loading (indicated by isModelLoaded=false AND selectedStyle being set)
            .disabled(
                styleTransferService.isProcessing
                    || (selectedModelFamily.isSingleStyleFamily
                        && !styleTransferService.isModelLoaded && selectedStyle != nil)
            )
        }
    }

    // Helper computed properties for style section UI
    private var styleSectionTitle: String {
        switch selectedModelFamily {
        case .adain, .fst:
            return "3. Select Style (Loads Model)"  // Step 3 for these families
        case .aesfa, .stytr2:
            return "3. Select Style Input"  // Step 3 for these families
        }
    }

    private var styleSectionPrompt: String {
        switch selectedModelFamily {
        case .adain: return "Tap an AdaIN style below to load its model"
        case .fst: return "Tap an FST style below to load its model"
        case .aesfa, .stytr2: return "Tap a style to use as input"
        }
    }

    // Dynamically returns the correct list of StyleInfo based on the selected family
    private var currentStyleList: [StyleInfo] {
        switch selectedModelFamily {
        case .adain: return availableAdaINStyles
        case .fst: return availableFSTStyles
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
        userMessage = nil  // Clear previous message

        if selectedModelFamily.isArbitraryStyleFamily {
            // Load the tapped style as *input data* for the already loaded arbitrary model
            print("-> Loading style asset '\(style.assetName)' as input.")
            userMessage = "Loading style input: \(style.name)..."  // Provide feedback
            styleTransferService.loadStyleImage(named: style.assetName)
            // Error handling for style loading is done in handleServiceError

        } else if selectedModelFamily.isSingleStyleFamily {
            // Load the specific single-style *model* associated with this style
            if let modelToLoad = style.associatedModelFilename {
                print("-> Switching to single-style model: '\(modelToLoad)'")
                userMessage = "Loading \(style.name) (\(selectedModelFamily.rawValue)) model..."
                styleTransferService.switchModel(to: modelToLoad)
            } else {
                // This shouldn't happen if StyleInfo is set up correctly
                let errorMsg = "Error: Configuration issue for style \(style.name)."
                print("🔴 \(errorMsg)")
                userMessage = errorMsg
                styleTransferService.error = StyleTransferService.StyleTransferError
                    .modelFileNotFound("Associated with \(style.name)")
            }
        }
    }

    func applyStyle() {
        guard let content = contentImage else {
            userMessage = "Please select a content image first."
            return
        }

        // Use canProcess to simplify the guard, but keep specific messages
        guard canProcess else {
            // Provide more specific feedback why it's disabled
            if contentImage == nil {
                userMessage = "Please select a content image."
            } else if !styleTransferService.isModelLoaded {
                if selectedModelFamily.isSingleStyleFamily {
                    userMessage =
                        "Please select a \(selectedModelFamily.rawValue) style to load the model."
                } else {
                    userMessage =
                        "\(selectedModelFamily.rawValue) model is not loaded or failed to load."
                }
            } else if selectedStyle == nil {
                if selectedModelFamily.isArbitraryStyleFamily {
                    userMessage =
                        "Please select a style input for the \(selectedModelFamily.rawValue) model."
                } else {
                    userMessage = "Please select a \(selectedModelFamily.rawValue) style."
                }
            } else if (selectedModelFamily.isArbitraryStyleFamily
                && styleTransferService.isSingleInputModel)
                || (selectedModelFamily.isSingleStyleFamily
                    && !styleTransferService.isSingleInputModel)
            {
                userMessage =
                    "Error: Loaded model type doesn't match selected family. Please re-select style/family."
                print(
                    "🔴 Mismatch: Family=\(selectedModelFamily.rawValue), isSingleInputModel=\(styleTransferService.isSingleInputModel)"
                )
            } else if styleTransferService.isProcessing {
                userMessage = "Already processing..."  // Should be handled by button state, but good fallback
            } else {
                userMessage = "Cannot apply style. Check selections."  // Generic fallback
            }
            return
        }

        // If all checks pass
        print(
            "🚀 Triggering style transfer. Family: \(selectedModelFamily.rawValue), Style: \(selectedStyle!.name), Model Loaded: \(styleTransferService.isModelLoaded), Is Single Input: \(styleTransferService.isSingleInputModel)"
        )
        // Set processing message *before* calling process
        userMessage = "Processing..."
        styleTransferService.process(contentImage: content)
    }

    func handlePhotoSelection(newItem: PhotosPickerItem?) {
        // Clear previous result immediately
        resultImage = nil

        // Set loading message immediately
        // Use a temporary flag to ensure message is set only once per selection attempt
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
        let wasLoading =
            userMessage?.lowercased().contains("loading") ?? false
            || userMessage?.lowercased().contains("switching") ?? false

        if loaded {
            if wasLoading {
                let modelType =
                    styleTransferService.isSingleInputModel ? "(Single-Style)" : "(Arbitrary)"
                let specificName = styleName ?? familyName  // Use style name if available, else family name
                let successMsg = "'\(specificName)' \(modelType) loaded successfully."
                userMessage = successMsg
                print("✅ \(successMsg)")
                // Optionally clear the message after a delay
                DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
                    if userMessage == successMsg {
                        userMessage = nil
                    }
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
            // If it was reset intentionally (e.g., switching to AdaIN/FST family), the message should be the prompt.
            if styleTransferService.error == nil {
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
            // Don't show the generic "Loading..." message if a real error occurred
            if let styleError = error as? StyleTransferService.StyleTransferError,
                styleError == .modelLoading
            {
                // Keep the specific "Loading..." or "Switching..." message if it's already set by handleStyleTap/handleModelFamilyChange
                if !(userMessage?.lowercased().contains("loading \(selectedStyle?.name ?? "")")
                    ?? false
                    || userMessage?.lowercased().contains(
                        "switching to \(selectedModelFamily.rawValue)") ?? false)
                {
                    // Only set generic loading if a specific one isn't present
                    if userMessage != description { userMessage = description }
                }
            }
            // Handle case where resetService might trigger modelFileNotFound("")
            else if let styleError = error as? StyleTransferService.StyleTransferError,
                case .modelFileNotFound(let name) = styleError, name.isEmpty
            {
                print("ℹ️ Model unloaded or cleared by resetService.")
                // Message should have been set by the action triggering resetService (e.g., family change prompt).
                // Avoid overwriting the prompt.
            }
            // For all other errors, display the description
            else {
                // Update message only if it's different to avoid redundant UI updates
                if userMessage != "Error: \(description)" {
                    userMessage = "Error: \(description)"
                }
                print("🔴 Service Error Received: \(description)")
            }
        } else {
            // Error is nil. Clear the message *only if* it was previously an error message
            // or a transient loading message that wasn't cleared by success.
            if let msg = userMessage,
                messageColor(msg) == .red || msg.lowercased().contains("loading")
                    || msg.lowercased().contains("processing")
                    || msg.lowercased().contains("switching")
            {
                userMessage = nil
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
            // nor failure (handleServiceError) updated the message. This indicates an unexpected state.
            if userMessage == "Processing..." {
                if styleTransferService.error == nil && resultImage == nil {
                    userMessage = "Processing finished unexpectedly."  // Or clear: userMessage = nil
                    print("⚠️ Processing finished but no result or error was reported.")
                }
                // If error is not nil, handleServiceError should have updated the message.
                // If resultImage is not nil, handleStyledImageUpdate should have cleared the message.
            }
        }
    }

    // MARK: - Save Image Function

    func saveImage(image: Image) {  // `image` here is the SwiftUI Image from resultImage
        // Ensure coordinator is available
        guard let coordinator = coordinator else {
            userMessage = "Error: Cannot save image (Internal state error)."
            print("🔴 Cannot save: Coordinator not initialized.")
            return
        }

        if #available(iOS 16.0, *) {
            // --- Main Thread ---
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

            // --- Saving ---
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
