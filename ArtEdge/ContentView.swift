import PhotosUI
import SwiftUI

struct ContentView: View {
    // State for UI elements and selections
    @State private var contentImage: UIImage?
    @State private var selectedPhotoItem: PhotosPickerItem?
    @State private var selectedStyle: StyleInfo?
    @State private var resultImage: Image?
    @State private var userMessage: String?
    // List of available model *filenames*
    let availableModels: [String] = ["AdaIn", "AesFA", "StarryNightStyleTransfer"]  // Ensure StarryNightStyleTransfer is added
    @State private var selectedModelName: String

    // State to control camera picker presentation
    @State private var showCameraPicker = false

    // Initialize StyleTransferService with the initial model
    // Use @StateObject correctly with initial parameter
    @StateObject private var styleTransferService: StyleTransferService

    // Computed property to check if processing can start
    private var canProcess: Bool {
        // Allow processing if:
        // - Content image exists AND
        // - EITHER it's a single-input model OR a style is selected for dual-input models AND
        // - Not currently processing AND
        // - Model is loaded and ready
        contentImage != nil && (styleTransferService.isSingleInputModel || selectedStyle != nil)  // Adjusted logic
            && !styleTransferService.isProcessing && styleTransferService.isModelLoaded
    }

    // Available styles (only relevant for dual-input models)
    let availableStyles: [StyleInfo] = [
        StyleInfo(name: "Starry Night", assetName: "starry_night"),
        StyleInfo(name: "The Scream", assetName: "the_scream"),
        StyleInfo(name: "Composition VII", assetName: "composition_vii"),
        StyleInfo(name: "Brushstrokes", assetName: "brushstrokes"),
    ]

    // Custom Initializer for StateObject
    init() {
        // Set the initial model name here
        let initialModelName = "AdaIn"
        _selectedModelName = State(initialValue: initialModelName)
        // Initialize the StateObject with the initial model name
        _styleTransferService = StateObject(
            wrappedValue: StyleTransferService(modelName: initialModelName))
        print("ContentView init: Initializing with model '\(initialModelName)'")
    }

    var body: some View {

        ZStack {
            // Set the global background color to #FCF7F3
            Color(red: 252 / 255, green: 247 / 255, blue: 243 / 255)
                .ignoresSafeArea()
            NavigationView {  // NavigationView for title and structure
                VStack(spacing: 0) {  // Use spacing 0 initially, control with padding
                    Image("ArtEdgeLogoBanner")
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: .infinity)
                        .clipped()
                        .padding(.bottom, 10)

                    ScrollView {
                        VStack(spacing: 15) {
                            modelSelectionSection
                            Divider()
                            contentImageSection
                            Divider()
                            // Conditionally show style selection
                            if !styleTransferService.isSingleInputModel {
                                styleSelectionSection
                                Divider()
                            } else {
                                // Optional: Show a message instead of style selection
                                Text("Style is built into the selected model.")
                                    .font(.headline)
                                    .foregroundColor(.secondary)
                                    .padding(.horizontal)
                                    .padding(.vertical, 10)  // Add some spacing
                                Divider()
                            }
                            actionButtonSection
                            userMessageSection
                            resultSection
                            Spacer()  // Pushes content up if it's short
                        }
                        .padding(.top)  // Add padding between banner and first content section
                    }

                }
                .navigationTitle("")
                .navigationBarTitleDisplayMode(.inline)
                .background(Color(red: 252 / 255, green: 247 / 255, blue: 243 / 255))
            }
        }

        // Load UIImage when photo item changes
        .onChange(of: selectedPhotoItem) { newItem in
            handlePhotoSelection(newItem: newItem)
        }
        // Clear photo item if image is set via camera
        .onChange(of: contentImage) { newImage in
            if newImage != nil && selectedPhotoItem != nil {
                // If we set the image (likely from camera), deselect the PhotosPickerItem
                // to avoid potential confusion or re-triggering library load.
                selectedPhotoItem = nil
            }
        }
        .onChange(of: selectedModelName) { newName in
            print("ContentView: Detected model selection change to '\(newName)'")
            resultImage = nil
            userMessage = "Switching to model \(newName)..."
            // Clear selected style if switching TO a single-input model (or always clear?)
            // Let's clear it always for simplicity when switching models
            selectedStyle = nil
            styleTransferService.switchModel(to: newName)
        }
        .onReceive(styleTransferService.$styledImage) { newStyledImage in
            self.resultImage = newStyledImage
            // Clear processing message only if successful
            if newStyledImage != nil && userMessage == "Processing..." {
                userMessage = nil
            }
        }
        // Update user message on error
        .onReceive(styleTransferService.$error) { error in
            handleServiceError(error: error)
        }
        // Update message based on model loading status
        .onReceive(styleTransferService.$isModelLoaded) { loaded in
            handleModelLoadingStatus(loaded: loaded)
        }
        // Update message based on processing status
        .onReceive(styleTransferService.$isProcessing) { processing in
            handleProcessingStatus(processing: processing)
        }
        // React to model type change to potentially clear messages
        .onReceive(styleTransferService.$isSingleInputModel) { isSingle in
            if isSingle && selectedStyle != nil {
                // If we switched to single input and a style was selected, clear it
                selectedStyle = nil
                // Maybe update user message?
                // userMessage = "Style selection disabled for this model."
            }
            // Clear style-related errors if switching to single input
            if isSingle,
                let currentError = styleTransferService.error
                    as? StyleTransferService.StyleTransferError,
                currentError == .styleImageNotSet
            {
                styleTransferService.error = nil  // Clear the error
                userMessage = nil  // Clear the message too
            }
        }
        .fullScreenCover(isPresented: $showCameraPicker) {
            ImagePicker(image: $contentImage, isPresented: $showCameraPicker, sourceType: .camera)
        }
    }

    private var modelSelectionSection: some View {
        VStack(alignment: .leading) {  // Align content to the left
            Text("Select Model").font(.headline)
                .padding(.horizontal)  // Add padding to match other sections

            Picker("Model", selection: $selectedModelName) {
                ForEach(availableModels, id: \.self) { modelName in
                    Text(modelName).tag(modelName)  // Use filename as tag
                }
            }
            .pickerStyle(.segmented)  // Or .menu for a dropdown
            .padding(.horizontal)  // Add padding to the picker
            .disabled(styleTransferService.isProcessing)  // Disable while processing
        }
        // Add padding below the section if needed
        // .padding(.bottom, 10)
    }

    private var contentImageSection: some View {
        VStack {
            Text("1. Choose Content Image").font(.headline)
            if let contentImage = contentImage {
                Image(uiImage: contentImage)
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 200)
                    .cornerRadius(8)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.gray, lineWidth: 1)
                    )
            } else {
                Rectangle()
                    .fill(Color.secondary.opacity(0.1))
                    .frame(height: 150)
                    .cornerRadius(8)
                    .overlay(Text("Select or take a photo").foregroundColor(.secondary))
            }

            HStack(spacing: 15) {  // Use HStack to place buttons side-by-side
                // Photos Picker Button
                PhotosPicker(
                    selection: $selectedPhotoItem,
                    matching: .images,
                    photoLibrary: .shared()
                ) {
                    Label("Library", systemImage: "photo.on.rectangle")
                }
                .buttonStyle(.bordered)

                // Take Photo Button
                Button {
                    // Action to show the camera picker
                    showCameraPicker = true
                } label: {
                    Label("Photo", systemImage: "camera.fill")
                }
                .buttonStyle(.bordered)
                // Disable if camera is not available (e.g., simulator)
                .disabled(!UIImagePickerController.isSourceTypeAvailable(.camera))

            }  // End HStack
            .padding(.top, 5)  // Add a little space above the buttons

        }
        .padding(.horizontal)
    }

    private var styleSelectionSection: some View {
        VStack {
            Text("2. Select Style").font(.headline)
            if selectedStyle != nil {
                Text("Selected: \(selectedStyle!.name)")
                    .font(.caption)
                    .foregroundColor(.blue)
            } else {
                Text("Tap a style below")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            ScrollView(.horizontal, showsIndicators: false) {
                LazyHStack(spacing: 15) {
                    ForEach(availableStyles) { style in
                        VStack {
                            Image(style.assetName)
                                .resizable()
                                .scaledToFill()
                                .frame(width: 80, height: 80)
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 8)
                                        .stroke(
                                            selectedStyle == style ? Color.blue : Color.gray,
                                            lineWidth: selectedStyle == style ? 3 : 1)
                                )
                                .onTapGesture {
                                    // Only process tap if it's NOT a single input model
                                    // (Although this section is hidden, add check for safety)
                                    guard !styleTransferService.isSingleInputModel else { return }
                                    selectedStyle = style
                                    // Load the style image in the service
                                    styleTransferService.loadStyleImage(named: style.assetName)
                                    // Clear user message if it was a style error
                                    if let currentError = styleTransferService.error
                                        as? StyleTransferService.StyleTransferError,
                                        currentError == .styleImageNotSet
                                    {
                                        userMessage = nil
                                    }
                                }
                            Text(style.name).font(.caption).lineLimit(1)
                        }
                    }
                }
                .padding(.horizontal)
                .padding(.vertical, 5)
            }
            .frame(height: 110)
            // Disable interaction if processing OR if it's a single input model (redundant but safe)
            .disabled(styleTransferService.isProcessing || styleTransferService.isSingleInputModel)
        }
    }

    private var actionButtonSection: some View {
        Button {
            applyStyle()
        } label: {
            if styleTransferService.isProcessing {
                HStack {
                    ProgressView()
                        .padding(.trailing, 5)
                    Text("Applying Style...")
                }
            } else {
                Label("Apply Style", systemImage: "paintbrush.pointed.fill")
            }
        }
        .buttonStyle(.borderedProminent)
        .disabled(!canProcess)
        .padding(.top)
    }

    private var userMessageSection: some View {
        Group {
            if let message = userMessage {
                Text(message)
                    .foregroundColor(styleTransferService.error != nil ? .red : .secondary)
                    .font(.caption)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
                    .padding(.top, 5)  // Add some spacing
            } else {
                // Optionally add an empty Text view to maintain layout stability
                Text(" ").font(.caption)
                    .padding(.horizontal)
                    .padding(.top, 5)
            }
        }
    }

    private var resultSection: some View {
        Group {  // Use Group as the container might be empty
            if styleTransferService.isProcessing {
                Spacer()  // Push result area down while processing
                // Optionally add a small indicator here too
                // ProgressView("Processing...")
                //    .padding(.top)
            } else if let result = resultImage {
                VStack {
                    Text("Result").font(.headline).padding(.top)
                    result
                        .resizable()
                        .scaledToFit()
                        .frame(maxHeight: 250)
                        .cornerRadius(8)
                    Text(
                        "Processing Time: \(String(format: "%.1f", styleTransferService.processingTime)) ms"
                    )
                    .font(.caption)
                    .foregroundColor(.secondary)
                }
                .padding(.horizontal)
                Spacer()  // Push result to bottom when available
            } else {
                Spacer()  // Keep layout consistent when no result/not processing
            }
        }
    }

    // Function to trigger style transfer
    func applyStyle() {
        guard let content = contentImage else {
            userMessage = "Please select a content image."
            return
        }

        // Check if a style is required (i.e., not a single-input model) AND if it's missing
        if !styleTransferService.isSingleInputModel && selectedStyle == nil {
            userMessage = "Please select a style for the '\(selectedModelName)' model."
            // Ensure the service reflects this error state too
            styleTransferService.error = StyleTransferService.StyleTransferError.styleImageNotSet
            return
        }

        // If it's a dual-input model, ensure the selected style is loaded (should be done on tap, but double-check)
        if !styleTransferService.isSingleInputModel, let style = selectedStyle {
            styleTransferService.loadStyleImage(named: style.assetName)
            // Check if loading the style failed immediately before processing
            if let currentError = styleTransferService.error
                as? StyleTransferService.StyleTransferError,
                currentError == .styleImageNotFound || currentError == .styleImageProcessingFailed
            {
                userMessage =
                    "Error: Could not load the selected style image. \(currentError.localizedDescription)"
                return
            }
        }

        print("🚀 Triggering style transfer process...")
        styleTransferService.process(contentImage: content)
    }

    // Handle Photo Picker Selection
    func handlePhotoSelection(newItem: PhotosPickerItem?) {
        Task {
            // Clear previous state immediately
            // Don't clear contentImage here, let the loading handle it
            // contentImage = nil
            resultImage = nil
            userMessage = "Loading image..."  // Show loading message

            guard let item = newItem else {
                // Only clear message if no image is currently loaded
                if contentImage == nil {
                    userMessage = nil
                } else {
                    // Keep the existing image, clear the loading message
                    userMessage = nil
                }
                return
            }

            do {
                if let data = try await item.loadTransferable(type: Data.self) {
                    if let uiImage = UIImage(data: data) {
                        // Successfully loaded image from library
                        contentImage = uiImage  // Update the content image
                        userMessage = nil  // Clear message
                        print("✅ Content image loaded from library.")
                    } else {
                        userMessage = "Error: Could not decode selected image."
                        print("🔴 Failed to create UIImage from selected data.")
                        // Keep existing image if decoding fails? Or set to nil?
                        // contentImage = nil // Optional: clear image on decode failure
                    }
                } else {
                    userMessage = "Error: Could not load image data."
                    print("🔴 Failed to load transferable data from PhotosPickerItem.")
                    // Keep existing image if loading fails? Or set to nil?
                    // contentImage = nil // Optional: clear image on load failure
                }
            } catch {
                userMessage = "Error loading image: \(error.localizedDescription)"
                print("🔴 Error loading transferable data: \(error)")
                // Keep existing image on error? Or set to nil?
                // contentImage = nil // Optional: clear image on error
            }
        }
    }

    // Handle Model Loading Status
    func handleModelLoadingStatus(loaded: Bool) {
        if !loaded && styleTransferService.error == nil {
            // Loading initiated, message handled by onChange(of: selectedModelName)
        } else if loaded {
            if userMessage == "Loading initial model..."
                || userMessage == "Switching to model \(selectedModelName)..."
                || userMessage == "Loading selected model..."
            {
                userMessage = "Model '\(selectedModelName)' loaded. Ready."
                // Add specific info if it's single input
                if styleTransferService.isSingleInputModel {
                    userMessage = "Model '\(selectedModelName)' loaded (Style built-in). Ready."
                }

                DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {  // Slightly longer delay
                    if userMessage?.contains("loaded. Ready.") ?? false {
                        userMessage = nil
                    }
                }
            }
        }
        // If loaded is false AND there's an error, handleServiceError will display it.
    }

    // Handle Errors from Service
    func handleServiceError(error: Error?) {
        if let error = error {
            // Don't show the generic "Loading..." message if a real error occurred
            if let styleError = error as? StyleTransferService.StyleTransferError,
                styleError == .modelLoading
            {
                // Keep the "Loading selected model..." message
                userMessage = styleError.localizedDescription
            } else {
                // Show actual error description
                userMessage = "Error: \(error.localizedDescription)"
                print("🔴 Service Error: \(error.localizedDescription)")
            }
        }
        // Don't clear message here automatically
    }

    // Handle Processing Status Updates
    func handleProcessingStatus(processing: Bool) {
        if processing {
            userMessage = "Processing..."
        } else if userMessage == "Processing..." {
            if styleTransferService.error == nil && resultImage != nil {
                userMessage = nil
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
