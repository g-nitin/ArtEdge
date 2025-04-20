import PhotosUI
import SwiftUI

struct ContentView: View {
    // State for UI elements and selections
    @State private var contentImage: UIImage?
    @State private var selectedPhotoItem: PhotosPickerItem?  // For PhotoPicker selection binding
    @State private var selectedStyle: StyleInfo?
    @State private var resultImage: Image?  // Use SwiftUI Image for display
    @State private var userMessage: String?  // For showing errors or status
    // List of available model *filenames* (without .mlmodelc extension)
    // Ensure these files are actually added to your Xcode project target!
    let availableModels: [String] = ["AdaIn", "AesFA"]  // Add your actual model names here
    @State private var selectedModelName: String

    // Initialize StyleTransferService with the initial model
    // Use @StateObject correctly with initial parameter
    @StateObject private var styleTransferService: StyleTransferService

    // Computed property to check if processing can start
    private var canProcess: Bool {
        contentImage != nil && selectedStyle != nil && !styleTransferService.isProcessing
            && styleTransferService.isModelLoaded  // Check if model is loaded AND ready
    }

    let availableStyles: [StyleInfo] = [
        StyleInfo(name: "Starry Night", assetName: "starry_night"),
        StyleInfo(name: "The Scream", assetName: "the_scream"),
        StyleInfo(name: "Composition VII", assetName: "composition_vii"),
        StyleInfo(name: "Brushstrokes", assetName: "brushstrokes"),
    ]

    // Custom Initializer for StateObject
    init() {
        // Set the initial model name here
        let initialModelName = "AesFA"
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
                        // Define the desired frame for the *visible* banner area
                        // Use maxWidth: .infinity to take available width
                        // Set the desired *visual height* for your banner content
                        .frame(maxWidth: .infinity)
                        .clipped()
                        .padding(.bottom, 10)  // Add space below the banner
                    // Optional: Add horizontal padding if you DON'T want it edge-to-edge
                    // .padding(.horizontal)
                    // Optional: Add a background or divider if desired
                    // Color: #FCF7F3
                    // .background(Color(red: 252/255, green: 247/255, blue: 243/255))

                    ScrollView {
                        VStack(spacing: 15) {  // Original spacing for content sections
                            modelSelectionSection
                            Divider()
                            contentImageSection
                            Divider()
                            styleSelectionSection
                            Divider()
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
                // Apply the same background inside the navigation view
                .background(Color(red: 252 / 255, green: 247 / 255, blue: 243 / 255))
            }
        }

        // Load UIImage when photo item changes
        .onChange(of: selectedPhotoItem) { newItem in
            handlePhotoSelection(newItem: newItem)
        }

        .onChange(of: selectedModelName) { newName in
            print("ContentView: Detected model selection change to '\(newName)'")
            // Reset UI state related to previous model's results
            resultImage = nil
            // Optionally clear content/style if models are incompatible
            // contentImage = nil
            // selectedStyle = nil
            // styleTransferService.loadStyleImage(named: "") // Clear style in service?

            // Set loading message immediately
            userMessage = "Switching to model \(newName)..."
            // Tell the service to switch
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
        // Optional: Add onAppear if initial load needs specific handling
        // .onAppear {
        //     if !styleTransferService.isModelLoaded && styleTransferService.error == nil {
        //         // If service was initialized without a model or failed initially
        //         userMessage = "Loading initial model..."
        //         // You might trigger loadModel here if not done in init
        //         // styleTransferService.loadModel(named: selectedModelName)
        //     }
        // }
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
                    .overlay(Text("Select an image").foregroundColor(.secondary))
            }

            PhotosPicker(
                selection: $selectedPhotoItem,  // Bind selection
                matching: .images,  // Filter for images
                photoLibrary: .shared()  // Use shared library
            ) {  // Provide the label content in the trailing closure
                Label("Select from Library", systemImage: "photo.on.rectangle")
            }
            .buttonStyle(.bordered)
            // TODO: Add "Take Photo" button here later if needed
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
                Text("Tap a style below")  // Placeholder
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            ScrollView(.horizontal, showsIndicators: false) {
                LazyHStack(spacing: 15) {
                    ForEach(availableStyles) { style in
                        VStack {
                            Image(style.assetName)  // Assumes images are in Assets
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
                                    selectedStyle = style
                                    // Load the style image into the service
                                    styleTransferService.loadStyleImage(named: style.assetName)
                                    userMessage = nil  // Clear message on style selection
                                }
                            Text(style.name).font(.caption).lineLimit(1)
                        }
                    }
                }
                .padding(.horizontal)
                .padding(.vertical, 5)
            }
            .frame(height: 110)  // Give the scroll view a fixed height
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
        .disabled(!canProcess)  // Disable if not ready or processing
        .padding(.top)
    }

    private var userMessageSection: some View {
        Group {  // Use Group to return optional view
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
        guard let content = contentImage, let style = selectedStyle else {
            userMessage = "Please select both a content image and a style."
            return
        }

        // Ensure the correct style is loaded in the service (might be redundant if already done on tap, but safe)
        styleTransferService.loadStyleImage(named: style.assetName)

        print("🚀 Triggering style transfer process...")
        styleTransferService.process(contentImage: content)
    }

    // Handle Photo Picker Selection
    func handlePhotoSelection(newItem: PhotosPickerItem?) {
        Task {  // Use Task for async work
            // Clear previous state immediately for better UX
            contentImage = nil
            resultImage = nil
            userMessage = "Loading image..."

            guard let item = newItem else {
                userMessage = nil  // Clear message if selection is cancelled
                return
            }

            do {
                if let data = try await item.loadTransferable(type: Data.self) {
                    if let uiImage = UIImage(data: data) {
                        contentImage = uiImage
                        userMessage = nil  // Clear message
                        print("✅ Content image loaded.")
                    } else {
                        userMessage = "Error: Could not decode selected image."
                        print("🔴 Failed to create UIImage from selected data.")
                    }
                } else {
                    userMessage = "Error: Could not load image data."
                    print("🔴 Failed to load transferable data from PhotosPickerItem.")
                }
            } catch {
                userMessage = "Error loading image: \(error.localizedDescription)"
                print("🔴 Error loading transferable data: \(error)")
            }
        }
    }

    // Handle Model Loading Status
    func handleModelLoadingStatus(loaded: Bool) {
        // Update message based on loading status, avoid overwriting specific errors
        if !loaded && styleTransferService.error == nil {
            // This might briefly show if switching starts before the loading message is set
            // userMessage = "Loading style transfer model..."
        } else if loaded {
            // Check if the current message indicates loading, then clear it
            if userMessage == "Loading initial model..."
                || userMessage == "Switching to model \(selectedModelName)..."
                || userMessage == "Loading selected model..."
            {
                userMessage = "Model '\(selectedModelName)' loaded. Ready."
                // Optionally clear the "Ready" message after a delay
                DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                    if userMessage == "Model '\(selectedModelName)' loaded. Ready." {
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
            userMessage = "Processing..."  // Set status message when processing starts
        } else if userMessage == "Processing..." {
            // If processing finished and the message was "Processing...",
            // clear it *only if* there's no error and there is a result.
            // If there's an error, the error handler will set the message.
            // If there's no result (conversion failed), that error handler sets the message.
            if styleTransferService.error == nil && resultImage != nil {
                userMessage = nil  // Clear message on successful completion
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
