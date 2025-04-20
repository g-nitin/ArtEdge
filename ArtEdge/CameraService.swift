import AVFoundation
import Combine
import CoreImage
import UIKit

// Manages the camera session and provides video frames.
class CameraService: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {

    @Published var currentFrame: CVPixelBuffer?  // Publishes the current frame

    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "com.artedge.sessionQueue", qos: .userInitiated)
    private var permissionGranted = false
    private var device: AVCaptureDevice?
    // Add the RotationCoordinator property (available iOS 17+)
    private var rotationCoordinator: AVCaptureDevice.RotationCoordinator?

    override init() {
        super.init()
        checkPermission()
    }

    // Check for camera permissions
    func checkPermission() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            print("Camera permission already granted.")
            permissionGranted = true
            // Call setup and start session *after* confirming permission
            sessionQueue.async { [weak self] in
                self?.setupSession()  // Setup first
                self?.startSession()  // Then start
            }
        case .notDetermined:
            requestPermission()
        default:
            permissionGranted = false
            print("Camera permission denied or restricted.")
        // TODO: Handle this case appropriately in the UI if needed
        }
    }

    // Request camera permissions
    private func requestPermission() {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            self?.permissionGranted = granted
            if granted {
                print("Camera permission granted by user.")
                // Call setup and start session *after* getting permission
                self?.sessionQueue.async {
                    self?.setupSession()  // Setup first
                    self?.startSession()  // Then start
                }
            } else {
                print("Camera permission denied by user.")
                // TODO: Handle denial in UI if needed
            }
        }
    }

    // Configure the AVCaptureSession
    private func setupSession() {
        // Check permission *inside* setupSession as well, as a safeguard
        guard permissionGranted else {
            print("Permission not granted, cannot setup session.")
            return
        }

        // Prevent running setup multiple times if called accidentally
        guard !captureSession.isRunning && captureSession.inputs.isEmpty else {
            print("Session already set up or running.")
            return
        }

        print("Setting up capture session...")  // Added log

        captureSession.beginConfiguration()
        captureSession.sessionPreset = .hd1280x720

        guard
            let videoDevice = AVCaptureDevice.default(
                .builtInWideAngleCamera, for: .video, position: .back)
        else {
            print("Could not find back camera.")
            captureSession.commitConfiguration()
            return
        }
        self.device = videoDevice  // Assign the device

        guard let videoDeviceInput = try? AVCaptureDeviceInput(device: videoDevice),
            captureSession.canAddInput(videoDeviceInput)
        else {
            print("Could not create video device input or add it to session.")
            captureSession.commitConfiguration()
            return
        }
        captureSession.addInput(videoDeviceInput)

        videoOutput.setSampleBufferDelegate(
            self, queue: DispatchQueue(label: "com.artedge.videoOutputQueue", qos: .userInitiated))
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]

        guard captureSession.canAddOutput(videoOutput) else {
            print("Could not add video data output to the session")
            captureSession.commitConfiguration()
            return
        }
        captureSession.addOutput(videoOutput)

        // Set mirroring based on camera position (still relevant)
        if let connection = videoOutput.connection(with: .video) {
            if connection.isVideoMirroringSupported {
                connection.isVideoMirrored = (videoDevice.position == .front)
            }
        }

        captureSession.commitConfiguration()
        print("Capture session setup complete.")
    }

    // Method to set up Rotation Coordinator
    // This needs to be called AFTER the preview layer is created.
    func setupRotationCoordinator(previewLayer: AVCaptureVideoPreviewLayer) {
        guard let device = self.device else {
            // This error might still appear briefly if makeUIView runs before setupSession completes
            // but it should eventually succeed once the device is set.
            print(
                "Warning: Device not available for RotationCoordinator setup yet. Will retry if needed or rely on fallback."
            )
            return
        }
        if #available(iOS 17.0, *) {
            // Create the coordinator, linking device rotation to the preview and video output
            rotationCoordinator = AVCaptureDevice.RotationCoordinator(
                device: device, previewLayer: previewLayer)
            // The coordinator now handles orientation/rotation automatically.
            print("RotationCoordinator setup complete.")

            // Optional: Set initial rotation angle if needed (coordinator might override)
            // if let connection = videoOutput.connection(with: .video), connection.isVideoRotationAngleSupported(90) {
            //     connection.videoRotationAngle = 90 // Example: Portrait
            // }
            // if let previewConnection = previewLayer.connection, previewConnection.isVideoRotationAngleSupported(90) {
            //     previewConnection.videoRotationAngle = 90 // Example: Portrait
            // }

        } else {
            // Fallback for iOS < 17: Manually set initial orientation
            if let connection = videoOutput.connection(with: .video) {
                if connection.isVideoOrientationSupported {
                    connection.videoOrientation = .portrait
                }
            }
            if let previewConnection = previewLayer.connection {
                if previewConnection.isVideoOrientationSupported {
                    previewConnection.videoOrientation = .portrait
                }
            }
            print("Using legacy orientation for iOS < 17.")
        }
    }

    // Start the capture session (no changes needed)
    func startSession() {
        sessionQueue.async { [weak self] in
            // Only start if permission is granted and session is not already running
            guard let self = self, self.permissionGranted, !self.captureSession.isRunning else {
                return
            }
            // Ensure setup has actually added inputs/outputs before starting
            guard !self.captureSession.inputs.isEmpty, !self.captureSession.outputs.isEmpty else {
                print("Attempted to start session before it was fully configured.")
                return
            }
            self.captureSession.startRunning()
            print("Capture session started.")
        }
    }

    // Stop the capture session (no changes needed)
    func stopSession() {
        sessionQueue.async { [weak self] in
            guard let self = self, self.captureSession.isRunning else { return }
            self.captureSession.stopRunning()
            print("Capture session stopped.")
        }
    }

    // AVCaptureVideoDataOutputSampleBufferDelegate method (no changes needed here)
    func captureOutput(
        _ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        DispatchQueue.main.async {
            self.currentFrame = pixelBuffer
        }
    }

    // Provide the session for the preview layer
    var session: AVCaptureSession {
        return captureSession
    }
}
