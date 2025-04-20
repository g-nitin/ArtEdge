import SwiftUI
import AVFoundation

// A SwiftUI View that displays the camera preview layer.
struct CameraPreviewView: UIViewRepresentable {
    // Inject the CameraService instead of just the session
    @ObservedObject var cameraService: CameraService

    // Custom UIView subclass remains the same
    class PreviewUIView: UIView {
        var previewLayer: AVCaptureVideoPreviewLayer?
        override func layoutSubviews() {
            super.layoutSubviews()
            previewLayer?.frame = self.bounds
        }
    }

    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.backgroundColor = .black

        let previewLayer = AVCaptureVideoPreviewLayer(session: cameraService.session) // Get session from service
        previewLayer.videoGravity = .resizeAspectFill
        
        view.layer.addSublayer(previewLayer)
        view.previewLayer = previewLayer // Assign layer to view property

        cameraService.setupRotationCoordinator(previewLayer: previewLayer)

        return view
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {
        // Frame updates are handled by PreviewUIView.layoutSubviews()
        // Orientation updates are handled by RotationCoordinator (iOS 17+)
        // or fallback (iOS < 17). No manual updates needed here anymore.

        // If supporting iOS < 17 AND needing manual preview orientation updates:
        // if #unavailable(iOS 17.0) {
        //     context.coordinator.updatePreviewOrientationLegacy(previewLayer: uiView.previewLayer)
        // }
    }

    // Optional: Back Coordinator ONLY for legacy orientation if needed
    /*
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject {
        var parent: CameraPreviewView

        init(_ parent: CameraPreviewView) {
            self.parent = parent
            super.init()
        }

        // Only needed if supporting manual orientation updates for iOS < 17 preview
        func updatePreviewOrientationLegacy(previewLayer: AVCaptureVideoPreviewLayer?) {
             guard #unavailable(iOS 17.0),
                   let connection = previewLayer?.connection,
                   connection.isVideoOrientationSupported else { return }

            // ... (logic similar to old updateOrientation using UIDevice.current.orientation) ...
            // let currentOrientation = UIDevice.current.orientation
            // let videoOrientation: AVCaptureVideoOrientation = ... map orientation ...
            // if connection.videoOrientation != videoOrientation {
            //     connection.videoOrientation = videoOrientation
            // }
        }
    }
    */
}
