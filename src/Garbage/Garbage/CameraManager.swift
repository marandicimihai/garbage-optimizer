import AVFoundation
import Combine
import CoreVideo
import Foundation
import SwiftUI

final class CameraManager: NSObject, ObservableObject {
    @Published var predictionLabel = "Waiting for camera..."
    @Published var predictionConfidence = ""
    @Published var statusMessage = "Starting camera session."
    @Published var actionMessage = "No actuator command yet."
    @Published var canSendRoute = false

    let session = AVCaptureSession()

    private let sessionQueue = DispatchQueue(label: "garbage.camera.session")
    private let videoOutput = AVCaptureVideoDataOutput()
    private let classifier = OfflineClassifier()
    private let esp32BaseURL = URL(string: "http://192.168.4.1")!
    private var isConfigured = false
    private var isProcessingFrame = false
    private var pendingRouteDirection: String?
    private var pendingRouteLabel: String?

    func start() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            configureAndStart()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    if granted {
                        self?.configureAndStart()
                    } else {
                        self?.statusMessage = "Camera access is required to classify in real time."
                    }
                }
            }
        default:
            statusMessage = "Camera access is denied. Enable it in Settings."
        }
    }

    func stop() {
        sessionQueue.async { [session] in
            if session.isRunning {
                session.stopRunning()
            }
        }
    }

    private func configureAndStart() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            if !self.isConfigured {
                self.configureSession()
                self.isConfigured = true
            }
            if !self.session.isRunning {
                self.session.startRunning()
            }
        }
    }

    private func configureSession() {
        session.beginConfiguration()
        session.sessionPreset = .high

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            DispatchQueue.main.async { [weak self] in
                self?.statusMessage = "Unable to open the back camera."
            }
            session.commitConfiguration()
            return
        }
        session.addInput(input)

        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: sessionQueue)
        guard session.canAddOutput(videoOutput) else {
            DispatchQueue.main.async { [weak self] in
                self?.statusMessage = "Unable to attach the camera video output."
            }
            session.commitConfiguration()
            return
        }
        session.addOutput(videoOutput)

        if let connection = videoOutput.connection(with: .video), connection.isVideoOrientationSupported {
            connection.videoOrientation = .portrait
        }

        session.commitConfiguration()
        DispatchQueue.main.async { [weak self] in
            self?.statusMessage = "Camera ready. Classifying frames locally."
        }
    }

    private func classify(pixelBuffer: CVPixelBuffer) {
        guard !isProcessingFrame else { return }
        isProcessingFrame = true

        Task {
            let result = await classifier.predict(pixelBuffer: pixelBuffer)
            await MainActor.run {
                switch result {
                case .success(let output):
                    self.handlePrediction(output)
                    self.predictionConfidence = String(format: "Confidence: %.1f%%", output.confidence * 100)
                    self.statusMessage = "Inference completed locally on device."
                case .failure(let error):
                    self.statusMessage = error.localizedDescription
                }
                self.isProcessingFrame = false
            }
        }
    }

    private func handlePrediction(_ output: ClassifierOutput) {
        predictionLabel = output.label

        guard let direction = routeDirection(for: output.label.lowercased()) else {
            actionMessage = "No actuator command for \(output.label)."
            pendingRouteDirection = nil
            pendingRouteLabel = nil
            canSendRoute = false
            return
        }

        pendingRouteDirection = direction
        pendingRouteLabel = output.label
        canSendRoute = true
        actionMessage = "Ready to route \(output.label) to /\(direction). Tap Route to send."
    }

    func sendPendingRoute() {
        guard let direction = pendingRouteDirection,
              let label = pendingRouteLabel else {
            actionMessage = "No recyclable item selected to route."
            return
        }

        canSendRoute = false
        actionMessage = "Routing \(label) to /\(direction)..."
        sendToESP32(direction: direction, label: label)
    }

    private func routeDirection(for label: String) -> String? {
        switch label {
        case "paper", "cardboard":
            return "1"
        case "plastic":
            return "2"
        case "glass":
            return "3"
        case "metal":
            return "4"
        default:
            return nil
        }
    }

    private func sendToESP32(direction: String, label: String) {
        let url = esp32BaseURL.appendingPathComponent(direction)
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = 4

        URLSession.shared.dataTask(with: request) { [weak self] _, response, error in
            DispatchQueue.main.async {
                if let error {
                    self?.actionMessage = "Failed to route \(label): \(error.localizedDescription)"
                    self?.canSendRoute = true
                    return
                }

                if let httpResponse = response as? HTTPURLResponse, 200 ..< 300 ~= httpResponse.statusCode {
                    self?.actionMessage = "Sent \(label) to /\(direction)."
                    self?.pendingRouteDirection = nil
                    self?.pendingRouteLabel = nil
                } else {
                    self?.actionMessage = "ESP32 returned an unexpected response for \(label)."
                    self?.canSendRoute = true
                }
            }
        }.resume()
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        Task { @MainActor in
            self.classify(pixelBuffer: pixelBuffer)
        }
    }
}

struct LiveCameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.videoPreviewLayer.session = session
        view.videoPreviewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        uiView.videoPreviewLayer.session = session
    }
}

final class PreviewView: UIView {
    override class var layerClass: AnyClass {
        AVCaptureVideoPreviewLayer.self
    }

    var videoPreviewLayer: AVCaptureVideoPreviewLayer {
        layer as! AVCaptureVideoPreviewLayer
    }
}