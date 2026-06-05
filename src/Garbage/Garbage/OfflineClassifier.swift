import CoreML
import Vision
import UIKit

struct ClassifierOutput {
    let label: String
    let confidence: Double
}

enum ClassifierError: LocalizedError {
    case modelNotFound
    case predictionFailed
    case invalidImage

    var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "ResNet.mlmodelc is missing from the app bundle. Add the converted model to the target and rebuild."
        case .predictionFailed:
            return "The CoreML model could not classify the image."
        case .invalidImage:
            return "The selected image could not be decoded."
        }
    }
}

final class OfflineClassifier {
    private let labels = [
        "glass",
        "metal",
        "paper",
        "plastic",
    ]

    private lazy var visionModel: VNCoreMLModel? = {
        guard let url = Bundle.main.url(forResource: "ResNet", withExtension: "mlmodelc") else {
            return nil
        }
        do {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .all
            let model = try MLModel(contentsOf: url, configuration: configuration)
            return try VNCoreMLModel(for: model)
        } catch {
            return nil
        }
    }()

    func predict(pixelBuffer: CVPixelBuffer) async -> Result<ClassifierOutput, Error> {
        guard let visionModel else {
            return .failure(ClassifierError.modelNotFound)
        }

        return await withCheckedContinuation { continuation in
            do {
                let request = VNCoreMLRequest(model: visionModel) { [labels] request, error in
                    if let error {
                        continuation.resume(returning: .failure(error))
                        return
                    }
                    guard let results = request.results as? [VNCoreMLFeatureValueObservation],
                          let topResult = results.first,
                          let scores = topResult.featureValue.multiArrayValue else {
                        continuation.resume(returning: .failure(ClassifierError.predictionFailed))
                        return
                    }

                    let (index, confidence) = self.topClass(in: scores)
                    guard index >= 0 && index < labels.count else {
                        continuation.resume(returning: .failure(ClassifierError.predictionFailed))
                        return
                    }
                    let label = labels[index]
                    continuation.resume(returning: .success(ClassifierOutput(label: label, confidence: confidence)))
                }
                request.imageCropAndScaleOption = .centerCrop

                let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
                try handler.perform([request])
            } catch {
                continuation.resume(returning: .failure(error))
            }
        }
    }

    private func topClass(in scores: MLMultiArray) -> (Int, Double) {
        let count = scores.count
        guard count > 0 else { return (0, 0) }

        var bestIndex = 0
        var bestValue = Double(truncating: scores[0])
        if count > 1 {
            for index in 1..<count {
                let value = Double(truncating: scores[index])
                if value > bestValue {
                    bestValue = value
                    bestIndex = index
                }
            }
        }

        let exponentials = (0..<count).map { exp(Double(truncating: scores[$0]) - bestValue) }
        let total = exponentials.reduce(0, +)
        let confidence = total > 0 ? exponentials[bestIndex] / total : 0
        return (bestIndex, confidence)
    }
}