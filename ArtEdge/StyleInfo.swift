import SwiftUI

struct StyleInfo: Identifiable, Hashable {
    let id = UUID()
    let name: String  // User-facing name
    let assetName: String  // Name of the image in Assets.xcassets (must match an imageset)
    let associatedModelFilename: String?  // Filename (without extension) for AdaIN/FST models

    // Initializer for styles that are just inputs (for arbitrary models)
    init(name: String, assetName: String) {
        self.name = name
        self.assetName = assetName
        self.associatedModelFilename = nil
        // Verify asset exists
        if UIImage(named: assetName) == nil {
            print("⚠️ Warning: StyleInfo asset '\(assetName)' not found in Assets.")
        }
    }

    // Initializer for styles that load a specific model (AdaIN/FST)
    init(name: String, assetName: String, modelFilename: String) {
        self.name = name
        self.assetName = assetName
        self.associatedModelFilename = modelFilename
        // Verify asset exists
        if UIImage(named: assetName) == nil {
            print("⚠️ Warning: StyleInfo asset '\(assetName)' not found in Assets.")
        }
    }
}

// Styles Associated with Specific Single-Style Models

// Styles that load specific AdaIN models
let availableAdaINStyles: [StyleInfo] = [
    StyleInfo(name: "Brushstrokes", assetName: "brushstrokes", modelFilename: "AdaIN-brushstrokes"),
    StyleInfo(name: "Mondrian", assetName: "mondrian", modelFilename: "AdaIN-mondrian"),
    StyleInfo(name: "Starry Night", assetName: "starry_night", modelFilename: "AdaIN-starry"),
]

// Styles that load specific FST models
let availableFSTStyles: [StyleInfo] = [
    StyleInfo(name: "Mosaic", assetName: "mosaic", modelFilename: "FST-Mosaic"),
    StyleInfo(
        name: "Rain Princess", assetName: "rain_princess", modelFilename: "FST-Rain_Princess"),
    StyleInfo(name: "Starry Night", assetName: "starry_night", modelFilename: "FST-Starry"),
]

// Styles Used as INPUT for Arbitrary Models
// These styles can be used as input for AesFA and StyTr2
// They use the assetName but don't load a model themselves.
let availableArbitraryStyleInputs: [StyleInfo] = [
    StyleInfo(name: "Brushstrokes", assetName: "brushstrokes"),
    StyleInfo(name: "Composition VII", assetName: "composition_vii"),
    StyleInfo(name: "Mosaic", assetName: "mosaic"),
    StyleInfo(name: "Rain Princess", assetName: "rain_princess"),
    StyleInfo(name: "Starry Night", assetName: "starry_night"),
    StyleInfo(name: "The Scream", assetName: "the_scream"),
    StyleInfo(name: "Mondrian", assetName: "mondrian"),
]

// Helper to get all unique asset names needed (for verification)
func allRequiredAssetNames() -> Set<String> {
    var names = Set<String>()
    availableAdaINStyles.forEach { names.insert($0.assetName) }
    availableFSTStyles.forEach { names.insert($0.assetName) }
    availableArbitraryStyleInputs.forEach { names.insert($0.assetName) }
    return names
}
