import SwiftUI

struct StyleInfo: Identifiable, Hashable {
    let id = UUID()
    let name: String  // User-facing name
    let assetName: String  // Name of the image in Assets.xcassets
}

let availableStyles: [StyleInfo] = [
    StyleInfo(name: "Starry Night", assetName: "starry_night"),
    StyleInfo(name: "The Scream", assetName: "the_scream"),
    StyleInfo(name: "Composition VII", assetName: "composition_vii"),
    StyleInfo(name: "Brushstrokes", assetName: "brushstrokes"),
]
