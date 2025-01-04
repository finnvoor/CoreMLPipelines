import CoreML
@preconcurrency import Hub
import Tokenizers

class CausalLMModel {
    // MARK: Lifecycle

    init(model: MLModel) {
        assert(model.modelDescription.inputDescriptionsByName["input_ids"] != nil)
        assert(model.modelDescription.inputDescriptionsByName["causal_mask"] != nil)
        assert(model.modelDescription.stateDescriptionsByName["key_cache"] != nil)
        assert(model.modelDescription.stateDescriptionsByName["value_cache"] != nil)
        assert(model.modelDescription.outputDescriptionsByName["logits"] != nil)
        self.model = model
        self.contextSize = model.modelDescription.stateDescriptionsByName["key_cache"]!
            .stateConstraint!.bufferShape[3]
    }

    // MARK: Internal

    typealias KVCache = MLState
    
    let contextSize: Int

    class func load(
        contentsOf url: URL,
        configuration: MLModelConfiguration = MLModelConfiguration()
    ) async throws -> CausalLMModel {
        try await CausalLMModel(
            model: MLModel.load(contentsOf: url, configuration: configuration)
        )
    }

    class func from(pretrained model: String, hubAPI: HubApi = .shared) async throws -> CausalLMModel {
        async let _fileNames = try hubAPI.getFilenames(from: model, matching: ["*.mlmodelc/*"])
        async let _modelURL = try hubAPI.snapshot(from: model, matching: ["*.mlmodelc/*"])
        let (fileNames, modelURL) = try await (_fileNames, _modelURL)
        return try await CausalLMModel.load(contentsOf: modelURL.appending(
            path: fileNames[0].split(separator: "/")[0]
        ))
    }

    func forward(
        inputIDs: MLShapedArray<Int32>,
        causalMask: MLShapedArray<Float16>,
        kvCache: KVCache,
        options: MLPredictionOptions = .init()
    ) async throws -> MLShapedArray<Float16> {
        try await MLShapedArray<Float16>(model.prediction(
            from: MLDictionaryFeatureProvider(dictionary: [
                "input_ids": MLMultiArray(inputIDs),
                "causal_mask": MLMultiArray(causalMask)
            ]),
            using: kvCache,
            options: options
        ).featureValue(for: "logits")!.multiArrayValue!)
    }

    func callAsFunction(
        inputIDs: MLShapedArray<Int32>,
        causalMask: MLShapedArray<Float16>,
        kvCache: KVCache,
        options: MLPredictionOptions = .init()
    ) async throws -> MLShapedArray<Float16> {
        try await forward(
            inputIDs: inputIDs,
            causalMask: causalMask,
            kvCache: kvCache,
            options: options
        )
    }

    func makeKVCache() -> KVCache {
        model.makeState()
    }

    // MARK: Private

    private let model: MLModel
}
