using Microsoft.ML.OnnxRuntime;
using Xdows_Model_Config;

namespace Xdows_Model_Invoker;

public readonly record struct AdaptiveScanResult(bool IsVirus, float Probability, ModelMode FinalMode);

public sealed class AdaptiveModelSession : IDisposable
{
    private readonly InferenceSession _flashSession;
    private readonly InferenceSession _standardSession;
    private readonly InferenceSession _proSession;
    private readonly ProEnsembleSession? _proEnsemble;
    private readonly TrainingConfig _config;

    internal AdaptiveModelSession(string flashPath, string standardPath, string proPath, TrainingConfig config)
    {
        _config = config;
        _flashSession = new InferenceSession(flashPath);
        _standardSession = new InferenceSession(standardPath);
        _proSession = new InferenceSession(proPath);

        ValidateDimension(_flashSession, FeatureSchema.FlashFeatureCount, "Flash");
        ValidateDimension(_standardSession, FeatureSchema.StandardFeatureCount, "Standard");
        int proDimension = ProEnsembleSession.ReadFeatureDimension(_proSession);
        if (proDimension == FeatureSchema.ProFusionFeatureCount)
            _proEnsemble = new ProEnsembleSession(proPath);
        else if (proDimension != FeatureSchema.ProHybridFeatureCount)
            throw new InvalidOperationException($"Pro 模型维度为 {proDimension}，期望 {FeatureSchema.ProFusionFeatureCount} 或 {FeatureSchema.ProHybridFeatureCount}。");
    }

    public AdaptiveScanResult ScanFile(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException("找不到指定文件", filePath);
        byte[] bytes = File.ReadAllBytes(filePath);
        if (!FeatureExtractor.IsPeFile(bytes))
            throw new NotSupportedException("不支持该文件类型");

        float[] flashFeatures = FlashFeatureExtractor.ExtractFromBytes(bytes).ToFloatArray();
        float flashProbability = ModelInvoker.RunProbability(
            _flashSession,
            flashFeatures,
            FeatureSchema.FlashFeatureCount);
        var flashDecision = AdaptiveDecisionPolicy.EvaluateIntermediate(flashProbability, _config.FlashThreshold);
        if (flashDecision != AdaptiveIntermediateDecision.Escalate)
            return CreateSafeResult(flashProbability, ModelMode.Flash);

        float[] standardFeatures = FeatureExtractor.ExtractFromBytes(bytes).ToFloatArray();
        float standardProbability = ModelInvoker.RunProbability(
            _standardSession,
            standardFeatures,
            FeatureSchema.StandardFeatureCount);
        var standardDecision = AdaptiveDecisionPolicy.EvaluateIntermediate(standardProbability, _config.StandardThreshold);
        if (standardDecision != AdaptiveIntermediateDecision.Escalate)
            return CreateSafeResult(standardProbability, ModelMode.Standard);

        float[] proFeatures = AdaptiveFeatureComposer.ComposePro(bytes, standardFeatures, flashFeatures);
        float proProbability = _proEnsemble != null
            ? _proEnsemble.Predict(_proSession, proFeatures)
            : ModelInvoker.RunProbability(_proSession, proFeatures, FeatureSchema.ProHybridFeatureCount);
        return CreateResult(proProbability, _config.ProThreshold, ModelMode.Pro);
    }

    private static AdaptiveScanResult CreateResult(float probability, double threshold, ModelMode mode)
    {
        return new AdaptiveScanResult(probability >= threshold, probability, mode);
    }

    private static AdaptiveScanResult CreateSafeResult(float probability, ModelMode mode)
    {
        return new AdaptiveScanResult(false, probability, mode);
    }

    private static void ValidateDimension(InferenceSession session, int expected, string name)
    {
        int actual = ProEnsembleSession.ReadFeatureDimension(session);
        if (actual > 0 && actual != expected)
            throw new InvalidOperationException($"{name} 模型维度为 {actual}，期望 {expected}。");
    }

    public void Dispose()
    {
        _proEnsemble?.Dispose();
        _proSession.Dispose();
        _standardSession.Dispose();
        _flashSession.Dispose();
    }
}
