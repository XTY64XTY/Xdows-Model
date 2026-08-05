using Xdows_Model_Config;

namespace Xdows_Model_Invoker;

public static class AdaptiveFeatureComposer
{
    public static float[] ComposePro(byte[] bytes, float[] standardFeatures, float[] flashFeatures)
    {
        if (standardFeatures.Length != FeatureSchema.StandardFeatureCount)
            throw new ArgumentException("Standard feature count mismatch.", nameof(standardFeatures));
        if (flashFeatures.Length != FeatureSchema.FlashFeatureCount)
            throw new ArgumentException("Flash feature count mismatch.", nameof(flashFeatures));

        var result = new float[FeatureSchema.ProHybridFeatureCount];
        standardFeatures.CopyTo(result, FeatureSchema.ProStandardOffset);
        flashFeatures.CopyTo(result, FeatureSchema.ProFlashOffset);
        ProRawStatExtractor.ExtractFromBytes(bytes).ToFloatArray().CopyTo(result, FeatureSchema.ProRawStatOffset);
        ProHybridFeatureExtractor.ExtractStructuralFeatures(bytes).CopyTo(result, FeatureSchema.ProStructuralOffset);
        return result;
    }
}
