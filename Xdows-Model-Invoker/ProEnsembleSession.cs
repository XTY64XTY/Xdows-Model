using Microsoft.ML.OnnxRuntime;
using Xdows_Model_Config;

namespace Xdows_Model_Invoker;

internal sealed class ProEnsembleSession : IDisposable
{
    private readonly InferenceSession[] _branches;

    public ProEnsembleSession(string fusionModelPath)
    {
        string[] suffixes = ["-Standard", "-Flash", "-RawStat", "-Structural"];
        int[] dimensions =
        [
            FeatureSchema.StandardFeatureCount,
            FeatureSchema.FlashFeatureCount,
            FeatureSchema.ProRawStatCount,
            FeatureSchema.ProStructuralCount
        ];

        _branches = new InferenceSession[suffixes.Length];
        try
        {
            for (int i = 0; i < suffixes.Length; i++)
            {
                string path = AddSuffix(fusionModelPath, suffixes[i]);
                if (!File.Exists(path))
                    throw new FileNotFoundException($"缺少 Pro Stacking 分支模型：{Path.GetFileName(path)}", path);
                _branches[i] = new InferenceSession(path);
                ValidateDimension(_branches[i], dimensions[i], Path.GetFileName(path));
            }
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    public float Predict(InferenceSession fusionSession, float[] hybridFeatures)
    {
        if (hybridFeatures.Length != FeatureSchema.ProHybridFeatureCount)
            throw new ArgumentException($"Pro 混合特征必须为 {FeatureSchema.ProHybridFeatureCount} 维。", nameof(hybridFeatures));

        var fusionFeatures = new float[FeatureSchema.ProFusionFeatureCount];
        fusionFeatures[0] = PredictBranch(0, hybridFeatures, FeatureSchema.ProStandardOffset, FeatureSchema.StandardFeatureCount);
        fusionFeatures[1] = PredictBranch(1, hybridFeatures, FeatureSchema.ProFlashOffset, FeatureSchema.FlashFeatureCount);
        fusionFeatures[2] = PredictBranch(2, hybridFeatures, FeatureSchema.ProRawStatOffset, FeatureSchema.ProRawStatCount);
        fusionFeatures[3] = PredictBranch(3, hybridFeatures, FeatureSchema.ProStructuralOffset, FeatureSchema.ProStructuralCount);
        return ModelInvoker.RunProbability(fusionSession, fusionFeatures, FeatureSchema.ProFusionFeatureCount);
    }

    private float PredictBranch(int index, float[] source, int offset, int count)
    {
        var features = new float[count];
        Array.Copy(source, offset, features, 0, count);
        return ModelInvoker.RunProbability(_branches[index], features, count) / 100f;
    }

    public void Dispose()
    {
        foreach (var branch in _branches)
            branch?.Dispose();
    }

    internal static string AddSuffix(string path, string suffix)
    {
        string directory = Path.GetDirectoryName(path) ?? string.Empty;
        return Path.Combine(directory, Path.GetFileNameWithoutExtension(path) + suffix + Path.GetExtension(path));
    }

    internal static int ReadFeatureDimension(InferenceSession session)
    {
        if (!session.InputMetadata.TryGetValue("Features", out var nodeMeta))
            return -1;
        var dimensions = nodeMeta.Dimensions;
        if (dimensions.Length == 2 && dimensions[1] > 0)
            return dimensions[1];
        if (dimensions.Length == 1 && dimensions[0] > 0)
            return dimensions[0];
        return -1;
    }

    private static void ValidateDimension(InferenceSession session, int expected, string modelName)
    {
        int actual = ReadFeatureDimension(session);
        if (actual > 0 && actual != expected)
            throw new InvalidOperationException($"{modelName} 特征维度不匹配：{actual}，期望 {expected}。");
    }
}
