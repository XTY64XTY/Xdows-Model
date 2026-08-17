using System.Buffers;
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
                _branches[i] = new InferenceSession(path, ModelInvoker.CreateSessionOptions());
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

        var fusionFeatures = ArrayPool<float>.Shared.Rent(FeatureSchema.ProFusionFeatureCount);
        try
        {
            Parallel.For(0, 4, i => PredictBranch(i, hybridFeatures, fusionFeatures));
            return ModelInvoker.RunProbability(fusionSession, fusionFeatures, FeatureSchema.ProFusionFeatureCount);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(fusionFeatures);
        }
    }

    private void PredictBranch(int index, float[] source, float[] fusionFeatures)
    {
        (int offset, int count) = index switch
        {
            0 => (FeatureSchema.ProStandardOffset, FeatureSchema.StandardFeatureCount),
            1 => (FeatureSchema.ProFlashOffset, FeatureSchema.FlashFeatureCount),
            2 => (FeatureSchema.ProRawStatOffset, FeatureSchema.ProRawStatCount),
            3 => (FeatureSchema.ProStructuralOffset, FeatureSchema.ProStructuralCount),
            _ => throw new ArgumentOutOfRangeException(nameof(index))
        };

        var features = ArrayPool<float>.Shared.Rent(count);
        try
        {
            Array.Copy(source, offset, features, 0, count);
            fusionFeatures[index] = ModelInvoker.RunProbability(_branches[index], features, count) / 100f;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(features);
        }
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
