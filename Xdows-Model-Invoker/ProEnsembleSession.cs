using Microsoft.ML.OnnxRuntime;
using System.Buffers;
using Xdows_Model_Config;

namespace Xdows_Model_Invoker;

/// <summary>
/// Pro Stacking 分支会话：按模型 Manifest 加载分支（新版五分支），
/// 对没有 Manifest 的旧版四分支模型走兼容路径。加载期完成全部维度与布局校验，
/// 校验失败抛出明确异常，不做静默降级。
/// </summary>
internal sealed class ProEnsembleSession : IDisposable
{
    private readonly InferenceSession[] _branches;
    private readonly ProBranch[] _branchKinds;

    /// <summary>
    /// 创建分支会话。<paramref name="fusionInputCount"/> 是融合模型的实际输入维度，
    /// 由调用方从融合模型元数据读取后传入，用于判定新旧模型代数。
    /// </summary>
    public ProEnsembleSession(string fusionModelPath, int fusionInputCount)
    {
        string[] branchPaths = ResolveBranchPaths(fusionModelPath, fusionInputCount, out _branchKinds);

        _branches = new InferenceSession[branchPaths.Length];
        try
        {
            for (int i = 0; i < branchPaths.Length; i++)
            {
                if (!File.Exists(branchPaths[i]))
                    throw new FileNotFoundException($"缺少 Pro Stacking 分支模型：{Path.GetFileName(branchPaths[i])}", branchPaths[i]);
                _branches[i] = new InferenceSession(branchPaths[i], ModelInvoker.CreateSessionOptions());
                ValidateDimension(_branches[i], ProBranches.FeatureCount(_branchKinds[i]), Path.GetFileName(branchPaths[i]));
            }
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>融合模型的输入维度（= 分支数）。</summary>
    public int FusionInputCount => _branchKinds.Length;

    /// <summary>
    /// 解析分支模型文件列表。清单存在时严格校验并按清单加载；
    /// 清单缺失时仅接受旧版四维融合输入（四分支），新版模型必须随附清单。
    /// 公开给架构测试是刻意的：这是"哪些分支、按什么顺序、从哪加载"的唯一判别点。
    /// </summary>
    internal static string[] ResolveBranchPaths(string fusionModelPath, int fusionInputCount, out ProBranch[] branchKinds)
    {
        if (ProModelManifest.Exists(fusionModelPath))
        {
            ProModelManifest manifest = ProModelManifest.Load(fusionModelPath);
            manifest.Validate(fusionModelPath);
            if (fusionInputCount > 0 && fusionInputCount != manifest.FusionInputCount)
            {
                throw new InvalidOperationException(
                    $"Pro 融合模型输入维度为 {fusionInputCount}，清单声明 {manifest.FusionInputCount}。");
            }

            string manifestDirectory = Path.GetDirectoryName(fusionModelPath) ?? string.Empty;
            branchKinds = manifest.Branches
                .Select(branch => Enum.Parse<ProBranch>(branch.Name, ignoreCase: false))
                .ToArray();
            return manifest.Branches
                .Select(branch => Path.Combine(manifestDirectory, branch.FileName))
                .ToArray();
        }

        if (fusionInputCount == FeatureSchema.ProFusionFeatureCount)
        {
            throw new InvalidOperationException(
                $"Pro 融合模型输入维度为 {fusionInputCount}（五分支架构），但缺少模型清单" +
                $"（{Path.GetFileName(ProModelManifest.ResolvePath(fusionModelPath))}）。请使用训练端同步生成的清单文件。");
        }

        if (fusionInputCount != FeatureSchema.ProLegacyFusionFeatureCount)
        {
            throw new InvalidOperationException(
                $"Pro 融合模型输入维度为 {fusionInputCount}，期望 {FeatureSchema.ProLegacyFusionFeatureCount}（旧版四分支）或 {FeatureSchema.ProFusionFeatureCount}（新版五分支）。");
        }

        branchKinds = ProBranches.Legacy.ToArray();
        return branchKinds
            .Select(branch => ProBranches.PathFor(branch, fusionModelPath))
            .ToArray();
    }

    public float Predict(InferenceSession fusionSession, float[] hybridFeatures)
    {
        int requiredSpan = 0;
        foreach (ProBranch branch in _branchKinds)
            requiredSpan = Math.Max(requiredSpan, ProBranches.Offset(branch) + ProBranches.FeatureCount(branch));

        if (hybridFeatures.Length < requiredSpan)
            throw new ArgumentException($"Pro 混合特征至少需要 {requiredSpan} 维，实际 {hybridFeatures.Length} 维。", nameof(hybridFeatures));

        var fusionFeatures = ArrayPool<float>.Shared.Rent(_branchKinds.Length);
        try
        {
            Parallel.For(0, _branchKinds.Length, i => PredictBranch(i, hybridFeatures, fusionFeatures));
            return ModelInvoker.RunProbability(fusionSession, fusionFeatures, _branchKinds.Length);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(fusionFeatures);
        }
    }

    private void PredictBranch(int index, float[] source, float[] fusionFeatures)
    {
        ProBranch branch = _branchKinds[index];
        int offset = ProBranches.Offset(branch);
        int count = ProBranches.FeatureCount(branch);

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
