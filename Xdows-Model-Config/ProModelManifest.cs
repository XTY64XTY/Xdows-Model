using System.Text.Json;

namespace Xdows_Model_Config;

/// <summary>
/// Pro Stacking 模型的结构清单，与融合模型文件放在一起（文件名 = 模型名 + <see cref="FileSuffix"/>）。
/// 训练端导出模型时写出，推理端加载时严格校验：特征版本、分支维度与偏移、哈希算法与种子、
/// 分支文件完整性、清单与模型文件名一致性，任何一项不匹配都抛出明确异常，拒绝静默降级。
/// 旧版四分支模型没有该清单，推理端走兼容路径，不受影响。
/// </summary>
public sealed class ProModelManifest
{
    public const int CurrentSchemaVersion = 1;
    public const string FileSuffix = ".manifest.json";

    public int SchemaVersion { get; set; } = CurrentSchemaVersion;

    /// <summary>产生该模型的训练程序集版本，便于追溯。</summary>
    public string ModelVersion { get; set; } = string.Empty;

    /// <summary>训练时的 <see cref="FeatureSchema.Version"/>。</summary>
    public int FeatureSchemaVersion { get; set; }

    /// <summary>Pro 混合特征总维度（训练时的 <see cref="FeatureSchema.ProHybridFeatureCount"/>）。</summary>
    public int FeatureCount { get; set; }

    /// <summary>特征布局指纹，见 <see cref="ProBranches.ComputeFeatureLayoutHash"/>。</summary>
    public string FeatureHash { get; set; } = string.Empty;

    /// <summary>融合模型文件名（不含目录），防止清单被复制到别的模型旁。</summary>
    public string FusionModelFileName { get; set; } = string.Empty;

    /// <summary>融合模型输入维度，必须等于分支数量。</summary>
    public int FusionInputCount { get; set; }

    public List<ProBranchManifestEntry> Branches { get; set; } = new();

    /// <summary>ImportBehavior 特征组使用的哈希算法标识。</summary>
    public string HashAlgorithm { get; set; } = string.Empty;

    public uint HashSeed { get; set; }
    public int DllHashDimensions { get; set; }
    public int ApiHashDimensions { get; set; }

    /// <summary>训练配置中的固定判毒阈值（百分比）。</summary>
    public double Threshold { get; set; }

    /// <summary>训练阶段按代价敏感目标校准出的推荐阈值（百分比）。</summary>
    public double RecommendedThreshold { get; set; }

    public string GeneratedAt { get; set; } = string.Empty;

    /// <summary>清单文件名（不含目录），与融合模型同名。</summary>
    public static string FileNameFor(string modelPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
        return Path.GetFileNameWithoutExtension(modelPath) + FileSuffix;
    }

    public static string ResolvePath(string modelPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
        string directory = Path.GetDirectoryName(modelPath) ?? string.Empty;
        return Path.Combine(directory, FileNameFor(modelPath));
    }

    public void Save(string modelPath)
    {
        string path = ResolvePath(modelPath);
        File.WriteAllText(path, JsonSerializer.Serialize(this, ModelManifestJsonContext.Default.ProModelManifest));
    }

    /// <summary>清单是否存在（不解析）。</summary>
    public static bool Exists(string modelPath) => File.Exists(ResolvePath(modelPath));

    /// <summary>
    /// 读取清单并反序列化。文件缺失、JSON 损坏或内容为空时抛出带原因的明确异常。
    /// 调用方若需要兼容无清单的旧模型，应先通过 <see cref="Exists"/> 判断。
    /// </summary>
    public static ProModelManifest Load(string modelPath)
    {
        string path = ResolvePath(modelPath);
        if (!File.Exists(path))
            throw new FileNotFoundException($"缺少 Pro 模型清单：{Path.GetFileName(path)}", path);

        ProModelManifest? manifest;
        try
        {
            manifest = JsonSerializer.Deserialize(
                File.ReadAllText(path),
                ModelManifestJsonContext.Default.ProModelManifest);
        }
        catch (JsonException ex)
        {
            throw new InvalidOperationException($"Pro 模型清单无法解析：{ex.Message}", ex);
        }

        return manifest ?? throw new InvalidOperationException("Pro 模型清单内容为空。");
    }

    /// <summary>
    /// 对照当前代码的特征 Schema 与哈希配置校验清单，并确认各分支模型文件齐备。
    /// 校验失败抛出 <see cref="InvalidOperationException"/>，消息说明具体不符项。
    /// </summary>
    public void Validate(string fusionModelPath)
    {
        if (SchemaVersion != CurrentSchemaVersion)
            throw new InvalidOperationException($"Pro 模型清单版本为 {SchemaVersion}，期望 {CurrentSchemaVersion}。");

        if (FeatureSchemaVersion != FeatureSchema.Version)
            throw new InvalidOperationException(
                $"Pro 模型特征版本为 {FeatureSchemaVersion}，当前代码为 {FeatureSchema.Version}。请重新训练或换用匹配的运行库。");

        string expectedLayoutHash = ProBranches.ComputeFeatureLayoutHash();
        if (!string.Equals(FeatureHash, expectedLayoutHash, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException($"Pro 模型特征布局指纹不匹配：清单为 {FeatureHash}，当前为 {expectedLayoutHash}。");

        if (!string.Equals(HashAlgorithm, ImportHashConfig.Algorithm, StringComparison.Ordinal))
            throw new InvalidOperationException($"Pro 模型哈希算法为 {HashAlgorithm}，当前为 {ImportHashConfig.Algorithm}。");
        if (HashSeed != ImportHashConfig.Seed)
            throw new InvalidOperationException($"Pro 模型哈希种子为 {HashSeed}，当前为 {ImportHashConfig.Seed}。");
        if (DllHashDimensions != ImportHashConfig.DllHashDimensions)
            throw new InvalidOperationException($"Pro 模型 DLL 哈希维度为 {DllHashDimensions}，当前为 {ImportHashConfig.DllHashDimensions}。");
        if (ApiHashDimensions != ImportHashConfig.ApiHashDimensions)
            throw new InvalidOperationException($"Pro 模型 API 哈希维度为 {ApiHashDimensions}，当前为 {ImportHashConfig.ApiHashDimensions}。");

        string fusionFileName = Path.GetFileName(fusionModelPath);
        if (!string.Equals(FusionModelFileName, fusionFileName, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException($"Pro 清单对应的融合模型为 {FusionModelFileName}，当前加载的是 {fusionFileName}。");

        if (FusionInputCount != Branches.Count)
            throw new InvalidOperationException($"Pro 清单融合输入维度为 {FusionInputCount}，但分支数量为 {Branches.Count}。");

        if (FeatureCount != FeatureSchema.ProHybridFeatureCount)
            throw new InvalidOperationException($"Pro 清单混合特征维度为 {FeatureCount}，当前为 {FeatureSchema.ProHybridFeatureCount}。");

        string directory = Path.GetDirectoryName(fusionModelPath) ?? string.Empty;
        var seen = new HashSet<ProBranch>();
        foreach (ProBranchManifestEntry branch in Branches)
        {
            if (!Enum.TryParse(branch.Name, ignoreCase: false, out ProBranch kind))
                throw new InvalidOperationException($"Pro 清单包含未知分支：{branch.Name}。");
            if (!seen.Add(kind))
                throw new InvalidOperationException($"Pro 清单包含重复分支：{branch.Name}。");

            int expectedCount = ProBranches.FeatureCount(kind);
            int expectedOffset = ProBranches.Offset(kind);
            if (branch.FeatureCount != expectedCount || branch.InputDimension != expectedCount)
                throw new InvalidOperationException(
                    $"Pro {branch.Name} 分支维度不匹配：清单特征数 {branch.FeatureCount}、输入维度 {branch.InputDimension}，期望 {expectedCount}。");
            if (branch.FeatureOffset != expectedOffset)
                throw new InvalidOperationException($"Pro {branch.Name} 分支偏移为 {branch.FeatureOffset}，期望 {expectedOffset}。");

            string branchPath = Path.Combine(directory, branch.FileName);
            if (!File.Exists(branchPath))
                throw new FileNotFoundException($"缺少 Pro 分支模型：{branch.FileName}", branchPath);
        }

        if (!double.IsFinite(Threshold) || Threshold < 0 || Threshold > 100)
            throw new InvalidOperationException($"Pro 清单固定阈值 {Threshold} 不在 0-100 范围内。");
        if (!double.IsFinite(RecommendedThreshold) || RecommendedThreshold < 0 || RecommendedThreshold > 100)
            throw new InvalidOperationException($"Pro 清单推荐阈值 {RecommendedThreshold} 不在 0-100 范围内。");
    }
}

public sealed class ProBranchManifestEntry
{
    /// <summary>分支名，对应 <see cref="ProBranch"/> 枚举成员名。</summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>分支模型文件名（不含目录），与融合模型同目录。</summary>
    public string FileName { get; set; } = string.Empty;

    /// <summary>分支 ONNX 模型的输入维度。</summary>
    public int InputDimension { get; set; }

    /// <summary>该分支特征在 Pro 混合特征向量中的偏移。</summary>
    public int FeatureOffset { get; set; }

    /// <summary>该分支的特征维度。</summary>
    public int FeatureCount { get; set; }
}
