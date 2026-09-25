namespace Xdows_Model_Config;

/// <summary>
/// Pro Stacking 的分支标识。训练端（Maker）与推理端（Invoker）共享同一份定义，
/// 分支顺序即融合特征向量的顺序，不允许调整已有成员的数值。
/// </summary>
public enum ProBranch
{
    Standard = 0,
    Flash = 1,
    RawStat = 2,
    Structural = 3,
    ImportBehavior = 4
}

/// <summary>
/// Pro 分支布局的单一权威来源：每个分支在混合特征向量中的偏移、维度与模型文件后缀。
/// 训练端的样本切分、ONNX 导出与推理端的分支加载都从这里取布局，避免偏移规则散落多处。
/// </summary>
public static class ProBranches
{
    /// <summary>当前 Schema 的全部分支，顺序与融合特征向量一致。</summary>
    public static IReadOnlyList<ProBranch> All { get; } = (ProBranch[])Enum.GetValues(typeof(ProBranch));

    /// <summary>旧版（Schema Version 2）四分支集合，用于无 Manifest 旧模型的兼容加载。</summary>
    public static IReadOnlyList<ProBranch> Legacy { get; } =
        new[] { ProBranch.Standard, ProBranch.Flash, ProBranch.RawStat, ProBranch.Structural };

    public static int FeatureCount(ProBranch branch) => branch switch
    {
        ProBranch.Standard => FeatureSchema.StandardFeatureCount,
        ProBranch.Flash => FeatureSchema.FlashFeatureCount,
        ProBranch.RawStat => FeatureSchema.ProRawStatCount,
        ProBranch.Structural => FeatureSchema.ProStructuralCount,
        ProBranch.ImportBehavior => FeatureSchema.ProImportBehaviorCount,
        _ => throw new ArgumentOutOfRangeException(nameof(branch))
    };

    public static int Offset(ProBranch branch) => branch switch
    {
        ProBranch.Standard => FeatureSchema.ProStandardOffset,
        ProBranch.Flash => FeatureSchema.ProFlashOffset,
        ProBranch.RawStat => FeatureSchema.ProRawStatOffset,
        ProBranch.Structural => FeatureSchema.ProStructuralOffset,
        ProBranch.ImportBehavior => FeatureSchema.ProImportBehaviorOffset,
        _ => throw new ArgumentOutOfRangeException(nameof(branch))
    };

    public static string ModelFileSuffix(ProBranch branch) => branch switch
    {
        ProBranch.Standard => "-Standard",
        ProBranch.Flash => "-Flash",
        ProBranch.RawStat => "-RawStat",
        ProBranch.Structural => "-Structural",
        ProBranch.ImportBehavior => "-ImportBehavior",
        _ => throw new ArgumentOutOfRangeException(nameof(branch))
    };

    /// <summary>分支模型的文件名（不含目录），与融合模型同目录、同扩展名。</summary>
    public static string FileNameFor(ProBranch branch, string fusionModelPath)
    {
        return Path.GetFileNameWithoutExtension(fusionModelPath) + ModelFileSuffix(branch) + Path.GetExtension(fusionModelPath);
    }

    /// <summary>分支模型的完整路径，目录与融合模型一致。</summary>
    public static string PathFor(ProBranch branch, string fusionModelPath)
    {
        string directory = Path.GetDirectoryName(fusionModelPath) ?? string.Empty;
        return Path.Combine(directory, FileNameFor(branch, fusionModelPath));
    }

    /// <summary>从混合特征向量中取出指定分支的特征段（复制）。</summary>
    public static float[] Extract(float[] features, ProBranch branch)
    {
        var result = new float[FeatureCount(branch)];
        Copy(features, branch, result);
        return result;
    }

    /// <summary>把混合特征向量中指定分支的段复制到目标缓冲区，长度不符即抛出。</summary>
    public static void Copy(float[] features, ProBranch branch, Span<float> destination)
    {
        int count = FeatureCount(branch);
        if (destination.Length != count)
            throw new ArgumentException($"Pro {branch} 分支目标缓冲区长度必须为 {count}。", nameof(destination));
        features.AsSpan(Offset(branch), count).CopyTo(destination);
    }

    /// <summary>
    /// 特征布局的稳定指纹：由分支名、偏移、维度与 ImportBehavior 哈希配置串接后取 FNV-1a 哈希。
    /// 写入模型 Manifest，加载时重算比对，任何特征顺序/维度/哈希规则变化都会被拒绝。
    /// </summary>
    public static string ComputeFeatureLayoutHash()
    {
        var builder = new System.Text.StringBuilder();
        builder.Append("schema=").Append(FeatureSchema.Version);
        foreach (ProBranch branch in All)
        {
            builder.Append('|').Append(branch).Append(':').Append(Offset(branch)).Append(':').Append(FeatureCount(branch));
        }
        builder.Append("|hash=").Append(ImportHashConfig.Algorithm)
            .Append(':').Append(ImportHashConfig.Seed)
            .Append(':').Append(ImportHashConfig.DllHashDimensions)
            .Append(':').Append(ImportHashConfig.ApiHashDimensions);

        uint hash = ImportHashConfig.Seed;
        foreach (char c in builder.ToString())
        {
            hash ^= c;
            hash *= 16777619u;
        }
        return hash.ToString("x8");
    }
}
