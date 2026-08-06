using System.Text.Json;
using System.Text.Json.Serialization;

namespace Xdows_Model_Config;

/// <summary>
/// 与模型文件放在一起的阈值清单，让训练阶段校准出的工作点可以被调用端自动采用，
/// 而不需要手工同步 <see cref="TrainingConfig"/> 里的固定阈值。
/// 文件名为模型文件名加 <see cref="FileSuffix"/>，例如 Xdows-Model-Pro.onnx 对应
/// Xdows-Model-Pro.threshold.json。
/// </summary>
public sealed class ModelThresholdManifest
{
    public const int CurrentSchemaVersion = 1;
    public const string FileSuffix = ".threshold.json";



    public int SchemaVersion { get; set; } = CurrentSchemaVersion;

    /// <summary>
    /// 该阈值所属的模型模式名称，用于防止清单被放到错误的模型旁边。
    /// </summary>
    public string ModelMode { get; set; } = string.Empty;

    /// <summary>
    /// 推荐的判毒阈值，百分比。
    /// </summary>
    public double RecommendedThreshold { get; set; }

    /// <summary>
    /// 阈值的选择方式，便于排查线上工作点的来源。
    /// </summary>
    public string SelectionMethod { get; set; } = string.Empty;

    public double FalsePositiveCostRatio { get; set; }

    public long FalseNegative { get; set; }

    public long FalsePositive { get; set; }

    public double TruePositiveRate { get; set; }

    public double FalsePositiveRate { get; set; }

    /// <summary>
    /// 校准所用的测试集样本数，供调用端判断该阈值的可信度。
    /// </summary>
    public long EvaluatedSamples { get; set; }

    public string GeneratedAt { get; set; } = string.Empty;

    public static string ResolvePath(string modelPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
        string directory = Path.GetDirectoryName(modelPath) ?? string.Empty;
        return Path.Combine(directory, Path.GetFileNameWithoutExtension(modelPath) + FileSuffix);
    }

    public void Save(string modelPath)
    {
        string path = ResolvePath(modelPath);
        File.WriteAllText(path, JsonSerializer.Serialize(this, ModelThresholdManifestJsonContext.Default.ModelThresholdManifest));
    }

    /// <summary>
    /// 读取模型旁的阈值清单。清单缺失、损坏、版本不符或阈值越界时返回 false 并给出原因，
    /// 由调用方回退到配置里的固定阈值。
    /// </summary>
    public static bool TryLoad(string modelPath, string expectedMode, out ModelThresholdManifest? manifest, out string? failureReason)
    {
        manifest = null;
        failureReason = null;

        string path;
        try
        {
            path = ResolvePath(modelPath);
        }
        catch (Exception ex)
        {
            failureReason = $"无法解析阈值清单路径：{ex.Message}";
            return false;
        }

        if (!File.Exists(path))
            return false;

        ModelThresholdManifest? loaded;
        try
        {
            loaded = JsonSerializer.Deserialize(
                File.ReadAllText(path),
                ModelThresholdManifestJsonContext.Default.ModelThresholdManifest);
        }
        catch (Exception ex)
        {
            failureReason = $"阈值清单无法解析：{ex.Message}";
            return false;
        }

        if (loaded == null)
        {
            failureReason = "阈值清单内容为空。";
            return false;
        }

        if (loaded.SchemaVersion != CurrentSchemaVersion)
        {
            failureReason = $"阈值清单版本为 {loaded.SchemaVersion}，期望 {CurrentSchemaVersion}。";
            return false;
        }

        if (!string.IsNullOrEmpty(expectedMode) &&
            !string.Equals(loaded.ModelMode, expectedMode, StringComparison.OrdinalIgnoreCase))
        {
            failureReason = $"阈值清单属于 {loaded.ModelMode} 模型，当前为 {expectedMode}。";
            return false;
        }

        if (!double.IsFinite(loaded.RecommendedThreshold) ||
            loaded.RecommendedThreshold < 0 ||
            loaded.RecommendedThreshold > 100)
        {
            failureReason = $"阈值清单中的阈值 {loaded.RecommendedThreshold} 不在 0-100 范围内。";
            return false;
        }

        manifest = loaded;
        return true;
    }
}

/// <summary>
/// 源生成的序列化上下文。调用器启用了 AOT 与裁剪，反射序列化在那里不可用。
/// </summary>
[JsonSourceGenerationOptions(WriteIndented = true, DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(ModelThresholdManifest))]
internal sealed partial class ModelThresholdManifestJsonContext : JsonSerializerContext;
