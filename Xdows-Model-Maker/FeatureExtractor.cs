using Xdows_Model_Invoker;

namespace Xdows_Model_Maker;

public class FileData
{
    public string FilePath { get; set; } = string.Empty;
    public FileFeatures Features { get; set; } = new FileFeatures();
    public FlashFileFeatures FlashFeatures { get; set; } = new FlashFileFeatures();
    public float[]? ProFeatures { get; set; }
    internal bool ProFeaturesAttempted { get; set; }
    public bool Label { get; set; }

    /// <summary>
    /// 文件内容的 SHA-256 十六进制摘要，仅在 Pro 数据加载模式下计算（见 <see cref="DataLoader"/>）。
    /// Pro 训练用它剔除完全重复的样本；未计算时为 null，切分报告会说明有多少样本未参与去重。
    /// </summary>
    public string? ContentHash { get; set; }
}
