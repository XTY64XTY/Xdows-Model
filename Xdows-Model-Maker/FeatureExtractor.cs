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
}
