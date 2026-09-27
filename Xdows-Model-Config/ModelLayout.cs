namespace Xdows_Model_Config;

/// <summary>
/// 模型产物的部署布局约定。训练端（Maker 把产物复制回 Invoker 源目录）、推理端
/// （Invoker 按目录加载模型）、原生运行库（xdows_model_native.cpp 的 ResolveModelPath）
/// 以及宿主程序（Xdows-Security 的资产收集与自检）共享这一份定义，
/// 避免各方各写一份目录名与查找顺序而漂移。
/// </summary>
public static class ModelLayout
{
    public const string StandardFileName = "Xdows-Model.onnx";
    public const string FlashFileName = "Xdows-Model-Flash.onnx";

    /// <summary>Pro 融合模型文件名；分支模型名与模型清单名都由它推导。</summary>
    public const string ProFusionFileName = "Xdows-Model-Pro.onnx";

    /// <summary>模型文件所在子目录名，相对程序集目录。</summary>
    public const string ModelDirectoryName = "Models";

    /// <summary>
    /// 推理端必须齐备的模型文件：融合模型 + 各分支模型 + Pro 模型清单。
    /// 不含 <c>*.threshold.json</c>：它缺失时推理端回退到固定阈值，属于降级而非不可用。
    /// </summary>
    public static IReadOnlyList<string> RequiredFileNames { get; } = BuildRequiredFileNames();

    /// <summary>把模型文件名解析成相对程序集目录的标准相对路径，例如 <c>Models\Xdows-Model.onnx</c>。</summary>
    public static string RelativePath(string fileName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(fileName);
        return Path.Combine(ModelDirectoryName, fileName);
    }

    /// <summary>解析模型目录：<paramref name="baseDirectory"/> 下的 <see cref="ModelDirectoryName"/> 子目录。</summary>
    public static string ResolveModelDirectory(string baseDirectory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(baseDirectory);
        return Path.Combine(baseDirectory, ModelDirectoryName);
    }

    /// <summary>
    /// 按推理端的查找顺序解析一个已存在的模型文件：先 <c>Models\</c>，再回退目录本身
    /// （兼容模型直接摊在程序集目录的旧部署）。两处都不存在时返回 null，
    /// 需要"必须存在"语义的调用方自行抛出带两个候选路径的异常。
    /// </summary>
    public static string? ResolveExistingModelPath(string baseDirectory, string fileName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(baseDirectory);
        ArgumentException.ThrowIfNullOrWhiteSpace(fileName);

        string preferred = Path.Combine(ResolveModelDirectory(baseDirectory), fileName);
        if (File.Exists(preferred))
            return preferred;

        string legacy = Path.Combine(baseDirectory, fileName);
        return File.Exists(legacy) ? legacy : null;
    }

    private static string[] BuildRequiredFileNames()
    {
        var names = new List<string> { StandardFileName, FlashFileName, ProFusionFileName };
        foreach (ProBranch branch in ProBranches.All)
            names.Add(ProBranches.FileNameFor(branch, ProFusionFileName));
        names.Add(ProModelManifest.FileNameFor(ProFusionFileName));
        return names.ToArray();
    }
}
