using Xdows_Model_Config;

namespace Xdows_Model_Maker;

/// <summary>
/// 训练完成后把 ONNX 模型与阈值清单复制到调用器源目录（Xdows-Model-Invoker\），
/// 使 csproj 的 EmbeddedResource/Content 在下次构建时自动带上最新训练产物。
/// </summary>
internal static class TrainingOutputCopier
{
    /// <summary>单个模型的 ONNX 产物名（不含 Pro 分支）。</summary>
    private static readonly string[] SingleOnnxFileNames =
    [
        "Xdows-Model.onnx",
        "Xdows-Model-Flash.onnx"
    ];

    /// <summary>Pro 融合模型文件名；分支模型名与清单名都由它推导。</summary>
    private const string ProFusionFileName = "Xdows-Model-Pro.onnx";

    private static readonly string[] ThresholdManifestNames =
    [
        "Xdows-Model.threshold.json",
        "Xdows-Model-Flash.threshold.json",
        "Xdows-Model-Pro.threshold.json"
    ];

    /// <summary>
    /// 需要复制的 ONNX 产物名。Pro 分支与模型清单从 <see cref="ProBranches"/> /
    /// <see cref="ProModelManifest"/> 推导，新增分支时不会漏复制而让推理端加载失败。
    /// 公开给架构测试，用来锁住"复制清单必须覆盖全部分支 + 清单"这条约定。
    /// </summary>
    internal static IEnumerable<string> EnumerateOnnxFileNames()
    {
        foreach (string fileName in SingleOnnxFileNames)
            yield return fileName;

        yield return ProFusionFileName;
        foreach (ProBranch branch in ProBranches.All)
            yield return ProBranches.FileNameFor(branch, ProFusionFileName);
        yield return ProModelManifest.FileNameFor(ProFusionFileName);
    }

    /// <summary>
    /// 把训练输出目录（AppContext.BaseDirectory）中的产物复制到调用器源目录的模型子目录
    /// （Xdows-Model-Invoker\Models\），目录不存在时创建。
    /// 仅复制实际存在的文件；找不到调用器工程时打印提示并返回 false（不中断训练流程）。
    /// </summary>
    public static bool CopyToInvokerSource()
    {
        string baseDir = AppContext.BaseDirectory;
        string? modelDir = FindInvokerModelDirectory(baseDir);
        if (modelDir == null)
        {
            Console.WriteLine("  [复制] 未找到 Xdows-Model-Invoker 源目录，跳过复制（可手动复制训练产物）。");
            return false;
        }

        Directory.CreateDirectory(modelDir);

        var copied = new List<string>();
        foreach (string fileName in EnumerateOnnxFileNames().Concat(ThresholdManifestNames))
        {
            string sourcePath = Path.Combine(baseDir, fileName);
            if (!File.Exists(sourcePath))
                continue;

            File.Copy(sourcePath, Path.Combine(modelDir, fileName), overwrite: true);
            copied.Add(fileName);
        }

        if (copied.Count == 0)
        {
            Console.WriteLine("  [复制] 训练输出目录中没有可复制的产物。");
            return false;
        }

        Console.WriteLine($"  [复制] {copied.Count} 个训练产物已复制至: {modelDir}");
        foreach (string fileName in copied)
            Console.WriteLine($"         - {fileName}");
        return true;
    }

    /// <summary>
    /// 从训练输出目录向上逐级查找包含 Xdows-Model-Invoker 子目录的仓库根，
    /// 返回其中的模型子目录。公开给架构测试，用来锁住训练端与推理端用的是同一个目录。
    /// </summary>
    internal static string? FindInvokerModelDirectory(string baseDir)
    {
        var directory = new DirectoryInfo(baseDir);
        while (directory != null)
        {
            string candidate = Path.Combine(directory.FullName, "Xdows-Model-Invoker");
            if (Directory.Exists(candidate))
                return ModelLayout.ResolveModelDirectory(candidate);

            directory = directory.Parent;
        }

        return null;
    }
}