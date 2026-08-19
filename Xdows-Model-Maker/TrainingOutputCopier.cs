namespace Xdows_Model_Maker;

/// <summary>
/// 训练完成后把 ONNX 模型与阈值清单复制到调用器源目录（Xdows-Model-Invoker\），
/// 使 csproj 的 EmbeddedResource/Content 在下次构建时自动带上最新训练产物。
/// </summary>
internal static class TrainingOutputCopier
{
    /// <summary>
    /// 调用器嵌入的 ONNX 产物名（与 Xdows-Model-Invoker.csproj 的 EmbeddedResource 保持一致）。
    /// </summary>
    private static readonly string[] OnnxFileNames =
    [
        "Xdows-Model.onnx",
        "Xdows-Model-Flash.onnx",
        "Xdows-Model-Pro.onnx",
        "Xdows-Model-Pro-Standard.onnx",
        "Xdows-Model-Pro-Flash.onnx",
        "Xdows-Model-Pro-RawStat.onnx",
        "Xdows-Model-Pro-Structural.onnx"
    ];

    private static readonly string[] ThresholdManifestNames =
    [
        "Xdows-Model.threshold.json",
        "Xdows-Model-Flash.threshold.json",
        "Xdows-Model-Pro.threshold.json"
    ];

    /// <summary>
    /// 把训练输出目录（AppContext.BaseDirectory）中的产物复制到调用器源目录。
    /// 仅复制实际存在的文件；找不到源目录时打印提示并返回 false（不中断训练流程）。
    /// </summary>
    public static bool CopyToInvokerSource()
    {
        string baseDir = AppContext.BaseDirectory;
        string? invokerDir = FindInvokerSourceDirectory(baseDir);
        if (invokerDir == null)
        {
            Console.WriteLine("  [复制] 未找到 Xdows-Model-Invoker 源目录，跳过复制（可手动复制训练产物）。");
            return false;
        }

        var copied = new List<string>();
        foreach (string fileName in OnnxFileNames.Concat(ThresholdManifestNames))
        {
            string sourcePath = Path.Combine(baseDir, fileName);
            if (!File.Exists(sourcePath))
                continue;

            File.Copy(sourcePath, Path.Combine(invokerDir, fileName), overwrite: true);
            copied.Add(fileName);
        }

        if (copied.Count == 0)
        {
            Console.WriteLine("  [复制] 训练输出目录中没有可复制的产物。");
            return false;
        }

        Console.WriteLine($"  [复制] {copied.Count} 个训练产物已复制至: {invokerDir}");
        foreach (string fileName in copied)
            Console.WriteLine($"         - {fileName}");
        return true;
    }

    /// <summary>
    /// 从训练输出目录向上逐级查找包含 Xdows-Model-Invoker 子目录的仓库根。
    /// </summary>
    private static string? FindInvokerSourceDirectory(string baseDir)
    {
        var directory = new DirectoryInfo(baseDir);
        while (directory != null)
        {
            string candidate = Path.Combine(directory.FullName, "Xdows-Model-Invoker");
            if (Directory.Exists(candidate))
                return candidate;

            directory = directory.Parent;
        }

        return null;
    }
}