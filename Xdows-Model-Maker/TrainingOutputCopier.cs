using Xdows_Model_Config;

namespace Xdows_Model_Maker;

/// <summary>
/// 训练完成后把 ONNX 模型与阈值清单复制到调用器源目录（Xdows-Model-Invoker\），
/// 使 csproj 的 EmbeddedResource/Content 在下次构建时自动带上最新训练产物。
/// </summary>
internal static class TrainingOutputCopier
{
    /// <summary>
    /// 阈值清单不是硬性依赖（缺失时推理端回退到固定阈值），但要跟着模型一起带走，
    /// 否则三档判定会退化成二档。文件名由模型名加 <see cref="ModelThresholdManifest.FileSuffix"/> 推导。
    /// </summary>
    private static readonly string[] ThresholdManifestNames = BuildThresholdManifestNames();

    /// <summary>
    /// 需要复制的 ONNX 产物名。清单本身由 <see cref="ModelLayout.RequiredFileNames"/> 唯一决定
    /// （Pro 分支与模型清单都从 <see cref="ProBranches"/> / <see cref="ProModelManifest"/> 推导），
    /// 新增分支时训练端不会漏复制、推理端不会加载失败。
    /// 公开给架构测试，用来锁住"训练端复制清单 == 推理端必需清单"这条约定。
    /// </summary>
    internal static IEnumerable<string> EnumerateOnnxFileNames() => ModelLayout.RequiredFileNames;

    private static string[] BuildThresholdManifestNames()
    {
        string[] modelFileNames =
        [
            ModelLayout.StandardFileName,
            ModelLayout.FlashFileName,
            ModelLayout.ProFusionFileName
        ];

        return modelFileNames
            .Select(Path.GetFileNameWithoutExtension)
            .Select(name => name + ModelThresholdManifest.FileSuffix)
            .ToArray();
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