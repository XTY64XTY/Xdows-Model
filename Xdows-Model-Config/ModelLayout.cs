namespace Xdows_Model_Config;

/// <summary>
/// 模型产物的部署布局约定。训练端（Maker 把产物复制回 Invoker 源目录）与
/// 推理端（Invoker 按目录加载模型）共享这一份定义，避免两边各写一份目录名而漂移。
/// </summary>
public static class ModelLayout
{
    /// <summary>模型文件所在子目录名，相对程序集目录。</summary>
    public const string ModelDirectoryName = "Models";

    /// <summary>解析模型目录：<paramref name="baseDirectory"/> 下的 <see cref="ModelDirectoryName"/> 子目录。</summary>
    public static string ResolveModelDirectory(string baseDirectory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(baseDirectory);
        return Path.Combine(baseDirectory, ModelDirectoryName);
    }
}
