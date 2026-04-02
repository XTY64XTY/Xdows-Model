namespace Xdows_Model_Maker;

public class TrainingConfig
{
    public string BlackFolder { get; set; } = @"D:\Code\Model\Files\Black";
    public string WhiteFolder { get; set; } = @"D:\Code\Model\Files\White";
    public string ModelPath { get; set; } = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Xdows-Model.zip");
    public string OnnxPath { get; set; } = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Xdows-Model.onnx");

    public double LearningRate { get; set; } = 0.1;
    public int NumberOfLeaves { get; set; } = 31;
    public int MinimumExampleCountPerLeaf { get; set; } = 20;
    public int NumberOfIterations { get; set; } = 400;
    public int? RandomSeed { get; set; } = 42;

    public void PrintConfig()
    {
        Console.WriteLine("\n=== 训练配置 ===");
        Console.WriteLine($"黑文件目录: {BlackFolder}");
        Console.WriteLine($"白文件目录: {WhiteFolder}");
        Console.WriteLine($"ML.NET 模型路径: {ModelPath}");
        Console.WriteLine($"ONNX 模型路径: {OnnxPath}");
        Console.WriteLine($"\n学习率 (Learning Rate): {LearningRate}");
        Console.WriteLine($"叶子数 (Number of Leaves): {NumberOfLeaves}");
        Console.WriteLine($"最小叶节点样本数: {MinimumExampleCountPerLeaf}");
        Console.WriteLine($"迭代次数 (Iterations): {NumberOfIterations}");
        Console.WriteLine($"随机种子: {RandomSeed}");
        Console.WriteLine("================\n");
    }
}
