namespace Xdows_Model_Maker;

public class TrainingConfig
{
    public string BlackFolder { get; set; } = @"D:\Code\Model\Files\Black";
    public string WhiteFolder { get; set; } = @"D:\Code\Model\Files\White";
    public string ModelPath { get; set; } = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Xdows-Model.zip");
    public string OnnxPath { get; set; } = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Xdows-Model.onnx");
    public string FlashModelPath { get; set; } = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Xdows-Model-Flash.zip");
    public string FlashOnnxPath { get; set; } = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Xdows-Model-Flash.onnx");
    public string ProModelPath { get; set; } = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Xdows-Model-Pro.zip");
    public string ProOnnxPath { get; set; } = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Xdows-Model-Pro.onnx");

    public int ProExpansionStartBytesPerSection { get; set; } = 512;
    public int ProExpansionFactor { get; set; } = 2;
    public int ProExpansionMaxBytesPerSection { get; set; } = 4096;
    public double ProExpansionAucThreshold { get; set; } = 0.001;
    public int ProExpansionPatience { get; set; } = 2;

    public double LearningRate { get; set; } = 0.1;
    public int NumberOfLeaves { get; set; } = 31;
    public int MinimumExampleCountPerLeaf { get; set; } = 20;
    public int NumberOfIterations { get; set; } = 400;
    public int? RandomSeed { get; set; } = 42;

    public double FlashLearningRate { get; set; } = 0.05;
    public int FlashNumberOfLeaves { get; set; } = 20;
    public int FlashMinimumExampleCountPerLeaf { get; set; } = 10;
    public int FlashNumberOfIterations { get; set; } = 500;

    public void PrintConfig()
    {
        Console.WriteLine("\n=== 训练配置 ===");
        Console.WriteLine($"黑文件目录: {BlackFolder}");
        Console.WriteLine($"白文件目录: {WhiteFolder}");
        Console.WriteLine($"ML.NET 模型路径: {ModelPath}");
        Console.WriteLine($"ONNX 模型路径: {OnnxPath}");
        Console.WriteLine($"Flash ML.NET 模型路径: {FlashModelPath}");
        Console.WriteLine($"Flash ONNX 模型路径: {FlashOnnxPath}");
        Console.WriteLine($"Pro ML.NET 模型路径: {ProModelPath}");
        Console.WriteLine($"Pro ONNX 模型路径: {ProOnnxPath}");
        Console.WriteLine($"\nPro 渐进式扩展起始字节/段: {ProExpansionStartBytesPerSection}");
        Console.WriteLine($"Pro 渐进式扩展因子: {ProExpansionFactor}");
        Console.WriteLine($"Pro 渐进式扩展最大字节/段: {ProExpansionMaxBytesPerSection}");
        Console.WriteLine($"Pro 渐进式扩展 AUC 阈值: {ProExpansionAucThreshold}");
        Console.WriteLine($"Pro 渐进式扩展耐心步数: {ProExpansionPatience}");
        Console.WriteLine($"学习率 (Learning Rate): {LearningRate}");
        Console.WriteLine($"叶子数 (Number of Leaves): {NumberOfLeaves}");
        Console.WriteLine($"最小叶节点样本数: {MinimumExampleCountPerLeaf}");
        Console.WriteLine($"迭代次数 (Iterations): {NumberOfIterations}");
        Console.WriteLine($"随机种子: {RandomSeed}");
        Console.WriteLine($"\nFlash 专用超参数:");
        Console.WriteLine($"  Flash 学习率: {FlashLearningRate}");
        Console.WriteLine($"  Flash 叶子数: {FlashNumberOfLeaves}");
        Console.WriteLine($"  Flash 最小叶节点样本数: {FlashMinimumExampleCountPerLeaf}");
        Console.WriteLine($"  Flash 迭代次数: {FlashNumberOfIterations}");
        Console.WriteLine("================\n");
    }
}
