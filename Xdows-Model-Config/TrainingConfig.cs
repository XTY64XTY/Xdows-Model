namespace Xdows_Model_Config;

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

    public double StandardThreshold { get; set; } = 92.0;
    public double FlashThreshold { get; set; } = 96.0;
    public double ProThreshold { get; set; } = 94.0;

    public int ProExpansionStartBytesPerSection { get; set; } = 512;
    public int ProExpansionFactor { get; set; } = 2;
    public int ProExpansionMaxBytesPerSection { get; set; } = 4096;
    public double ProExpansionAucThreshold { get; set; } = 0.0005;
    public int ProExpansionPatience { get; set; } = 3;

    public double LearningRate { get; set; } = 0.005;
    public int NumberOfLeaves { get; set; } = 63;
    public int MinimumExampleCountPerLeaf { get; set; } = 31;
    public int NumberOfIterations { get; set; } = 1024;
    public double StandardL1Regularization { get; set; } = 0.01;
    public double StandardL2Regularization { get; set; } = 0.1;
    public int? RandomSeed { get; set; } = 43846;

    public double FlashLearningRate { get; set; } = 0.1;
    public int FlashNumberOfLeaves { get; set; } = 31;
    public int FlashMinimumExampleCountPerLeaf { get; set; } = 8;
    public int FlashNumberOfIterations { get; set; } = 800;
    public double FlashL1Regularization { get; set; } = 0.01;
    public double FlashL2Regularization { get; set; } = 0.2;

    public double ProLearningRate { get; set; } = 0.01;
    public int ProNumberOfLeaves { get; set; } = 63;
    public int ProMinimumExampleCountPerLeaf { get; set; } = 10;
    public int ProNumberOfIterations { get; set; } = 1200;
    public double ProL1Regularization { get; set; } = 0.01;
    public double ProL2Regularization { get; set; } = 0.1;

    public void PrintStandardConfig()
    {
        Console.WriteLine("\n=== Standard 模型配置 ===");
        Console.WriteLine($"学习率 (Learning Rate): {LearningRate}");
        Console.WriteLine($"叶子数 (Number of Leaves): {NumberOfLeaves}");
        Console.WriteLine($"最小叶节点样本数: {MinimumExampleCountPerLeaf}");
        Console.WriteLine($"迭代次数 (Iterations): {NumberOfIterations}");
        Console.WriteLine($"L1 正则化: {StandardL1Regularization}");
        Console.WriteLine($"L2 正则化: {StandardL2Regularization}");
        Console.WriteLine($"判毒阈值: {StandardThreshold}%");
        Console.WriteLine($"随机种子: {RandomSeed}");
        Console.WriteLine("========================\n");
    }

    public void PrintFlashConfig()
    {
        Console.WriteLine("\n=== Flash 模型配置 ===");
        Console.WriteLine($"学习率 (Learning Rate): {FlashLearningRate}");
        Console.WriteLine($"叶子数 (Number of Leaves): {FlashNumberOfLeaves}");
        Console.WriteLine($"最小叶节点样本数: {FlashMinimumExampleCountPerLeaf}");
        Console.WriteLine($"迭代次数 (Iterations): {FlashNumberOfIterations}");
        Console.WriteLine($"L1 正则化: {FlashL1Regularization}");
        Console.WriteLine($"L2 正则化: {FlashL2Regularization}");
        Console.WriteLine($"判毒阈值: {FlashThreshold}%");
        Console.WriteLine("========================\n");
    }

    public void PrintProConfig()
    {
        Console.WriteLine("\n=== Pro 模型配置 ===");
        Console.WriteLine($"Pro 渐进式扩展起始字节/段: {ProExpansionStartBytesPerSection}");
        Console.WriteLine($"Pro 渐进式扩展因子: {ProExpansionFactor}");
        Console.WriteLine($"Pro 渐进式扩展最大字节/段: {ProExpansionMaxBytesPerSection}");
        Console.WriteLine($"Pro 渐进式扩展 AUC 阈值: {ProExpansionAucThreshold}");
        Console.WriteLine($"Pro 渐进式扩展耐心步数: {ProExpansionPatience}");
        Console.WriteLine($"学习率 (Learning Rate): {ProLearningRate}");
        Console.WriteLine($"叶子数 (Number of Leaves): {ProNumberOfLeaves}");
        Console.WriteLine($"最小叶节点样本数: {ProMinimumExampleCountPerLeaf}");
        Console.WriteLine($"迭代次数 (Iterations): {ProNumberOfIterations}");
        Console.WriteLine($"L1 正则化: {ProL1Regularization}");
        Console.WriteLine($"L2 正则化: {ProL2Regularization}");
        Console.WriteLine($"判毒阈值: {ProThreshold}%");
        Console.WriteLine("========================\n");
    }
}
