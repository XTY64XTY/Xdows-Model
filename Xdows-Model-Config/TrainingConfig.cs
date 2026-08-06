namespace Xdows_Model_Config;

public static class FeatureSchema
{
    public const int Version = 2;
    public const int StandardFeatureCount = 299;
    public const int FlashFeatureCount = 68;
    public const int ProHybridFeatureCount = 519;
    public const int ProRawStatCount = 120;
    public const int ProStructuralCount = 32;
    public const int ProFusionFeatureCount = 4;
    public const int ProStandardOffset = 0;
    public const int ProFlashOffset = ProStandardOffset + StandardFeatureCount;
    public const int ProRawStatOffset = ProFlashOffset + FlashFeatureCount;
    public const int ProStructuralOffset = ProRawStatOffset + ProRawStatCount;
}

public class TrainingConfig
{
    public string BlackFolder { get; set; } = "D:\\Code\\Model\\Files\\Black";
    public string WhiteFolder { get; set; } = "D:\\Code\\Model\\Files\\White";
    public string ModelPath { get; set; } = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Xdows-Model.zip");
    public string OnnxPath { get; set; } = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Xdows-Model.onnx");
    public string FlashModelPath { get; set; } = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Xdows-Model-Flash.zip");
    public string FlashOnnxPath { get; set; } = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Xdows-Model-Flash.onnx");
    public string ProModelPath { get; set; } = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Xdows-Model-Pro.zip");
    public string ProOnnxPath { get; set; } = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Xdows-Model-Pro.onnx");

    public double StandardThreshold { get; set; } = 92.0;
    public double FlashThreshold { get; set; } = 96.0;
    public double ProThreshold { get; set; } = 94.0;

    public double LearningRate { get; set; } = 0.025;
    public int NumberOfLeaves { get; set; } = 127;
    public int MinimumExampleCountPerLeaf { get; set; } = 16;
    public int NumberOfIterations { get; set; } = 1400;
    public double StandardL1Regularization { get; set; } = 0.02;
    public double StandardL2Regularization { get; set; } = 0.4;
    public int StandardMaximumTreeDepth { get; set; } = 10;
    public double StandardFeatureFraction { get; set; } = 0.9;
    public double StandardSubsampleFraction { get; set; } = 0.85;
    public double StandardTargetFalsePositiveRate { get; set; } = 0.005;
    public int? RandomSeed { get; set; } = 43846;

    public double FlashLearningRate { get; set; } = 0.1;
    public int FlashNumberOfLeaves { get; set; } = 31;
    public int FlashMinimumExampleCountPerLeaf { get; set; } = 8;
    public int FlashNumberOfIterations { get; set; } = 800;
    public double FlashL1Regularization { get; set; } = 0.01;
    public double FlashL2Regularization { get; set; } = 0.2;
    public int FlashMaximumTreeDepth { get; set; } = 5;

    public double ProLearningRate { get; set; } = 0.01;
    public int ProNumberOfLeaves { get; set; } = 63;
    public int ProMinimumExampleCountPerLeaf { get; set; } = 10;
    public int ProNumberOfIterations { get; set; } = 1200;
    public double ProL1Regularization { get; set; } = 0.01;
    public double ProL2Regularization { get; set; } = 0.1;
    public int ProMaximumTreeDepth { get; set; } = 8;
    public double ProFeatureFraction { get; set; } = 0.85;
    public double ProSubsampleFraction { get; set; } = 0.8;
    public int ProMaxParallelBranches { get; set; } = 4;

    public int? TrainingThreadCount { get; set; }
    public bool ForceColumnWiseHistogram { get; set; } = true;

    /// <summary>
    /// 一个误报相当于多少个漏报。用于阈值选择与训练样本加权，使两者的优化目标一致。
    /// </summary>
    public double FalsePositiveCostRatio { get; set; } = 2.65;

    /// <summary>
    /// 是否按 <see cref="FalsePositiveCostRatio"/> 自动挑选判毒阈值，覆盖固定阈值。
    /// </summary>
    public bool UseCostSensitiveThreshold { get; set; } = true;

    /// <summary>
    /// 是否把误报代价传导到 LightGBM 的正类权重，让训练目标与阈值目标一致。
    /// </summary>
    public bool UseCostSensitiveTrainingWeight { get; set; } = true;

    /// <summary>
    /// LightGBM 正类（黑样本）权重。默认由 <see cref="FalsePositiveCostRatio"/> 推导为 1/代价比。
    /// </summary>
    public double? WeightOfPositiveExamples { get; set; }

    /// <summary>
    /// 解析实际使用的正类权重。误报越贵，正类权重越低，模型越不愿意把白样本判黑。
    /// </summary>
    public double ResolveWeightOfPositiveExamples()
    {
        if (WeightOfPositiveExamples is { } explicitWeight)
        {
            if (!double.IsFinite(explicitWeight) || explicitWeight <= 0)
                throw new InvalidOperationException("正类权重必须是正有限值。");
            return explicitWeight;
        }

        if (!UseCostSensitiveTrainingWeight)
            return 1.0;

        if (!double.IsFinite(FalsePositiveCostRatio) || FalsePositiveCostRatio <= 0)
            throw new InvalidOperationException("误报代价比必须是正有限值。");

        return 1.0 / FalsePositiveCostRatio;
    }

    public void PrintThreadingConfig()
    {
        string threadLabel = TrainingThreadCount is { } configured && configured > 0
            ? configured.ToString()
            : "物理核心数";
        Console.WriteLine($"LightGBM 线程数: {threadLabel}");
        Console.WriteLine($"强制列向直方图: {ForceColumnWiseHistogram}");
        Console.WriteLine($"误报代价比 (1 误报 = N 漏报): {FalsePositiveCostRatio}");
        Console.WriteLine($"代价敏感阈值选择: {UseCostSensitiveThreshold}");
        Console.WriteLine($"正类权重: {ResolveWeightOfPositiveExamples():F4}");
    }

    public void PrintStandardConfig()
    {
        Console.WriteLine("\n=== Standard 模型配置 ===");
        Console.WriteLine($"学习率 (Learning Rate): {LearningRate}");
        Console.WriteLine($"叶子数 (Number of Leaves): {NumberOfLeaves}");
        Console.WriteLine($"最小叶节点样本数: {MinimumExampleCountPerLeaf}");
        Console.WriteLine($"迭代次数 (Iterations): {NumberOfIterations}");
        Console.WriteLine($"L1 正则化: {StandardL1Regularization}");
        Console.WriteLine($"L2 正则化: {StandardL2Regularization}");
        Console.WriteLine($"最大树深度: {StandardMaximumTreeDepth}");
        Console.WriteLine($"特征采样比例: {StandardFeatureFraction}");
        Console.WriteLine($"样本采样比例: {StandardSubsampleFraction}");
        Console.WriteLine($"阈值校准目标 FPR: {StandardTargetFalsePositiveRate:P2}");
        Console.WriteLine($"判毒阈值: {StandardThreshold}%");
        Console.WriteLine($"随机种子: {RandomSeed}");
        PrintThreadingConfig();
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
        Console.WriteLine($"最大树深度: {FlashMaximumTreeDepth}");
        Console.WriteLine($"判毒阈值: {FlashThreshold}%");
        PrintThreadingConfig();
        Console.WriteLine("========================\n");
    }

    public void PrintProConfig()
    {
        Console.WriteLine("\n=== Pro 模型配置 ===");
        Console.WriteLine("训练算法: GBDT (LightGBM)");
        Console.WriteLine("架构: Standard / Flash / RawStat / PE结构 四分支 + OOF逻辑回归融合");
        Console.WriteLine($"学习率 (Learning Rate): {ProLearningRate}");
        Console.WriteLine($"叶子数 (Number of Leaves): {ProNumberOfLeaves}");
        Console.WriteLine($"最小叶节点样本数: {ProMinimumExampleCountPerLeaf}");
        Console.WriteLine($"迭代次数 (Iterations): {ProNumberOfIterations}");
        Console.WriteLine($"L1 正则化: {ProL1Regularization}");
        Console.WriteLine($"L2 正则化: {ProL2Regularization}");
        Console.WriteLine($"最大树深度: {ProMaximumTreeDepth}");
        Console.WriteLine($"特征采样比例: {ProFeatureFraction}");
        Console.WriteLine($"样本采样比例: {ProSubsampleFraction}");
        Console.WriteLine($"并行分支数: {ProMaxParallelBranches}");
        PrintThreadingConfig();
        Console.WriteLine($"判毒阈值: {ProThreshold}%");
        Console.WriteLine($"Raw 统计特征: 3 段 × 40 维 = 120 维 (固定)");
        Console.WriteLine($"总特征维度: 519 (299 + 68 + 120 + 32)");
        Console.WriteLine("========================\n");
    }
}
