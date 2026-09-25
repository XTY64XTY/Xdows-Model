namespace Xdows_Model_Config;

public static class FeatureSchema
{
    public const int Version = 3;
    public const int StandardFeatureCount = 299;
    public const int FlashFeatureCount = 68;
    public const int ProHybridFeatureCount = 5143;
    public const int ProRawStatCount = 120;
    public const int ProStructuralCount = 32;
    public const int ProImportBehaviorCount = ImportHashConfig.DllHashDimensions + ImportHashConfig.ApiHashDimensions + ProImportStatsCount;
    public const int ProImportStatsCount = 16;
    public const int ProFusionFeatureCount = 5;
    public const int ProStandardOffset = 0;
    public const int ProFlashOffset = ProStandardOffset + StandardFeatureCount;
    public const int ProRawStatOffset = ProFlashOffset + FlashFeatureCount;
    public const int ProStructuralOffset = ProRawStatOffset + ProRawStatCount;
    public const int ProImportBehaviorOffset = ProStructuralOffset + ProStructuralCount;

    /// <summary>
    /// 旧版（Schema Version 2）四分支 Pro 模型的混合特征维度。新特征是追加式布局，
    /// 旧模型的偏移在新版中保持不变，推理端据此兼容旧产物。
    /// </summary>
    public const int ProLegacyHybridFeatureCount = 519;

    /// <summary>
    /// 旧版四分支 Pro 模型的融合输入维度（Standard/Flash/RawStat/Structural 四路概率）。
    /// </summary>
    public const int ProLegacyFusionFeatureCount = 4;
}

/// <summary>
/// ImportBehavior 特征组的哈希配置。算法、种子与维度在这里显式固定，
/// 训练端与推理端共享同一组常量；任何修改都必须同步提升 <see cref="FeatureSchema.Version"/>，
/// 模型 Manifest 会在加载时校验这些值，不匹配即拒绝加载，不允许隐式漂移。
/// </summary>
public static class ImportHashConfig
{
    /// <summary>哈希算法标识。当前为 32 位 FNV-1a，种子即偏移基值。</summary>
    public const string Algorithm = "Fnv1a32";
    public const uint Seed = 2166136261u;
    public const int DllHashDimensions = 512;
    public const int ApiHashDimensions = 4096;

    /// <summary>DLL 名在哈希前的规范化：小写、去首尾空白。训练与推理必须一致。</summary>
    public static string NormalizeDllName(string name) => name.Trim().ToLowerInvariant();

    /// <summary>API 名保持原始大小写（PE 导入名本身区分大小写），仅去空白。</summary>
    public static string NormalizeApiName(string name) => name.Trim();
}

/// <summary>
/// Pro 训练集的外层切分方式。切分只影响训练数据的划分，不改变特征提取与推理行为。
/// </summary>
public enum ProDatasetSplitMode
{
    /// <summary>按类别分层随机切分（默认，保持既有行为）。</summary>
    StratifiedRandom = 0,

    /// <summary>
    /// 按样本时间键排序后切分：较早的样本进训练集，最晚的一段进测试集。
    /// 需要每个样本都带 <c>TimeKey</c>，否则明确报错，不做静默退化。
    /// </summary>
    TimeOrdered = 1
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

    /// <summary>Pro 外层测试集比例，取值必须落在 (0, 1) 开区间。</summary>
    public double ProTestFraction { get; set; } = 0.2;

    /// <summary>Pro 训练集切分方式，见 <see cref="ProDatasetSplitMode"/>。</summary>
    public ProDatasetSplitMode ProSplitMode { get; set; } = ProDatasetSplitMode.StratifiedRandom;

    /// <summary>
    /// 是否按文件内容哈希（SHA-256）去除完全重复的样本。重复样本会同时污染训练集与测试集，
    /// 让测试指标虚高，因此默认开启。
    /// </summary>
    public bool ProDeduplicateByContentHash { get; set; } = true;

    /// <summary>
    /// 是否要求携带相同 <c>GroupKey</c>（如恶意软件家族）的样本整体落在训练集或测试集的同一侧。
    /// 样本未提供 <c>GroupKey</c> 时该项不生效，切分报告会明确说明未启用分组隔离。
    /// </summary>
    public bool ProEnforceGroupIsolation { get; set; } = true;

    /// <summary>解析并校验 Pro 外层测试集比例。</summary>
    public double ResolveProTestFraction()
    {
        if (!double.IsFinite(ProTestFraction) || ProTestFraction <= 0 || ProTestFraction >= 1)
            throw new InvalidOperationException("Pro 测试集比例必须在 0 与 1 之间（不含端点）。");
        return ProTestFraction;
    }

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
        Console.WriteLine("架构: Standard / Flash / RawStat / PE结构 / 导入行为 五分支 + OOF逻辑回归融合");
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
        Console.WriteLine($"导入行为特征: DLL {ImportHashConfig.DllHashDimensions} + API {ImportHashConfig.ApiHashDimensions} + 统计 {FeatureSchema.ProImportStatsCount} 维 ({ImportHashConfig.Algorithm}, 种子 {ImportHashConfig.Seed})");
        Console.WriteLine($"总特征维度: {FeatureSchema.ProHybridFeatureCount} (299 + 68 + 120 + 32 + {FeatureSchema.ProImportBehaviorCount})");
        Console.WriteLine($"外层切分方式: {(ProSplitMode == ProDatasetSplitMode.TimeOrdered ? "时间切分" : "分层随机切分")}，测试集比例: {ResolveProTestFraction():P2}");
        Console.WriteLine($"内容哈希去重: {ProDeduplicateByContentHash}，分组隔离: {ProEnforceGroupIsolation}");
        Console.WriteLine("========================\n");
    }
}
