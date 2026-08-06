using Microsoft.ML;
using Microsoft.ML.Data;
using Xdows_Model_Config;
using Xdows_Model_Invoker;

namespace Xdows_Model_Maker;

public class ModelTrainer
{
    private readonly MLContext _mlContext;
    private readonly TrainingConfig _config;
    private readonly ProGbdtLearner _proLearner;
    private ProStackingTrainingResult? _lastProTrainingResult;
    private volatile bool _proTrainingCancelled;

    public ModelTrainer(TrainingConfig config)
    {
        _config = config;
        _mlContext = new MLContext(seed: config.RandomSeed);
        _proLearner = new ProGbdtLearner();
        _proTrainingCancelled = false;
    }

    public void CancelProTraining()
    {
        _proTrainingCancelled = true;
    }

    public ITransformer TrainModel(List<FileData> fileData, string modelPath, string? onnxPath = null)
    {
        return TrainCore(fileData, modelPath, onnxPath, flash: false);
    }

    public ITransformer TrainFlashModel(List<FileData> fileData, string modelPath, string? onnxPath = null)
    {
        return TrainCore(fileData, modelPath, onnxPath, flash: true);
    }

    public ITransformer TrainModel(List<FileData> fileData)
    {
        return TrainModel(fileData, _config.ModelPath, _config.OnnxPath);
    }

    public ITransformer TrainFlashModel(List<FileData> fileData)
    {
        return TrainFlashModel(fileData, _config.FlashModelPath, _config.FlashOnnxPath);
    }

    public ITransformer? TrainProModel(List<FileData> fileData, string modelPath, string? onnxPath = null)
    {
        return TrainPro(fileData, modelPath, onnxPath);
    }

    public ITransformer? TrainProModel(List<FileData> fileData)
    {
        return TrainProModel(fileData, _config.ProModelPath, _config.ProOnnxPath);
    }

    private ITransformer? TrainPro(List<FileData> fileData, string modelPath, string? onnxPath)
    {
        Console.WriteLine("\n开始训练 Pro GBDT 混合特征模型...");
        Console.WriteLine($"固定特征组成：Standard {FileFeatures.FeatureCount} 维 + Flash {FlashFileFeatures.FeatureCount} 维 + RawStat {ProRawStatFeatures.TotalCount} 维 + PE结构 {ProHybridFileFeatures.StructuralFeatureCount} 维");
        Console.WriteLine($"总特征维度：{ProHybridFileFeatures.FeatureCount}\n");

        if (_proTrainingCancelled)
        {
            Console.WriteLine("Pro 训练已取消。");
            return null;
        }

        var featureCache = ProFeatureCache.Build(fileData);
        if (_proTrainingCancelled)
        {
            Console.WriteLine("Pro 训练已取消。");
            return null;
        }

        var result = TrainProStep(featureCache);
        if (result == null)
        {
            Console.WriteLine("警告：Pro 混合特征模型未产生有效模型。");
            return null;
        }

        Console.WriteLine($"\n正在保存 Pro 模型...");
        try
        {
            result.SaveArtifacts(_mlContext, modelPath, onnxPath);
            Console.WriteLine($"Pro 融合模型和 4 个分支模型已保存至: {Path.GetDirectoryName(modelPath)}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Pro 模型导出失败：{ex.Message}");
            return null;
        }

        _lastProTrainingResult = result;
        WriteProEvaluationReport(modelPath, result.Evaluation);
        WriteProThresholdManifest(modelPath, onnxPath, result.Evaluation);

        Console.WriteLine($"\n=== Pro GBDT 混合特征模型训练完成 ===");
        Console.WriteLine($"最终特征维度：{ProHybridFileFeatures.FeatureCount}");
        Console.WriteLine($"测试集 AUC：{result.Evaluation.TestAuc:P4}");
        Console.WriteLine($"训练集 AUC：{result.Evaluation.TrainAuc:P4}");
        Console.WriteLine($"AUC Gap：{result.Evaluation.AucGap:P4}");

        return result.FusionModel;
    }

    private ProStackingTrainingResult? TrainProStep(ProFeatureCache featureCache)
    {
        int featureCount = ProHybridFileFeatures.FeatureCount;

        var samples = new List<ProStackingSample>(featureCache.Entries.Count);
        int emptyFeaturesCount = 0;

        foreach (var entry in featureCache.Entries)
        {
            if (entry.Features.Length != featureCount)
            {
                emptyFeaturesCount++;
            }
            else
            {
                samples.Add(new ProStackingSample(entry.Features, entry.Label));
            }
        }

        if (emptyFeaturesCount > 0)
        {
            Console.WriteLine($"  警告：{emptyFeaturesCount} 个缓存样本组装失败");
            Console.WriteLine($"  提示：可使用选项7「清洗非PE文件（含Pro兼容性检查）」功能清理不兼容的文件");
        }

        if (featureCache.FailedCount > 0)
            Console.WriteLine($"  Pro 缓存阶段已跳过 {featureCache.FailedCount} 个不兼容文件");

        if (samples.Count == 0)
        {
            Console.WriteLine("  错误：没有有效的训练数据！");
            return null;
        }

        Console.WriteLine($"  有效训练数据：{samples.Count} 个");

        var blackCount = samples.Count(d => d.Label);
        var whiteCount = samples.Count(d => !d.Label);
        Console.WriteLine($"  黑文件：{blackCount}，白文件：{whiteCount}");

        if (blackCount == 0 || whiteCount == 0)
        {
            Console.WriteLine("  错误：有效数据中只有一类标签，无法训练！");
            return null;
        }

        Console.WriteLine($"  正在训练 Pro {_proLearner.Name} Stacking 模型...");
        var result = new ProStackingTrainer(_mlContext, _config, _proLearner).Train(samples);
        var evaluation = result.Evaluation;
        Console.WriteLine($"  测试集 AUC: {evaluation.TestAuc:P4}，AUPRC: {evaluation.TestAuprc:P4}");
        Console.WriteLine($"  检出率: {evaluation.TestThresholdMetrics.TruePositiveRate:P4}，误报率: {evaluation.TestThresholdMetrics.FalsePositiveRate:P4}");
        Console.WriteLine($"  OOF 训练 AUC: {evaluation.TrainAuc:P4}，AUC Gap: {evaluation.AucGap:P4}");
        return result;
    }

    private void WriteProEvaluationReport(string modelPath, ProTrainingEvaluation evaluation)
    {
        try
        {
            string reportPath = Path.ChangeExtension(modelPath, ".evaluation.json");
            var report = new
            {
                GeneratedAt = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
                ModelType = "Pro",
                HybridFeatureCount = FeatureSchema.ProHybridFeatureCount,
                FusionFeatureCount = evaluation.FeatureCount,
                Architecture = "4x GBDT branches + OOF logistic regression fusion",
                Samples = new
                {
                    Total = evaluation.TotalSamples,
                    Black = evaluation.BlackSamples,
                    White = evaluation.WhiteSamples
                },
                TestMetrics = new
                {
                    Auc = evaluation.TestAuc,
                    Auprc = evaluation.TestAuprc,
                    Accuracy = evaluation.TestThresholdMetrics.Accuracy,
                    TruePositiveRate = evaluation.TestThresholdMetrics.TruePositiveRate,
                    FalsePositiveRate = evaluation.TestThresholdMetrics.FalsePositiveRate,
                    F1Score = evaluation.TestThresholdMetrics.F1Score,
                    ConfusionMatrix = new
                    {
                        TP = evaluation.TestThresholdMetrics.TruePositive,
                        FN = evaluation.TestThresholdMetrics.FalseNegative,
                        FP = evaluation.TestThresholdMetrics.FalsePositive,
                        TN = evaluation.TestThresholdMetrics.TrueNegative
                    }
                },
                TrainMetrics = new { Auc = evaluation.TrainAuc },
                TrainTestGap = new
                {
                    AucGap = evaluation.AucGap,
                    Warning = double.IsNaN(evaluation.AucGap) ? "Evaluation failed" : evaluation.AucGap > 0.05 ? "Possible overfitting" : evaluation.AucGap < -0.02 ? "Anomalous" : null
                },
                BestF1Threshold = new
                {
                    Threshold = evaluation.BestThreshold,
                    F1Score = evaluation.TestBestThresholdMetrics.F1Score,
                    TruePositiveRate = evaluation.TestBestThresholdMetrics.TruePositiveRate,
                    FalsePositiveRate = evaluation.TestBestThresholdMetrics.FalsePositiveRate
                },
                CostSensitiveThreshold = new
                {
                    FalsePositiveCostRatio = _config.FalsePositiveCostRatio,
                    Applied = _config.UseCostSensitiveThreshold,
                    OperatingThreshold = evaluation.OperatingThreshold,
                    Threshold = evaluation.CostThreshold.Threshold,
                    WeightedCost = evaluation.CostThreshold.Cost,
                    TruePositiveRate = evaluation.CostThreshold.Metrics.TruePositiveRate,
                    FalsePositiveRate = evaluation.CostThreshold.Metrics.FalsePositiveRate,
                    ConfusionMatrix = new
                    {
                        TP = evaluation.CostThreshold.Metrics.TruePositive,
                        FN = evaluation.CostThreshold.Metrics.FalseNegative,
                        FP = evaluation.CostThreshold.Metrics.FalsePositive,
                        TN = evaluation.CostThreshold.Metrics.TrueNegative
                    }
                },
                ConfiguredThreshold = new
                {
                    Threshold = _config.ProThreshold,
                    WeightedCost = CostSensitiveThreshold.ComputeCost(evaluation.ConfiguredThresholdMetrics, _config.FalsePositiveCostRatio),
                    TruePositiveRate = evaluation.ConfiguredThresholdMetrics.TruePositiveRate,
                    FalsePositiveRate = evaluation.ConfiguredThresholdMetrics.FalsePositiveRate,
                    ConfusionMatrix = new
                    {
                        TP = evaluation.ConfiguredThresholdMetrics.TruePositive,
                        FN = evaluation.ConfiguredThresholdMetrics.FalseNegative,
                        FP = evaluation.ConfiguredThresholdMetrics.FalsePositive,
                        TN = evaluation.ConfiguredThresholdMetrics.TrueNegative
                    }
                },
                Config = new
                {
                    ProThreshold = _config.ProThreshold,
                    FalsePositiveCostRatio = _config.FalsePositiveCostRatio,
                    WeightOfPositiveExamples = _config.ResolveWeightOfPositiveExamples(),
                    ProLearningRate = _config.ProLearningRate,
                    ProNumberOfLeaves = _config.ProNumberOfLeaves,
                    ProMinimumExampleCountPerLeaf = _config.ProMinimumExampleCountPerLeaf,
                    ProNumberOfIterations = _config.ProNumberOfIterations,
                    ProL1Regularization = _config.ProL1Regularization,
                    ProL2Regularization = _config.ProL2Regularization,
                    Algorithm = _proLearner.Name
                }
            };

            var options = new System.Text.Json.JsonSerializerOptions { WriteIndented = true };
            string json = System.Text.Json.JsonSerializer.Serialize(report, options);
            File.WriteAllText(reportPath, json);
            Console.WriteLine($"  评估报告已保存至: {reportPath}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  评估报告保存失败：{ex.Message}");
        }
    }

    /// <summary>
    /// 把校准出的工作点写成模型旁的阈值清单，让调用端无需手改配置即可采用推荐阈值。
    /// ML.NET 与 ONNX 两个产物各写一份，因为调用端加载的是 ONNX。
    /// </summary>
    private void WriteProThresholdManifest(string modelPath, string? onnxPath, ProTrainingEvaluation evaluation)
    {
        if (!_config.UseCostSensitiveThreshold)
            return;

        var manifest = new ModelThresholdManifest
        {
            ModelMode = nameof(ModelMode.Pro),
            RecommendedThreshold = evaluation.OperatingThreshold,
            SelectionMethod = $"代价敏感（1 误报 = {_config.FalsePositiveCostRatio} 漏报）",
            FalsePositiveCostRatio = _config.FalsePositiveCostRatio,
            FalseNegative = evaluation.TestThresholdMetrics.FalseNegative,
            FalsePositive = evaluation.TestThresholdMetrics.FalsePositive,
            TruePositiveRate = evaluation.TestThresholdMetrics.TruePositiveRate,
            FalsePositiveRate = evaluation.TestThresholdMetrics.FalsePositiveRate,
            EvaluatedSamples = evaluation.TestThresholdMetrics.TruePositive
                + evaluation.TestThresholdMetrics.FalseNegative
                + evaluation.TestThresholdMetrics.FalsePositive
                + evaluation.TestThresholdMetrics.TrueNegative,
            GeneratedAt = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss")
        };

        SaveThresholdManifest(manifest, modelPath, onnxPath);
    }

    /// <summary>
    /// Standard / Flash 的阈值清单。与 Pro 使用同一套代价目标，保证三种模式的工作点口径一致。
    /// </summary>
    private void WriteThresholdManifest(
        string modelPath,
        string? onnxPath,
        ModelMode mode,
        CostThresholdSelection costSelection)
    {
        if (!_config.UseCostSensitiveThreshold)
            return;

        var manifest = new ModelThresholdManifest
        {
            ModelMode = mode.ToString(),
            RecommendedThreshold = costSelection.Threshold,
            SelectionMethod = $"代价敏感（1 误报 = {_config.FalsePositiveCostRatio} 漏报）",
            FalsePositiveCostRatio = _config.FalsePositiveCostRatio,
            FalseNegative = costSelection.Metrics.FalseNegative,
            FalsePositive = costSelection.Metrics.FalsePositive,
            TruePositiveRate = costSelection.Metrics.TruePositiveRate,
            FalsePositiveRate = costSelection.Metrics.FalsePositiveRate,
            EvaluatedSamples = costSelection.Metrics.TruePositive
                + costSelection.Metrics.FalseNegative
                + costSelection.Metrics.FalsePositive
                + costSelection.Metrics.TrueNegative,
            GeneratedAt = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss")
        };

        SaveThresholdManifest(manifest, modelPath, onnxPath);
    }

    private static void SaveThresholdManifest(ModelThresholdManifest manifest, string modelPath, string? onnxPath)
    {
        foreach (string? path in new[] { modelPath, onnxPath })
        {
            if (string.IsNullOrWhiteSpace(path))
                continue;

            try
            {
                manifest.Save(path);
                Console.WriteLine($"  阈值清单已保存至: {ModelThresholdManifest.ResolvePath(path)}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  阈值清单保存失败（{path}）：{ex.Message}");
            }
        }
    }

    public void PredictPro(ITransformer model, FileData fileData)
    {
        if (_lastProTrainingResult == null)
            throw new InvalidOperationException("当前训练器中没有可用的 Pro Stacking 分支模型。");

        float[] proFeatures = ProHybridFeatureExtractor.ExtractFeatures(fileData.FilePath).ToFloatArray();
        var branchScores = new float[FeatureSchema.ProFusionFeatureCount];
        foreach (var branch in _lastProTrainingResult.BranchModels)
        {
            using var engine = _mlContext.Model.CreatePredictionEngine<ProBinaryTrainingData, BinaryModelPrediction>(
                branch.Model,
                inputSchemaDefinition: ProStackingTrainer.CreateSchema(branch.FeatureCount));
            float[] branchFeatures = ProStackingTrainer.ExtractBranch(proFeatures, branch.Branch);
            branchScores[(int)branch.Branch] = engine.Predict(new ProBinaryTrainingData
            {
                Features = branchFeatures
            }).Probability;
        }

        using var fusionEngine = _mlContext.Model.CreatePredictionEngine<ProFusionTrainingData, BinaryModelPrediction>(model);
        var prediction = fusionEngine.Predict(new ProFusionTrainingData
        {
            Features = branchScores,
            Label = fileData.Label
        });
        PrintPrediction(fileData, prediction, _config.ProThreshold);
    }

    private ITransformer TrainCore(List<FileData> fileData, string modelPath, string? onnxPath, bool flash)
    {
        int expectedFeatureCount = flash ? FlashFileFeatures.FeatureCount : FileFeatures.FeatureCount;
        string modeLabel = flash ? "Flash " : "";

        Console.WriteLine($"\n开始训练{modeLabel}模型...");

        var validData = new List<FileData>(fileData.Count);
        var validFeatures = new List<float[]>(fileData.Count);
        int emptyFeaturesCount = 0;
        int wrongSizeCount = 0;

        foreach (var fd in fileData)
        {
            var features = flash ? fd.FlashFeatures.ToFloatArray() : fd.Features.ToFloatArray();
            if (features.Length == 0)
            {
                emptyFeaturesCount++;
            }
            else if (features.Length != expectedFeatureCount)
            {
                wrongSizeCount++;
                if (wrongSizeCount <= 3)
                {
                    Console.WriteLine($"警告：文件 {fd.FilePath} 特征数量为 {features.Length}，期望 {expectedFeatureCount}");
                }
            }
            else
            {
                validData.Add(fd);
                validFeatures.Add(features);
            }
        }

        if (emptyFeaturesCount > 0)
            Console.WriteLine($"警告：{emptyFeaturesCount} 个文件特征为空");
        if (wrongSizeCount > 0)
            Console.WriteLine($"警告：{wrongSizeCount} 个文件特征数量不正确");

        if (validData.Count == 0)
        {
            Console.WriteLine("错误：没有有效的训练数据！");
            return null!;
        }

        Console.WriteLine($"有效训练数据：{validData.Count} 个");

        IDataView fullDataView;
        string labelColumnName;
        List<BinaryTrainingData>? standardRows = null;

        if (flash)
        {
            var trainingData = new List<FlashBinaryTrainingData>(validData.Count);
            for (int i = 0; i < validData.Count; i++)
            {
                trainingData.Add(new FlashBinaryTrainingData
                {
                    Features = validFeatures[i],
                    Label = validData[i].Label
                });
            }
            fullDataView = _mlContext.Data.LoadFromEnumerable(trainingData);
            labelColumnName = nameof(FlashBinaryTrainingData.Label);
        }
        else
        {
            standardRows = new List<BinaryTrainingData>(validData.Count);
            for (int i = 0; i < validData.Count; i++)
            {
                standardRows.Add(new BinaryTrainingData
                {
                    Features = validFeatures[i],
                    Label = validData[i].Label
                });
            }
            fullDataView = _mlContext.Data.LoadFromEnumerable(standardRows);
            labelColumnName = nameof(BinaryTrainingData.Label);
        }

        IDataView trainData;
        IDataView testData;
        if (flash)
        {
            var trainTestSplit = _mlContext.Data.TrainTestSplit(fullDataView, testFraction: 0.2);
            trainData = trainTestSplit.TrainSet;
            testData = trainTestSplit.TestSet;
        }
        else
        {
            var split = StandardTrainingPolicy.CreateStratifiedHoldout(
                standardRows!,
                testFraction: 0.2,
                _config.RandomSeed ?? 43846);
            trainData = _mlContext.Data.LoadFromEnumerable(split.Train);
            testData = _mlContext.Data.LoadFromEnumerable(split.Test);
            Console.WriteLine($"Standard 分层切分：训练 {split.Train.Count}，测试 {split.Test.Count}");
        }

        var pipeline = flash ? BuildFlashPipeline(labelColumnName) : BuildPipeline(labelColumnName);

        Console.WriteLine($"正在训练{modeLabel}LightGBM 模型...");
        var evaluationModel = pipeline.Fit(trainData);

        Console.WriteLine($"正在评估{modeLabel}模型...");
        double threshold = flash ? _config.FlashThreshold : _config.StandardThreshold;
        CostThresholdSelection costSelection = EvaluateModel(
            evaluationModel,
            testData,
            labelColumnName,
            threshold,
            flash ? null : _config.StandardTargetFalsePositiveRate);

        ITransformer model = evaluationModel;
        if (!flash)
        {
            Console.WriteLine("正在使用全部有效样本重训最终 Standard 模型...");
            model = pipeline.Fit(fullDataView);
        }

        Console.WriteLine($"正在保存{modeLabel}ML.NET 模型到：{modelPath}");
        _mlContext.Model.Save(model, fullDataView.Schema, modelPath);
        Console.WriteLine($"{modeLabel}ML.NET 模型保存成功!");

        if (!string.IsNullOrEmpty(onnxPath))
        {
            Console.WriteLine($"\n正在导出{modeLabel}ONNX 模型到：{onnxPath}");
            try
            {
                ExportToOnnx(model, fullDataView, onnxPath);
                Console.WriteLine($"{modeLabel}ONNX 模型导出成功!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"{modeLabel}ONNX 导出失败：{ex.Message}");
                Console.WriteLine("注意：并非所有 ML.NET 模型都支持导出为 ONNX 格式。");
            }
        }

        WriteThresholdManifest(
            modelPath,
            onnxPath,
            flash ? ModelMode.Flash : ModelMode.Standard,
            costSelection);

        return model;
    }

    private void ExportToOnnx(ITransformer model, IDataView dataView, string onnxPath)
    {
        using var stream = File.Create(onnxPath);
        _mlContext.Model.ConvertToOnnx(model, dataView, stream);
    }

    private IEstimator<ITransformer> BuildPipeline(string labelColumnName)
    {
        var options = new Microsoft.ML.Trainers.LightGbm.LightGbmBinaryTrainer.Options
        {
            LabelColumnName = labelColumnName,
            FeatureColumnName = "Features",
            LearningRate = _config.LearningRate,
            NumberOfLeaves = _config.NumberOfLeaves,
            MinimumExampleCountPerLeaf = _config.MinimumExampleCountPerLeaf,
            NumberOfIterations = _config.NumberOfIterations,
            NumberOfThreads = TrainingHardware.ResolveTrainingThreadCount(_config.TrainingThreadCount),
            ForceColumnWise = _config.ForceColumnWiseHistogram,
            WeightOfPositiveExamples = _config.ResolveWeightOfPositiveExamples(),
            Deterministic = true,
            Seed = _config.RandomSeed,
            Booster = new Microsoft.ML.Trainers.LightGbm.GradientBooster.Options
            {
                L1Regularization = _config.StandardL1Regularization,
                L2Regularization = _config.StandardL2Regularization,
                MaximumTreeDepth = _config.StandardMaximumTreeDepth,
                FeatureFraction = _config.StandardFeatureFraction,
                SubsampleFraction = _config.StandardSubsampleFraction,
                SubsampleFrequency = 1
            }
        };

        return _mlContext.BinaryClassification.Trainers.LightGbm(options);
    }

    private IEstimator<ITransformer> BuildFlashPipeline(string labelColumnName)
    {
        var options = new Microsoft.ML.Trainers.LightGbm.LightGbmBinaryTrainer.Options
        {
            LabelColumnName = labelColumnName,
            FeatureColumnName = "Features",
            LearningRate = _config.FlashLearningRate,
            NumberOfLeaves = _config.FlashNumberOfLeaves,
            MinimumExampleCountPerLeaf = _config.FlashMinimumExampleCountPerLeaf,
            NumberOfIterations = _config.FlashNumberOfIterations,
            NumberOfThreads = TrainingHardware.ResolveTrainingThreadCount(_config.TrainingThreadCount),
            ForceColumnWise = _config.ForceColumnWiseHistogram,
            WeightOfPositiveExamples = _config.ResolveWeightOfPositiveExamples(),
            Deterministic = true,
            Seed = _config.RandomSeed,
            Booster = new Microsoft.ML.Trainers.LightGbm.GradientBooster.Options
            {
                L1Regularization = _config.FlashL1Regularization,
                L2Regularization = _config.FlashL2Regularization,
                MaximumTreeDepth = _config.FlashMaximumTreeDepth
            }
        };

        return _mlContext.BinaryClassification.Trainers.LightGbm(options);
    }

    private CostThresholdSelection EvaluateModel(
        ITransformer model,
        IDataView testData,
        string labelColumnName,
        double threshold,
        double? maximumFalsePositiveRate = null)
    {
        var predictions = model.Transform(testData);

        var metrics = _mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: labelColumnName);
        var rows = _mlContext.Data.CreateEnumerable<ThresholdEvaluationRow>(predictions, reuseRowObject: false).ToList();
        var sweep = new ThresholdSweep(rows);
        var thresholdMetrics = sweep.Compute(threshold);

        Console.WriteLine("\n=== 模型评估结果 ===");
        Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P4}");
        Console.WriteLine($"AUPRC: {metrics.AreaUnderPrecisionRecallCurve:P4}");
        Console.WriteLine($"判毒阈值: {threshold:F2}%");
        Console.WriteLine($"准确率 (Accuracy): {thresholdMetrics.Accuracy:P4}");
        Console.WriteLine($"检出率 (TPR): {thresholdMetrics.TruePositiveRate:P4}");
        Console.WriteLine($"误报率 (FPR): {thresholdMetrics.FalsePositiveRate:P4}");
        Console.WriteLine($"F1 分数: {thresholdMetrics.F1Score:P4}");

        Console.WriteLine("\n混淆矩阵:");
        Console.WriteLine($"TP: {thresholdMetrics.TruePositive}, FN: {thresholdMetrics.FalseNegative}");
        Console.WriteLine($"FP: {thresholdMetrics.FalsePositive}, TN: {thresholdMetrics.TrueNegative}");

        if (maximumFalsePositiveRate.HasValue)
        {
            var constrained = StandardTrainingPolicy.FindThresholdAtMaximumFalsePositiveRate(
                sweep,
                maximumFalsePositiveRate.Value);
            Console.WriteLine($"\n低误报校准（FPR <= {maximumFalsePositiveRate.Value:P2}）:");
            Console.WriteLine($"推荐阈值: {constrained.Threshold:F1}%");
            Console.WriteLine($"检出率 (TPR): {constrained.Metrics.TruePositiveRate:P4}");
            Console.WriteLine($"误报率 (FPR): {constrained.Metrics.FalsePositiveRate:P4}");
        }

        CostThresholdSelection costSelection = CostSensitiveThreshold.FindMinimumCostThreshold(
            sweep,
            _config.FalsePositiveCostRatio);
        Console.WriteLine($"\n代价敏感校准（1 误报 = {_config.FalsePositiveCostRatio} 漏报）:");
        Console.WriteLine($"推荐阈值: {costSelection.Threshold:F2}%");
        Console.WriteLine($"检出率 (TPR): {costSelection.Metrics.TruePositiveRate:P4}");
        Console.WriteLine($"误报率 (FPR): {costSelection.Metrics.FalsePositiveRate:P4}");
        Console.WriteLine($"FN: {costSelection.Metrics.FalseNegative}, FP: {costSelection.Metrics.FalsePositive}");
        Console.WriteLine(
            $"加权代价: {costSelection.Cost:F1}（固定阈值 {threshold:F2}% 为 " +
            $"{CostSensitiveThreshold.ComputeCost(thresholdMetrics, _config.FalsePositiveCostRatio):F1}）");

        return costSelection;
    }

    internal static ThresholdMetrics ComputeThresholdMetrics(List<ThresholdEvaluationRow> rows, double threshold)
    {
        return new ThresholdSweep(rows).Compute(threshold);
    }

    internal static ThresholdMetrics CreateThresholdMetrics(
        long truePositive,
        long falseNegative,
        long falsePositive,
        long trueNegative)
    {
        long total = truePositive + falseNegative + falsePositive + trueNegative;
        double accuracy = total > 0 ? (double)(truePositive + trueNegative) / total : 0;
        double precision = truePositive + falsePositive > 0 ? (double)truePositive / (truePositive + falsePositive) : 0;
        double truePositiveRate = truePositive + falseNegative > 0 ? (double)truePositive / (truePositive + falseNegative) : 0;
        double falsePositiveRate = falsePositive + trueNegative > 0 ? (double)falsePositive / (falsePositive + trueNegative) : 0;
        double f1Score = precision + truePositiveRate > 0 ? 2 * precision * truePositiveRate / (precision + truePositiveRate) : 0;

        return new ThresholdMetrics(
            truePositive,
            falseNegative,
            falsePositive,
            trueNegative,
            accuracy,
            truePositiveRate,
            falsePositiveRate,
            f1Score);
    }

    internal static (double threshold, ThresholdMetrics metrics) FindBestThreshold(List<ThresholdEvaluationRow> rows)
    {
        return FindBestThreshold(new ThresholdSweep(rows));
    }

    internal static (double threshold, ThresholdMetrics metrics) FindBestThreshold(ThresholdSweep sweep)
    {
        double bestThreshold = 50;
        ThresholdMetrics? bestMetrics = null;

        for (double threshold = 50; threshold <= 99.9; threshold += 0.1)
        {
            var metrics = sweep.Compute(threshold);
            if (bestMetrics == null ||
                metrics.F1Score > bestMetrics.F1Score + 0.000001 ||
                (Math.Abs(metrics.F1Score - bestMetrics.F1Score) <= 0.000001 &&
                 metrics.FalsePositiveRate < bestMetrics.FalsePositiveRate) ||
                (Math.Abs(metrics.F1Score - bestMetrics.F1Score) <= 0.000001 &&
                 Math.Abs(metrics.FalsePositiveRate - bestMetrics.FalsePositiveRate) <= 0.000001 &&
                 metrics.TruePositiveRate > bestMetrics.TruePositiveRate))
            {
                bestThreshold = threshold;
                bestMetrics = metrics;
            }
        }

        return (bestThreshold, bestMetrics ?? sweep.Compute(bestThreshold));
    }

    public void Predict(ITransformer model, FileData fileData)
    {
        var predictionEngine = _mlContext.Model.CreatePredictionEngine<BinaryTrainingData, BinaryModelPrediction>(model);
        var prediction = predictionEngine.Predict(new BinaryTrainingData
        {
            Features = fileData.Features.ToFloatArray(),
            Label = fileData.Label
        });
        PrintPrediction(fileData, prediction, _config.StandardThreshold);
    }

    public void PredictFlash(ITransformer model, FileData fileData)
    {
        var predictionEngine = _mlContext.Model.CreatePredictionEngine<FlashBinaryTrainingData, BinaryModelPrediction>(model);
        var prediction = predictionEngine.Predict(new FlashBinaryTrainingData
        {
            Features = fileData.FlashFeatures.ToFloatArray(),
            Label = fileData.Label
        });
        PrintPrediction(fileData, prediction, _config.FlashThreshold);
    }

    private static void PrintPrediction(FileData fileData, BinaryModelPrediction prediction, double threshold)
    {
        bool thresholdedLabel = prediction.Probability * 100 >= threshold;

        Console.WriteLine($"\n文件：{Path.GetFileName(fileData.FilePath)}");
        Console.WriteLine($"实际标签：{(fileData.Label ? "黑文件" : "白文件")}");
        Console.WriteLine($"预测标签：{(thresholdedLabel ? "黑文件" : "白文件")}");
        Console.WriteLine($"预测概率：{prediction.Probability:P4}");
        Console.WriteLine($"预测分数：{prediction.Score:F4}");
        Console.WriteLine($"判毒阈值：{threshold:F2}%");
    }
}

public class BinaryTrainingData
{
    [VectorType(FileFeatures.FeatureCount)]
    public float[] Features { get; set; } = Array.Empty<float>();

    public bool Label { get; set; }
}

public class ThresholdEvaluationRow
{
    public bool Label { get; set; }
    public float Probability { get; set; }
}

public record ThresholdMetrics(
    long TruePositive,
    long FalseNegative,
    long FalsePositive,
    long TrueNegative,
    double Accuracy,
    double TruePositiveRate,
    double FalsePositiveRate,
    double F1Score);

public record ProTrainingEvaluation(
    double TestAuc,
    double TestAuprc,
    double TrainAuc,
    double AucGap,
    ThresholdMetrics TestThresholdMetrics,
    ThresholdMetrics TestBestThresholdMetrics,
    double BestThreshold,
    int TotalSamples,
    int BlackSamples,
    int WhiteSamples,
    int FeatureCount,
    double OperatingThreshold,
    CostThresholdSelection CostThreshold,
    ThresholdMetrics ConfiguredThresholdMetrics);

public class FlashBinaryTrainingData
{
    [VectorType(FlashFileFeatures.FeatureCount)]
    public float[] Features { get; set; } = Array.Empty<float>();

    public bool Label { get; set; }
}

public class ProBinaryTrainingData
{
    public float[] Features { get; set; } = Array.Empty<float>();

    public bool Label { get; set; }
}

public class BinaryModelPrediction
{
    [ColumnName("PredictedLabel")]
    public bool PredictedLabel { get; set; }

    [ColumnName("Score")]
    public float Score { get; set; }

    [ColumnName("Probability")]
    public float Probability { get; set; }
}
