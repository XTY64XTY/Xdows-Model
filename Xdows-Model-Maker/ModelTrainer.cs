using Microsoft.ML;
using Microsoft.ML.Data;
using Xdows_Model_Config;

namespace Xdows_Model_Maker;

public class ModelTrainer
{
    private readonly MLContext _mlContext;
    private readonly TrainingConfig _config;
    private readonly IProLearner _proLearner;
    private volatile bool _proTrainingCancelled;

    public ModelTrainer(TrainingConfig config)
    {
        _config = config;
        _mlContext = new MLContext(seed: config.RandomSeed);
        _proLearner = ProLearnerFactory.Create(config.ProLearner);
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
        Console.WriteLine("\n开始训练 Pro 混合特征模型...");
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

        var (model, evaluation, fullDataView) = TrainProStep(featureCache);
        if (model == null || evaluation == null || fullDataView == null)
        {
            Console.WriteLine("警告：Pro 混合特征模型未产生有效模型。");
            return null;
        }

        Console.WriteLine($"\n正在保存 Pro 模型...");
        _mlContext.Model.Save(model, fullDataView.Schema, modelPath);
        Console.WriteLine($"Pro ML.NET 模型已保存至: {modelPath}");

        if (!string.IsNullOrEmpty(onnxPath))
        {
            try
            {
                ExportToOnnx(model, fullDataView, onnxPath);
                Console.WriteLine($"Pro ONNX 模型已保存至: {onnxPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Pro ONNX 导出失败：{ex.Message}");
            }
        }

        WriteProEvaluationReport(modelPath, evaluation);

        Console.WriteLine($"\n=== Pro 混合特征模型训练完成 ===");
        Console.WriteLine($"最终特征维度：{ProHybridFileFeatures.FeatureCount}");
        Console.WriteLine($"测试集 AUC：{evaluation.TestAuc:P4}");
        Console.WriteLine($"训练集 AUC：{evaluation.TrainAuc:P4}");
        Console.WriteLine($"AUC Gap：{evaluation.AucGap:P4}");

        return model;
    }

    private (ITransformer? model, ProTrainingEvaluation? evaluation, IDataView? dataView) TrainProStep(ProFeatureCache featureCache)
    {
        int featureCount = ProHybridFileFeatures.FeatureCount;

        var validData = new List<ProFeatureCacheEntry>();
        var validFeatures = new List<float[]>();
        int emptyFeaturesCount = 0;

        foreach (var entry in featureCache.Entries)
        {
            try
            {
                var floatArray = entry.CreateFeatures();
                if (floatArray.Length != featureCount)
                {
                    emptyFeaturesCount++;
                }
                else
                {
                    validData.Add(entry);
                    validFeatures.Add(floatArray);
                }
            }
            catch
            {
                emptyFeaturesCount++;
            }
        }

        if (emptyFeaturesCount > 0)
        {
            Console.WriteLine($"  警告：{emptyFeaturesCount} 个缓存样本组装失败");
            Console.WriteLine($"  提示：可使用选项7「清洗非PE文件（含Pro兼容性检查）」功能清理不兼容的文件");
        }

        if (featureCache.FailedCount > 0)
            Console.WriteLine($"  Pro 缓存阶段已跳过 {featureCache.FailedCount} 个不兼容文件");

        if (validData.Count == 0)
        {
            Console.WriteLine("  错误：没有有效的训练数据！");
            return (null, default, null);
        }

        Console.WriteLine($"  有效训练数据：{validData.Count} 个");

        var blackCount = validData.Count(d => d.Label);
        var whiteCount = validData.Count(d => !d.Label);
        Console.WriteLine($"  黑文件：{blackCount}，白文件：{whiteCount}");

        if (blackCount == 0 || whiteCount == 0)
        {
            Console.WriteLine("  错误：有效数据中只有一类标签，无法训练！");
            return (null, default, null);
        }

        var trainingData = validData.Select((fd, idx) => new ProBinaryTrainingData(featureCount)
        {
            Features = validFeatures[idx],
            Label = fd.Label
        }).ToList();

        var schemaDef = SchemaDefinition.Create(typeof(ProBinaryTrainingData));
        schemaDef["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, featureCount);

        var fullDataView = _mlContext.Data.LoadFromEnumerable(trainingData, schemaDef);
        var trainTestSplit = _mlContext.Data.TrainTestSplit(fullDataView, testFraction: 0.2);

        var pipeline = _proLearner.BuildPipeline(_mlContext, _config, featureCount);
        Console.WriteLine($"  正在训练 Pro {_proLearner.Name} 模型...");
        var model = pipeline.Fit(trainTestSplit.TrainSet);

        // 测试集评估
        double testAuc, testAuprc;
        ThresholdMetrics testThresholdMetrics, testBestThresholdMetrics;
        double bestThreshold;
        try
        {
            var testPredictions = model.Transform(trainTestSplit.TestSet);
            var testMetrics = _mlContext.BinaryClassification.Evaluate(testPredictions);
            testAuc = testMetrics.AreaUnderRocCurve;
            testAuprc = testMetrics.AreaUnderPrecisionRecallCurve;
            var testRows = _mlContext.Data.CreateEnumerable<ThresholdEvaluationRow>(testPredictions, reuseRowObject: false).ToList();
            testThresholdMetrics = ComputeThresholdMetrics(testRows, _config.ProThreshold);
            (bestThreshold, testBestThresholdMetrics) = FindBestThreshold(testRows);
        }
        catch (ArgumentOutOfRangeException)
        {
            testAuc = double.NaN;
            testAuprc = 0;
            testThresholdMetrics = new ThresholdMetrics(0, 0, 0, 0, 0, 0, 0, 0);
            testBestThresholdMetrics = testThresholdMetrics;
            bestThreshold = _config.ProThreshold;
            Console.WriteLine("  警告：测试集评估指标无法计算（类别不平衡）");
        }

        // 训练集评估（用于 train-test gap）
        double trainAuc;
        try
        {
            var trainPredictions = model.Transform(trainTestSplit.TrainSet);
            var trainMetrics = _mlContext.BinaryClassification.Evaluate(trainPredictions);
            trainAuc = trainMetrics.AreaUnderRocCurve;
        }
        catch (ArgumentOutOfRangeException)
        {
            trainAuc = double.NaN;
        }

        double aucGap = double.IsNaN(trainAuc) || double.IsNaN(testAuc) ? double.NaN : trainAuc - testAuc;

        Console.WriteLine($"  阈值: {_config.ProThreshold:F2}%");
        Console.WriteLine($"  === 测试集评估 ===");
        Console.WriteLine($"  准确率: {testThresholdMetrics.Accuracy:P4}，AUC: {testAuc:P4}，AUPRC: {testAuprc:P4}，F1: {testThresholdMetrics.F1Score:P4}");
        Console.WriteLine($"  检出率: {testThresholdMetrics.TruePositiveRate:P4}，误报率: {testThresholdMetrics.FalsePositiveRate:P4}");
        Console.WriteLine($"  混淆矩阵: TP={testThresholdMetrics.TruePositive}, FN={testThresholdMetrics.FalseNegative}, FP={testThresholdMetrics.FalsePositive}, TN={testThresholdMetrics.TrueNegative}");
        Console.WriteLine($"  最优 F1 阈值: {bestThreshold:F2}%");
        Console.WriteLine($"  最优 F1: {testBestThresholdMetrics.F1Score:P4}，检出率: {testBestThresholdMetrics.TruePositiveRate:P4}，误报率: {testBestThresholdMetrics.FalsePositiveRate:P4}");
        Console.WriteLine($"  === 训练集评估 ===");
        Console.WriteLine($"  训练集 AUC: {trainAuc:P4}");
        Console.WriteLine($"  === Train-Test Gap ===");
        Console.WriteLine($"  AUC Gap: {aucGap:P4}{(double.IsNaN(aucGap) ? "  ⚠ 评估失败" : aucGap > 0.05 ? "  ⚠ 可能过拟合" : aucGap < -0.02 ? "  ⚠ 异常" : "")}");

        var evaluation = new ProTrainingEvaluation(
            testAuc, testAuprc, trainAuc, aucGap,
            testThresholdMetrics, testBestThresholdMetrics, bestThreshold,
            validData.Count, blackCount, whiteCount, featureCount);

        return (model, evaluation, fullDataView);
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
                FeatureCount = evaluation.FeatureCount,
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
                Config = new
                {
                    ProThreshold = _config.ProThreshold,
                    ProLearningRate = _config.ProLearningRate,
                    ProNumberOfLeaves = _config.ProNumberOfLeaves,
                    ProMinimumExampleCountPerLeaf = _config.ProMinimumExampleCountPerLeaf,
                    ProNumberOfIterations = _config.ProNumberOfIterations,
                    ProL1Regularization = _config.ProL1Regularization,
                    ProL2Regularization = _config.ProL2Regularization,
                    ProLearner = _config.ProLearner
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

    public void PredictPro(ITransformer model, FileData fileData)
    {
        int featureCount = ProHybridFileFeatures.FeatureCount;
        var proFeatures = ProHybridFeatureExtractor.ExtractFeatures(fileData.FilePath);
        var schemaDef = SchemaDefinition.Create(typeof(ProBinaryTrainingData));
        schemaDef["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, featureCount);
        var predictionEngine = _mlContext.Model.CreatePredictionEngine<ProBinaryTrainingData, BinaryModelPrediction>(model, inputSchemaDefinition: schemaDef);
        var prediction = predictionEngine.Predict(new ProBinaryTrainingData(featureCount)
        {
            Features = proFeatures.ToFloatArray(),
            Label = fileData.Label
        });
        PrintPrediction(fileData, prediction, _config.ProThreshold);
    }

    private ITransformer TrainCore(List<FileData> fileData, string modelPath, string? onnxPath, bool flash)
    {
        int expectedFeatureCount = flash ? FlashFileFeatures.FeatureCount : FileFeatures.FeatureCount;
        string modeLabel = flash ? "Flash " : "";

        Console.WriteLine($"\n开始训练{modeLabel}模型...");

        var validData = new List<FileData>();
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

        if (flash)
        {
            var trainingData = validData.Select(fd => new FlashBinaryTrainingData
            {
                Features = fd.FlashFeatures.ToFloatArray(),
                Label = fd.Label
            }).ToList();
            fullDataView = _mlContext.Data.LoadFromEnumerable(trainingData);
            labelColumnName = nameof(FlashBinaryTrainingData.Label);
        }
        else
        {
            var trainingData = validData.Select(fd => new BinaryTrainingData
            {
                Features = fd.Features.ToFloatArray(),
                Label = fd.Label
            }).ToList();
            fullDataView = _mlContext.Data.LoadFromEnumerable(trainingData);
            labelColumnName = nameof(BinaryTrainingData.Label);
        }

        var trainTestSplit = _mlContext.Data.TrainTestSplit(fullDataView, testFraction: 0.2);
        var trainData = trainTestSplit.TrainSet;
        var testData = trainTestSplit.TestSet;

        var pipeline = flash ? BuildFlashPipeline(labelColumnName) : BuildPipeline(labelColumnName);

        Console.WriteLine($"正在训练{modeLabel}LightGBM 模型...");
        var model = pipeline.Fit(trainData);

        Console.WriteLine($"正在评估{modeLabel}模型...");
        double threshold = flash ? _config.FlashThreshold : _config.StandardThreshold;
        EvaluateModel(model, testData, labelColumnName, threshold);

        Console.WriteLine($"正在保存{modeLabel}ML.NET 模型到：{modelPath}");
        _mlContext.Model.Save(model, trainData.Schema, modelPath);
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
            Booster = new Microsoft.ML.Trainers.LightGbm.GradientBooster.Options
            {
                L1Regularization = _config.StandardL1Regularization,
                L2Regularization = _config.StandardL2Regularization
            }
        };

        return _mlContext.Transforms.Concatenate("Features", "Features")
            .Append(_mlContext.BinaryClassification.Trainers.LightGbm(options));
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
            Booster = new Microsoft.ML.Trainers.LightGbm.GradientBooster.Options
            {
                L1Regularization = _config.FlashL1Regularization,
                L2Regularization = _config.FlashL2Regularization
            }
        };

        return _mlContext.Transforms.Concatenate("Features", "Features")
            .Append(_mlContext.BinaryClassification.Trainers.LightGbm(options));
    }

    private void EvaluateModel(ITransformer model, IDataView testData, string labelColumnName, double threshold)
    {
        var predictions = model.Transform(testData);

        var metrics = _mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: labelColumnName);
        var rows = _mlContext.Data.CreateEnumerable<ThresholdEvaluationRow>(predictions, reuseRowObject: false).ToList();
        var thresholdMetrics = ComputeThresholdMetrics(rows, threshold);

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
    }

    private static ThresholdMetrics ComputeThresholdMetrics(List<ThresholdEvaluationRow> rows, double threshold)
    {
        long truePositive = 0;
        long falseNegative = 0;
        long falsePositive = 0;
        long trueNegative = 0;

        foreach (var row in rows)
        {
            bool predictedPositive = row.Probability * 100 >= threshold;
            if (row.Label && predictedPositive)
                truePositive++;
            else if (row.Label)
                falseNegative++;
            else if (predictedPositive)
                falsePositive++;
            else
                trueNegative++;
        }

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

    private static (double threshold, ThresholdMetrics metrics) FindBestThreshold(List<ThresholdEvaluationRow> rows)
    {
        double bestThreshold = 50;
        ThresholdMetrics? bestMetrics = null;

        for (double threshold = 50; threshold <= 99.9; threshold += 0.1)
        {
            var metrics = ComputeThresholdMetrics(rows, threshold);
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

        return (bestThreshold, bestMetrics ?? ComputeThresholdMetrics(rows, bestThreshold));
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
    int FeatureCount);

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

    public ProBinaryTrainingData(int featureCount)
    {
        Features = new float[featureCount];
    }

    public ProBinaryTrainingData() { }
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
