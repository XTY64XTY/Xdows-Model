using Microsoft.ML;
using Microsoft.ML.Data;
using Xdows_Model_Config;

namespace Xdows_Model_Maker;

public class ModelTrainer
{
    private readonly MLContext _mlContext;
    private readonly TrainingConfig _config;
    private volatile bool _proTrainingCancelled;

    public ModelTrainer(TrainingConfig config)
    {
        _config = config;
        _mlContext = new MLContext(seed: config.RandomSeed);
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

    public (ITransformer model, int optimalBytesPerSection) TrainProModel(List<FileData> fileData, string modelPath, string? onnxPath = null)
    {
        return TrainProWithProgressiveExpansion(fileData, modelPath, onnxPath);
    }

    public (ITransformer model, int optimalBytesPerSection) TrainProModel(List<FileData> fileData)
    {
        return TrainProModel(fileData, _config.ProModelPath, _config.ProOnnxPath);
    }

    private (ITransformer model, int optimalBytesPerSection) TrainProWithProgressiveExpansion(List<FileData> fileData, string modelPath, string? onnxPath)
    {
        Console.WriteLine("\n开始训练 Pro 混合特征模型...");
        Console.WriteLine($"特征组成：Standard {FileFeatures.FeatureCount} 维 + Flash {FlashFileFeatures.FeatureCount} 维 + Raw {ProHybridFileFeatures.RawFeatureCount} 维 + PE结构 {ProHybridFileFeatures.StructuralFeatureCount} 维");
        Console.WriteLine($"Pro 总特征维度：{ProHybridFileFeatures.FeatureCount}\n");

        if (_proTrainingCancelled)
        {
            Console.WriteLine("Pro 训练已取消。");
            return (null!, ProHybridFileFeatures.RawBytesPerSection);
        }

        var (model, auc, dataView) = TrainProStep(fileData, ProHybridFileFeatures.RawBytesPerSection);

        if (model == null || dataView == null)
        {
            Console.WriteLine("警告：Pro 混合特征模型未产生有效模型。");
            return (null!, ProHybridFileFeatures.RawBytesPerSection);
        }

        Console.WriteLine($"\n正在保存最优 Pro 模型...");
        _mlContext.Model.Save(model, dataView.Schema, modelPath);
        Console.WriteLine($"Pro ML.NET 模型已保存至: {modelPath}");

        if (!string.IsNullOrEmpty(onnxPath))
        {
            try
            {
                ExportToOnnx(model, dataView, onnxPath);
                Console.WriteLine($"Pro ONNX 模型已保存至: {onnxPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Pro ONNX 导出失败：{ex.Message}");
            }
        }

        Console.WriteLine($"\n=== Pro 混合特征模型训练完成 ===");
        Console.WriteLine($"Raw 每段字节数：{ProHybridFileFeatures.RawBytesPerSection}");
        Console.WriteLine($"最终特征维度：{ProHybridFileFeatures.FeatureCount}");
        Console.WriteLine($"AUC：{auc:P4}");

        return (model, ProHybridFileFeatures.RawBytesPerSection);
    }

    private (ITransformer model, double auc, IDataView dataView) TrainProStep(List<FileData> fileData, int bytesPerSection)
    {
        int featureCount = ProHybridFileFeatures.FeatureCount;

        var validData = new List<FileData>();
        var validFeatures = new List<float[]>();
        int emptyFeaturesCount = 0;

        foreach (var fd in fileData)
        {
            try
            {
                var proFeatures = ProHybridFeatureExtractor.ExtractFeatures(fd.FilePath);
                var floatArray = proFeatures.ToFloatArray();
                if (floatArray.Length != featureCount)
                {
                    emptyFeaturesCount++;
                }
                else
                {
                    validData.Add(fd);
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
            Console.WriteLine($"  警告：{emptyFeaturesCount} 个文件特征提取失败");
            Console.WriteLine($"  提示：可使用选项7「清洗非PE文件（含Pro兼容性检查）」功能清理不兼容的文件");
        }

        if (validData.Count == 0)
        {
            Console.WriteLine("  错误：没有有效的训练数据！");
            return (null!, 0, null!);
        }

        Console.WriteLine($"  有效训练数据：{validData.Count} 个");

        var blackCount = validData.Count(d => d.Label);
        var whiteCount = validData.Count(d => !d.Label);
        Console.WriteLine($"  黑文件：{blackCount}，白文件：{whiteCount}");

        if (blackCount == 0 || whiteCount == 0)
        {
            Console.WriteLine("  错误：有效数据中只有一类标签，无法训练！");
            return (null!, 0, null!);
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

        var pipeline = BuildProPipeline(featureCount);
        var model = pipeline.Fit(trainTestSplit.TrainSet);

        double auc;
        try
        {
            var predictions = model.Transform(trainTestSplit.TestSet);
            var metrics = _mlContext.BinaryClassification.Evaluate(predictions);
            auc = metrics.AreaUnderRocCurve;
            var rows = _mlContext.Data.CreateEnumerable<ThresholdEvaluationRow>(predictions, reuseRowObject: false).ToList();
            var thresholdMetrics = ComputeThresholdMetrics(rows, _config.ProThreshold);
            var (bestThreshold, bestThresholdMetrics) = FindBestThreshold(rows);
            Console.WriteLine($"  阈值: {_config.ProThreshold:F2}%");
            Console.WriteLine($"  准确率: {thresholdMetrics.Accuracy:P4}，AUC: {auc:P4}，F1: {thresholdMetrics.F1Score:P4}");
            Console.WriteLine($"  检出率: {thresholdMetrics.TruePositiveRate:P4}，误报率: {thresholdMetrics.FalsePositiveRate:P4}，BrewTotal 代理分数: {thresholdMetrics.BrewTotalProxyScore:F2}");
            Console.WriteLine($"  最优代理阈值: {bestThreshold:F2}%");
            Console.WriteLine($"  最优代理检出率: {bestThresholdMetrics.TruePositiveRate:P4}，误报率: {bestThresholdMetrics.FalsePositiveRate:P4}，BrewTotal 代理分数: {bestThresholdMetrics.BrewTotalProxyScore:F2}");
        }
        catch (ArgumentOutOfRangeException)
        {
            auc = 0;
            Console.WriteLine("  警告：AUC 无法计算（测试集类别不平衡），使用 AUC=0");
        }

        return (model, auc, fullDataView);
    }

    private IEstimator<ITransformer> BuildProPipeline(int featureCount)
    {
        var options = new Microsoft.ML.Trainers.LightGbm.LightGbmBinaryTrainer.Options
        {
            LabelColumnName = "Label",
            FeatureColumnName = "Features",
            LearningRate = _config.ProLearningRate,
            NumberOfLeaves = _config.ProNumberOfLeaves,
            MinimumExampleCountPerLeaf = _config.ProMinimumExampleCountPerLeaf,
            NumberOfIterations = _config.ProNumberOfIterations,
            Booster = new Microsoft.ML.Trainers.LightGbm.GradientBooster.Options
            {
                L1Regularization = _config.ProL1Regularization,
                L2Regularization = _config.ProL2Regularization
            }
        };

        return _mlContext.Transforms.NormalizeMinMax("Features", "Features")
            .Append(_mlContext.BinaryClassification.Trainers.LightGbm(options));
    }

    public void PredictPro(ITransformer model, FileData fileData, int bytesPerSection)
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
        Console.WriteLine($"BrewTotal 代理分数: {thresholdMetrics.BrewTotalProxyScore:F2}");

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

        double rawScore = truePositive * 10 - falseNegative * 7 - falsePositive * 10;
        double brewTotalProxyScore = rawScore * truePositiveRate - Math.Abs(rawScore) * falsePositiveRate * 1.27;

        return new ThresholdMetrics(
            truePositive,
            falseNegative,
            falsePositive,
            trueNegative,
            accuracy,
            truePositiveRate,
            falsePositiveRate,
            f1Score,
            brewTotalProxyScore);
    }

    private static (double threshold, ThresholdMetrics metrics) FindBestThreshold(List<ThresholdEvaluationRow> rows)
    {
        double bestThreshold = 50;
        ThresholdMetrics? bestMetrics = null;

        for (double threshold = 50; threshold <= 99.9; threshold += 0.1)
        {
            var metrics = ComputeThresholdMetrics(rows, threshold);
            if (bestMetrics == null || metrics.BrewTotalProxyScore > bestMetrics.BrewTotalProxyScore)
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
    double F1Score,
    double BrewTotalProxyScore);

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
