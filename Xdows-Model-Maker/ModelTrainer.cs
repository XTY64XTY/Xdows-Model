using Microsoft.ML;
using Microsoft.ML.Data;

namespace Xdows_Model_Maker;

public class ModelTrainer
{
    private readonly MLContext _mlContext;
    private readonly TrainingConfig _config;

    public ModelTrainer(TrainingConfig config)
    {
        _config = config;
        _mlContext = new MLContext(seed: config.RandomSeed);
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
        Console.WriteLine("\n开始训练 Pro 模型（渐进式扩展）...");

        int currentBytesPerSection = _config.ProExpansionStartBytesPerSection;
        int maxBytesPerSection = _config.ProExpansionMaxBytesPerSection;
        double aucThreshold = _config.ProExpansionAucThreshold;
        int patience = _config.ProExpansionPatience;

        ITransformer bestModel = null!;
        int bestBytesPerSection = currentBytesPerSection;
        double bestAuc = 0;
        int noImprovementCount = 0;
        IDataView? bestDataView = null;

        while (currentBytesPerSection <= maxBytesPerSection)
        {
            int featureCount = ProFileFeatures.SectionCount * currentBytesPerSection;
            Console.WriteLine($"\n--- Pro 渐进式扩展：每段 {currentBytesPerSection} 字节，特征维度 {featureCount} ---");

            var (model, auc, dataView) = TrainProStep(fileData, currentBytesPerSection);

            if (auc > bestAuc + aucThreshold)
            {
                bestAuc = auc;
                bestModel = model;
                bestBytesPerSection = currentBytesPerSection;
                bestDataView = dataView;
                noImprovementCount = 0;
                Console.WriteLine($"  AUC 提升：{auc:P4}（最佳）");
            }
            else
            {
                noImprovementCount++;
                Console.WriteLine($"  AUC 未显著提升：{auc:P4}（连续 {noImprovementCount}/{patience} 步）");

                if (noImprovementCount >= patience)
                {
                    Console.WriteLine($"  连续 {patience} 步无显著提升，停止扩展。");
                    break;
                }
            }

            currentBytesPerSection *= _config.ProExpansionFactor;
        }

        if (bestModel == null || bestDataView == null)
        {
            Console.WriteLine("警告：Pro 模型渐进式扩展未产生有效模型。");
            return (null!, bestBytesPerSection);
        }

        Console.WriteLine($"\n正在保存最优 Pro 模型...");
        _mlContext.Model.Save(bestModel, bestDataView.Schema, modelPath);
        Console.WriteLine($"Pro ML.NET 模型已保存至: {modelPath}");

        if (!string.IsNullOrEmpty(onnxPath))
        {
            try
            {
                ExportToOnnx(bestModel, bestDataView, onnxPath);
                Console.WriteLine($"Pro ONNX 模型已保存至: {onnxPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Pro ONNX 导出失败：{ex.Message}");
            }
        }

        Console.WriteLine($"\n=== Pro 模型渐进式扩展完成 ===");
        Console.WriteLine($"最优每段字节数：{bestBytesPerSection}");
        Console.WriteLine($"最优特征维度：{ProFileFeatures.SectionCount * bestBytesPerSection}");
        Console.WriteLine($"最优 AUC：{bestAuc:P4}");

        return (bestModel, bestBytesPerSection);
    }

    private (ITransformer model, double auc, IDataView dataView) TrainProStep(List<FileData> fileData, int bytesPerSection)
    {
        int featureCount = ProFileFeatures.SectionCount * bytesPerSection;

        var validData = new List<FileData>();
        var validFeatures = new List<float[]>();
        int emptyFeaturesCount = 0;

        foreach (var fd in fileData)
        {
            try
            {
                var proFeatures = ProFeatureExtractor.ExtractFeatures(fd.FilePath, bytesPerSection);
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
            Console.WriteLine($"  准确率: {metrics.Accuracy:P4}，AUC: {auc:P4}，F1: {metrics.F1Score:P4}");
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
            LearningRate = _config.LearningRate,
            NumberOfLeaves = _config.NumberOfLeaves,
            MinimumExampleCountPerLeaf = _config.MinimumExampleCountPerLeaf,
            NumberOfIterations = _config.NumberOfIterations
        };

        return _mlContext.Transforms.Concatenate("Features", "Features")
            .Append(_mlContext.BinaryClassification.Trainers.LightGbm(options));
    }

    public void PredictPro(ITransformer model, FileData fileData, int bytesPerSection)
    {
        int featureCount = ProFileFeatures.SectionCount * bytesPerSection;
        var proFeatures = ProFeatureExtractor.ExtractFeatures(fileData.FilePath, bytesPerSection);
        var schemaDef = SchemaDefinition.Create(typeof(ProBinaryTrainingData));
        schemaDef["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, featureCount);
        var predictionEngine = _mlContext.Model.CreatePredictionEngine<ProBinaryTrainingData, BinaryModelPrediction>(model, inputSchemaDefinition: schemaDef);
        var prediction = predictionEngine.Predict(new ProBinaryTrainingData(bytesPerSection)
        {
            Features = proFeatures.ToFloatArray(),
            Label = fileData.Label
        });
        PrintPrediction(fileData, prediction);
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

        var pipeline = BuildPipeline(labelColumnName);

        Console.WriteLine($"正在训练{modeLabel}LightGBM 模型...");
        var model = pipeline.Fit(trainData);

        Console.WriteLine($"正在评估{modeLabel}模型...");
        EvaluateModel(model, testData, labelColumnName);

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
            NumberOfIterations = _config.NumberOfIterations
        };

        return _mlContext.Transforms.Concatenate("Features", "Features")
            .Append(_mlContext.BinaryClassification.Trainers.LightGbm(options));
    }

    private void EvaluateModel(ITransformer model, IDataView testData, string labelColumnName)
    {
        var predictions = model.Transform(testData);

        var metrics = _mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: labelColumnName);

        Console.WriteLine("\n=== 模型评估结果 ===");
        Console.WriteLine($"准确率 (Accuracy): {metrics.Accuracy:P4}");
        Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P4}");
        Console.WriteLine($"AUPRC: {metrics.AreaUnderPrecisionRecallCurve:P4}");
        Console.WriteLine($"F1 分数：{metrics.F1Score:P4}");

        Console.WriteLine("\n混淆矩阵:");
        Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
    }

    public void Predict(ITransformer model, FileData fileData)
    {
        var predictionEngine = _mlContext.Model.CreatePredictionEngine<BinaryTrainingData, BinaryModelPrediction>(model);
        var prediction = predictionEngine.Predict(new BinaryTrainingData
        {
            Features = fileData.Features.ToFloatArray(),
            Label = fileData.Label
        });
        PrintPrediction(fileData, prediction);
    }

    public void PredictFlash(ITransformer model, FileData fileData)
    {
        var predictionEngine = _mlContext.Model.CreatePredictionEngine<FlashBinaryTrainingData, BinaryModelPrediction>(model);
        var prediction = predictionEngine.Predict(new FlashBinaryTrainingData
        {
            Features = fileData.FlashFeatures.ToFloatArray(),
            Label = fileData.Label
        });
        PrintPrediction(fileData, prediction);
    }

    private static void PrintPrediction(FileData fileData, BinaryModelPrediction prediction)
    {
        Console.WriteLine($"\n文件：{Path.GetFileName(fileData.FilePath)}");
        Console.WriteLine($"实际标签：{(fileData.Label ? "黑文件" : "白文件")}");
        Console.WriteLine($"预测标签：{(prediction.PredictedLabel ? "黑文件" : "白文件")}");
        Console.WriteLine($"预测概率：{prediction.Probability:P4}");
        Console.WriteLine($"预测分数：{prediction.Score:F4}");
    }
}

public class BinaryTrainingData
{
    [VectorType(FileFeatures.FeatureCount)]
    public float[] Features { get; set; } = Array.Empty<float>();

    public bool Label { get; set; }
}

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
