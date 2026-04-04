using Microsoft.ML;
using Microsoft.ML.Data;

namespace Xdows_Model_Maker;

public class ModelTrainer
{
    private readonly MLContext _mlContext;
    private IDataView? _fullDataView;
    private readonly TrainingConfig _config;
    private const int ExpectedFeatureCount = 276;

    public ModelTrainer(TrainingConfig config)
    {
        _config = config;
        _mlContext = new MLContext(seed: config.RandomSeed);
    }

    public ITransformer TrainModel(List<FileData> fileData, string modelPath, string? onnxPath = null)
    {
        Console.WriteLine("\n开始训练模型...");

        var validData = new List<FileData>();
        int emptyFeaturesCount = 0;
        int wrongSizeCount = 0;

        foreach (var fd in fileData)
        {
            var features = fd.Features.ToFloatArray();
            if (features.Length == 0)
            {
                emptyFeaturesCount++;
            }
            else if (features.Length != ExpectedFeatureCount)
            {
                wrongSizeCount++;
                if (wrongSizeCount <= 3)
                {
                    Console.WriteLine($"警告：文件 {fd.FilePath} 特征数量为 {features.Length}，期望 {ExpectedFeatureCount}");
                }
            }
            else
            {
                validData.Add(fd);
            }
        }

        if (emptyFeaturesCount > 0)
        {
            Console.WriteLine($"警告：{emptyFeaturesCount} 个文件特征为空");
        }
        if (wrongSizeCount > 0)
        {
            Console.WriteLine($"警告：{wrongSizeCount} 个文件特征数量不正确");
        }

        if (validData.Count == 0)
        {
            Console.WriteLine("错误：没有有效的训练数据！");
            return null!;
        }

        Console.WriteLine($"有效训练数据：{validData.Count} 个");

        var trainingData = validData.Select(fd => new BinaryTrainingData
        {
            Features = fd.Features.ToFloatArray(),
            Label = fd.Label
        }).ToList();

        _fullDataView = _mlContext.Data.LoadFromEnumerable(trainingData);

        var trainTestSplit = _mlContext.Data.TrainTestSplit(_fullDataView, testFraction: 0.2);
        var trainData = trainTestSplit.TrainSet;
        var testData = trainTestSplit.TestSet;

        var pipeline = BuildPipeline();

        Console.WriteLine("正在训练 LightGBM 模型...");
        var model = pipeline.Fit(trainData);

        Console.WriteLine("正在评估模型...");
        EvaluateModel(model, testData);

        Console.WriteLine($"正在保存 ML.NET 模型到：{modelPath}");
        _mlContext.Model.Save(model, trainData.Schema, modelPath);
        Console.WriteLine("ML.NET 模型保存成功!");

        if (!string.IsNullOrEmpty(onnxPath))
        {
            Console.WriteLine($"\n正在导出 ONNX 模型到：{onnxPath}");
            try
            {
                ExportToOnnx(model, _fullDataView, onnxPath);
                Console.WriteLine("ONNX 模型导出成功!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ONNX 导出失败：{ex.Message}");
                Console.WriteLine("注意：并非所有 ML.NET 模型都支持导出为 ONNX 格式。");
            }
        }

        return model;
    }

    public ITransformer TrainModel(List<FileData> fileData)
    {
        return TrainModel(fileData, _config.ModelPath, _config.OnnxPath);
    }

    private void ExportToOnnx(ITransformer model, IDataView dataView, string onnxPath)
    {
        using var stream = File.Create(onnxPath);
        _mlContext.Model.ConvertToOnnx(model, dataView, stream);
    }

    private IEstimator<ITransformer> BuildPipeline()
    {
        var options = new Microsoft.ML.Trainers.LightGbm.LightGbmBinaryTrainer.Options
        {
            LabelColumnName = nameof(BinaryTrainingData.Label),
            FeatureColumnName = "Features",
            LearningRate = _config.LearningRate,
            NumberOfLeaves = _config.NumberOfLeaves,
            MinimumExampleCountPerLeaf = _config.MinimumExampleCountPerLeaf,
            NumberOfIterations = _config.NumberOfIterations
        };

        var pipeline = _mlContext.Transforms.Concatenate("Features", nameof(BinaryTrainingData.Features))
            .Append(_mlContext.BinaryClassification.Trainers.LightGbm(options));

        return pipeline;
    }

    private void EvaluateModel(ITransformer model, IDataView testData)
    {
        var predictions = model.Transform(testData);

        var metrics = _mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: nameof(BinaryTrainingData.Label));

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

        var input = new BinaryTrainingData
        {
            Features = fileData.Features.ToFloatArray(),
            Label = fileData.Label
        };

        var prediction = predictionEngine.Predict(input);

        Console.WriteLine($"\n文件：{Path.GetFileName(fileData.FilePath)}");
        Console.WriteLine($"实际标签：{(fileData.Label ? "黑文件" : "白文件")}");
        Console.WriteLine($"预测标签：{(prediction.PredictedLabel ? "黑文件" : "白文件")}");
        Console.WriteLine($"预测概率：{prediction.Probability:P4}");
        Console.WriteLine($"预测分数：{prediction.Score:F4}");
    }
}

public class BinaryTrainingData
{
    [VectorType(276)]
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
