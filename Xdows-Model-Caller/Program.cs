using Microsoft.ML;
using Microsoft.ML.Data;

namespace Xdows_Model_Caller;

internal class Program
{
    private static void Main(string[] args)
    {
        Console.WriteLine("Xdows Model 调用器 By Shiyi");
        Console.WriteLine();

        string filePath = string.Empty;
        if (args.Length > 0)
        {
            filePath = args[0];
        }
        else
        {
            Console.WriteLine("用法: Xdows-Model-Caller.exe <文件路径>");
            return;
        }

        Console.WriteLine($"开始扫描：{filePath}");
        Console.WriteLine("测试模型：开启");
        Console.WriteLine();

        if (!File.Exists(filePath))
        {
            Console.WriteLine("错误：文件不存在！");
            return;
        }

        string modelPath = "Xdows-Model.zip";
        if (string.IsNullOrEmpty(modelPath))
        {
            Console.WriteLine("错误：未找到模型文件！");
            return;
        }

        try
        {
            var features = FeatureExtractor.ExtractFeatures(filePath);
            var floatFeatures = features.ToFloatArray();

            var (isVirus, probability) = PredictWithMlNet(modelPath, floatFeatures);

            if (!isVirus)
            {
                Console.WriteLine($"Safe({probability:F2}%)");
            }
            else
            {
                Console.WriteLine($"Virus({probability:F2}%)");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"错误：{ex.Message}");
        }
    }

    private static (bool isVirus, float probability) PredictWithMlNet(string modelPath, float[] features)
    {
        var mlContext = new MLContext();

        // Load the model
        ITransformer model;
        using (var fileStream = new FileStream(modelPath, FileMode.Open, FileAccess.Read))
        {
            model = mlContext.Model.Load(fileStream, out _);
        }

        // Create prediction engine
        var predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

        // Create input
        var input = new ModelInput
        {
            Features = features
        };

        // Predict
        var prediction = predictionEngine.Predict(input);

        return (prediction.PredictedLabel, prediction.Probability * 100);
    }
}

public class ModelInput
{
    [VectorType(279)]
    public float[] Features { get; set; } = Array.Empty<float>();
}

public class ModelOutput
{
    [ColumnName("PredictedLabel")]
    public bool PredictedLabel { get; set; }

    [ColumnName("Score")]
    public float Score { get; set; }

    [ColumnName("Probability")]
    public float Probability { get; set; }
}
