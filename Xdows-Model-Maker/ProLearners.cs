using Microsoft.ML;
using Xdows_Model_Config;

namespace Xdows_Model_Maker;

internal interface IProLearner
{
    string Name { get; }
    IEstimator<ITransformer> BuildPipeline(MLContext mlContext, TrainingConfig config, int featureCount);
}

internal static class ProLearnerFactory
{
    public static IProLearner Create(string learnerName)
    {
        string normalized = string.IsNullOrWhiteSpace(learnerName)
            ? "lightgbm"
            : learnerName.Trim().ToLowerInvariant();

        return normalized switch
        {
            "lightgbm" or "lgbm" => new LightGbmProLearner(),
            "catboost" => new CatBoostProLearner(),
            _ => throw new NotSupportedException($"未知 Pro learner: {learnerName}")
        };
    }
}

internal sealed class LightGbmProLearner : IProLearner
{
    public string Name => "LightGBM";

    public IEstimator<ITransformer> BuildPipeline(MLContext mlContext, TrainingConfig config, int featureCount)
    {
        var options = new Microsoft.ML.Trainers.LightGbm.LightGbmBinaryTrainer.Options
        {
            LabelColumnName = "Label",
            FeatureColumnName = "Features",
            LearningRate = config.ProLearningRate,
            NumberOfLeaves = config.ProNumberOfLeaves,
            MinimumExampleCountPerLeaf = config.ProMinimumExampleCountPerLeaf,
            NumberOfIterations = config.ProNumberOfIterations,
            Booster = new Microsoft.ML.Trainers.LightGbm.GradientBooster.Options
            {
                L1Regularization = config.ProL1Regularization,
                L2Regularization = config.ProL2Regularization
            }
        };

        return mlContext.Transforms.NormalizeMinMax("Features", "Features")
            .Append(mlContext.BinaryClassification.Trainers.LightGbm(options));
    }
}

internal sealed class CatBoostProLearner : IProLearner
{
    public string Name => "CatBoost";

    public IEstimator<ITransformer> BuildPipeline(MLContext mlContext, TrainingConfig config, int featureCount)
    {
        throw new NotSupportedException("CatBoost learner 已作为 Pro 引擎扩展点预留，但当前主流程依赖 ML.NET/ONNX，尚未接入稳定的 CatBoost 训练与导出链路。");
    }
}
