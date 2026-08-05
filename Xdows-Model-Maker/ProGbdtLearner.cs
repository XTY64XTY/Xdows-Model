using Microsoft.ML;
using Xdows_Model_Config;

namespace Xdows_Model_Maker;

internal sealed class ProGbdtLearner
{
    public string Name => "GBDT (LightGBM)";

    public IEstimator<ITransformer> BuildPipeline(MLContext mlContext, TrainingConfig config, int? numberOfThreads = null)
    {
        var options = new Microsoft.ML.Trainers.LightGbm.LightGbmBinaryTrainer.Options
        {
            LabelColumnName = "Label",
            FeatureColumnName = "Features",
            LearningRate = config.ProLearningRate,
            NumberOfLeaves = config.ProNumberOfLeaves,
            MinimumExampleCountPerLeaf = config.ProMinimumExampleCountPerLeaf,
            NumberOfIterations = config.ProNumberOfIterations,
            NumberOfThreads = numberOfThreads ?? TrainingHardware.ResolveTrainingThreadCount(config.TrainingThreadCount),
            ForceColumnWise = config.ForceColumnWiseHistogram,
            Deterministic = true,
            Seed = config.RandomSeed,
            Booster = new Microsoft.ML.Trainers.LightGbm.GradientBooster.Options
            {
                L1Regularization = config.ProL1Regularization,
                L2Regularization = config.ProL2Regularization,
                MaximumTreeDepth = config.ProMaximumTreeDepth,
                FeatureFraction = config.ProFeatureFraction,
                SubsampleFraction = config.ProSubsampleFraction,
                SubsampleFrequency = 1
            }
        };

        return mlContext.BinaryClassification.Trainers.LightGbm(options);
    }
}
