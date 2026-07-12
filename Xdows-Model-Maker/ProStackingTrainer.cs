using Microsoft.ML;
using Microsoft.ML.Data;
using Xdows_Model_Config;

namespace Xdows_Model_Maker;

internal enum ProBranch
{
    Standard,
    Flash,
    RawStat,
    Structural
}

internal sealed record ProStackingSample(float[] Features, bool Label);

internal sealed record ProBranchModel(ProBranch Branch, int FeatureCount, ITransformer Model, IDataView TrainingData);

internal sealed class ProStackingTrainingResult
{
    public required ITransformer FusionModel { get; init; }
    public required IDataView FusionTrainingData { get; init; }
    public required IReadOnlyList<ProBranchModel> BranchModels { get; init; }
    public required ProTrainingEvaluation Evaluation { get; init; }

    public void SaveArtifacts(MLContext mlContext, string modelPath, string? onnxPath)
    {
        mlContext.Model.Save(FusionModel, FusionTrainingData.Schema, modelPath);
        foreach (var branch in BranchModels)
            mlContext.Model.Save(branch.Model, branch.TrainingData.Schema, AddSuffix(modelPath, BranchSuffix(branch.Branch)));

        if (string.IsNullOrWhiteSpace(onnxPath))
            return;

        ExportToOnnx(mlContext, FusionModel, FusionTrainingData, onnxPath);
        foreach (var branch in BranchModels)
            ExportToOnnx(mlContext, branch.Model, branch.TrainingData, AddSuffix(onnxPath, BranchSuffix(branch.Branch)));
    }

    private static void ExportToOnnx(MLContext mlContext, ITransformer model, IDataView data, string path)
    {
        using var stream = File.Create(path);
        mlContext.Model.ConvertToOnnx(model, data, stream);
    }

    internal static string AddSuffix(string path, string suffix)
    {
        string directory = Path.GetDirectoryName(path) ?? string.Empty;
        return Path.Combine(directory, Path.GetFileNameWithoutExtension(path) + suffix + Path.GetExtension(path));
    }

    internal static string BranchSuffix(ProBranch branch) => branch switch
    {
        ProBranch.Standard => "-Standard",
        ProBranch.Flash => "-Flash",
        ProBranch.RawStat => "-RawStat",
        ProBranch.Structural => "-Structural",
        _ => throw new ArgumentOutOfRangeException(nameof(branch))
    };
}

internal sealed class ProStackingTrainer
{
    private const int RequestedFoldCount = 5;
    private readonly MLContext _mlContext;
    private readonly TrainingConfig _config;
    private readonly ProGbdtLearner _branchLearner;

    public ProStackingTrainer(MLContext mlContext, TrainingConfig config, ProGbdtLearner branchLearner)
    {
        _mlContext = mlContext;
        _config = config;
        _branchLearner = branchLearner;
    }

    public ProStackingTrainingResult Train(IReadOnlyList<ProStackingSample> samples)
    {
        var (trainIndices, testIndices) = CreateStratifiedHoldout(samples, 0.2, _config.RandomSeed ?? 43846);
        int minorityTrainCount = Math.Min(trainIndices.Count(i => samples[i].Label), trainIndices.Count(i => !samples[i].Label));
        int foldCount = Math.Min(RequestedFoldCount, minorityTrainCount);
        if (foldCount < 2)
            throw new InvalidOperationException("Pro Stacking 至少需要每类 3 个有效样本。");

        Console.WriteLine($"  Pro 架构：4 路 GBDT + Logistic Regression 融合，OOF={foldCount} 折");
        var folds = CreateStratifiedFolds(samples, trainIndices, foldCount, (_config.RandomSeed ?? 43846) + 1);
        var oofRows = new List<ProFusionTrainingData>(trainIndices.Length);

        for (int fold = 0; fold < folds.Count; fold++)
        {
            var validationSet = folds[fold].ToHashSet();
            var foldTraining = trainIndices.Where(i => !validationSet.Contains(i)).ToArray();
            var branchModels = TrainBranches(samples, foldTraining);
            using var engines = new BranchPredictionEngines(_mlContext, branchModels);

            foreach (int index in folds[fold])
            {
                oofRows.Add(new ProFusionTrainingData
                {
                    Features = engines.Predict(samples[index].Features),
                    Label = samples[index].Label
                });
            }
            Console.WriteLine($"  OOF 进度：{fold + 1}/{folds.Count}");
        }

        IDataView fusionTrainingData = _mlContext.Data.LoadFromEnumerable(oofRows);
        var fusionPipeline = _mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
            labelColumnName: nameof(ProFusionTrainingData.Label),
            featureColumnName: nameof(ProFusionTrainingData.Features));
        ITransformer fusionModel = fusionPipeline.Fit(fusionTrainingData);

        var finalBranches = TrainBranches(samples, trainIndices);
        List<ProFusionTrainingData> testRows;
        using (var engines = new BranchPredictionEngines(_mlContext, finalBranches))
        {
            testRows = testIndices.Select(index => new ProFusionTrainingData
            {
                Features = engines.Predict(samples[index].Features),
                Label = samples[index].Label
            }).ToList();
        }

        IDataView testData = _mlContext.Data.LoadFromEnumerable(testRows);
        var testPredictions = fusionModel.Transform(testData);
        var testMetrics = _mlContext.BinaryClassification.Evaluate(testPredictions);
        var thresholdRows = _mlContext.Data.CreateEnumerable<ThresholdEvaluationRow>(testPredictions, false).ToList();
        var trainMetrics = _mlContext.BinaryClassification.Evaluate(fusionModel.Transform(fusionTrainingData));
        var thresholdMetrics = ModelTrainer.ComputeThresholdMetrics(thresholdRows, _config.ProThreshold);
        var (bestThreshold, bestMetrics) = ModelTrainer.FindBestThreshold(thresholdRows);

        return new ProStackingTrainingResult
        {
            FusionModel = fusionModel,
            FusionTrainingData = fusionTrainingData,
            BranchModels = finalBranches,
            Evaluation = new ProTrainingEvaluation(
                testMetrics.AreaUnderRocCurve,
                testMetrics.AreaUnderPrecisionRecallCurve,
                trainMetrics.AreaUnderRocCurve,
                trainMetrics.AreaUnderRocCurve - testMetrics.AreaUnderRocCurve,
                thresholdMetrics,
                bestMetrics,
                bestThreshold,
                samples.Count,
                samples.Count(s => s.Label),
                samples.Count(s => !s.Label),
                FeatureSchema.ProFusionFeatureCount)
        };
    }

    private IReadOnlyList<ProBranchModel> TrainBranches(IReadOnlyList<ProStackingSample> samples, IReadOnlyList<int> indices)
    {
        return Enum.GetValues<ProBranch>().Select(branch => TrainBranch(branch, samples, indices)).ToArray();
    }

    private ProBranchModel TrainBranch(ProBranch branch, IReadOnlyList<ProStackingSample> samples, IReadOnlyList<int> indices)
    {
        int featureCount = BranchFeatureCount(branch);
        var rows = indices.Select(index => new ProBinaryTrainingData(featureCount)
        {
            Features = ExtractBranch(samples[index].Features, branch),
            Label = samples[index].Label
        }).ToList();
        IDataView data = CreateDataView(_mlContext, rows, featureCount);
        ITransformer model = _branchLearner.BuildPipeline(_mlContext, _config).Fit(data);
        return new ProBranchModel(branch, featureCount, model, data);
    }

    internal static IDataView CreateDataView(MLContext mlContext, IEnumerable<ProBinaryTrainingData> rows, int featureCount)
    {
        return mlContext.Data.LoadFromEnumerable(rows, CreateSchema(featureCount));
    }

    internal static SchemaDefinition CreateSchema(int featureCount)
    {
        var schema = SchemaDefinition.Create(typeof(ProBinaryTrainingData));
        schema[nameof(ProBinaryTrainingData.Features)].ColumnType = new VectorDataViewType(NumberDataViewType.Single, featureCount);
        return schema;
    }

    internal static float[] ExtractBranch(float[] features, ProBranch branch)
    {
        var (offset, count) = branch switch
        {
            ProBranch.Standard => (FeatureSchema.ProStandardOffset, FeatureSchema.StandardFeatureCount),
            ProBranch.Flash => (FeatureSchema.ProFlashOffset, FeatureSchema.FlashFeatureCount),
            ProBranch.RawStat => (FeatureSchema.ProRawStatOffset, FeatureSchema.ProRawStatCount),
            ProBranch.Structural => (FeatureSchema.ProStructuralOffset, FeatureSchema.ProStructuralCount),
            _ => throw new ArgumentOutOfRangeException(nameof(branch))
        };
        var result = new float[count];
        Array.Copy(features, offset, result, 0, count);
        return result;
    }

    internal static int BranchFeatureCount(ProBranch branch) => branch switch
    {
        ProBranch.Standard => FeatureSchema.StandardFeatureCount,
        ProBranch.Flash => FeatureSchema.FlashFeatureCount,
        ProBranch.RawStat => FeatureSchema.ProRawStatCount,
        ProBranch.Structural => FeatureSchema.ProStructuralCount,
        _ => throw new ArgumentOutOfRangeException(nameof(branch))
    };

    private static (int[] Train, int[] Test) CreateStratifiedHoldout(IReadOnlyList<ProStackingSample> samples, double testFraction, int seed)
    {
        var random = new Random(seed);
        var train = new List<int>();
        var test = new List<int>();
        foreach (bool label in new[] { false, true })
        {
            var indices = samples.Select((sample, index) => (sample, index))
                .Where(x => x.sample.Label == label)
                .Select(x => x.index)
                .OrderBy(_ => random.Next())
                .ToArray();
            if (indices.Length < 2)
                throw new InvalidOperationException("Pro Stacking 的每个类别至少需要 2 个样本。");
            int testCount = Math.Clamp((int)Math.Round(indices.Length * testFraction), 1, indices.Length - 1);
            test.AddRange(indices.Take(testCount));
            train.AddRange(indices.Skip(testCount));
        }
        return (train.OrderBy(_ => random.Next()).ToArray(), test.OrderBy(_ => random.Next()).ToArray());
    }

    private static List<int[]> CreateStratifiedFolds(IReadOnlyList<ProStackingSample> samples, IReadOnlyList<int> indices, int foldCount, int seed)
    {
        var random = new Random(seed);
        var folds = Enumerable.Range(0, foldCount).Select(_ => new List<int>()).ToArray();
        foreach (bool label in new[] { false, true })
        {
            var labelIndices = indices.Where(i => samples[i].Label == label).OrderBy(_ => random.Next()).ToArray();
            for (int i = 0; i < labelIndices.Length; i++)
                folds[i % foldCount].Add(labelIndices[i]);
        }
        return folds.Select(fold => fold.OrderBy(_ => random.Next()).ToArray()).ToList();
    }

    private sealed class BranchPredictionEngines : IDisposable
    {
        private readonly Dictionary<ProBranch, PredictionEngine<ProBinaryTrainingData, BinaryModelPrediction>> _engines = new();

        public BranchPredictionEngines(MLContext mlContext, IReadOnlyList<ProBranchModel> models)
        {
            foreach (var branch in models)
            {
                _engines[branch.Branch] = mlContext.Model.CreatePredictionEngine<ProBinaryTrainingData, BinaryModelPrediction>(
                    branch.Model,
                    inputSchemaDefinition: CreateSchema(branch.FeatureCount));
            }
        }

        public float[] Predict(float[] features)
        {
            var scores = new float[FeatureSchema.ProFusionFeatureCount];
            foreach (ProBranch branch in Enum.GetValues<ProBranch>())
            {
                float[] branchFeatures = ExtractBranch(features, branch);
                scores[(int)branch] = _engines[branch].Predict(new ProBinaryTrainingData(branchFeatures.Length)
                {
                    Features = branchFeatures
                }).Probability;
            }
            return scores;
        }

        public void Dispose()
        {
            foreach (var engine in _engines.Values)
                engine.Dispose();
        }
    }
}

public class ProFusionTrainingData
{
    [VectorType(FeatureSchema.ProFusionFeatureCount)]
    public float[] Features { get; set; } = Array.Empty<float>();
    public bool Label { get; set; }
}
