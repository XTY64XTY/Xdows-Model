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
    private static readonly ProBranch[] Branches = Enum.GetValues<ProBranch>();
    private const int RequestedFoldCount = 5;
    private const int MinimumSamplesForParallelBranches = 4_096;
    private readonly MLContext _mlContext;
    private readonly TrainingConfig _config;
    private readonly ProGbdtLearner _branchLearner;
    private readonly int _maxParallelBranchCount;

    public ProStackingTrainer(MLContext mlContext, TrainingConfig config, ProGbdtLearner branchLearner)
    {
        _mlContext = mlContext;
        _config = config;
        _branchLearner = branchLearner;
        _maxParallelBranchCount = Math.Clamp(config.ProMaxParallelBranches, 1, Branches.Length);
    }

    public ProStackingTrainingResult Train(IReadOnlyList<ProStackingSample> samples)
    {
        var (trainIndices, testIndices) = CreateStratifiedHoldout(samples, 0.2, _config.RandomSeed ?? 43846);
        int minorityTrainCount = Math.Min(trainIndices.Count(i => samples[i].Label), trainIndices.Count(i => !samples[i].Label));
        int foldCount = Math.Min(RequestedFoldCount, minorityTrainCount);
        if (foldCount < 2)
            throw new InvalidOperationException("Pro Stacking 至少需要每类 3 个有效样本。");

        Console.WriteLine($"  Pro 架构：4 路 GBDT + Logistic Regression 融合，OOF={foldCount} 折");
        int parallelBranchCount = samples.Count >= MinimumSamplesForParallelBranches
            ? _maxParallelBranchCount
            : 1;
        int trainingThreadCount = TrainingHardware.ResolveTrainingThreadCount(_config.TrainingThreadCount);
        int threadsPerBranch = Math.Max(1, trainingThreadCount / parallelBranchCount);
        Console.WriteLine($"  Pro 并行度：{parallelBranchCount} 个分支，LightGBM 每分支线程：{threadsPerBranch}");
        var folds = CreateStratifiedFolds(samples, trainIndices, foldCount, (_config.RandomSeed ?? 43846) + 1);
        var oofRows = new List<ProFusionTrainingData>(trainIndices.Length);

        for (int fold = 0; fold < folds.Count; fold++)
        {
            var validationSet = folds[fold].ToHashSet();
            var foldTraining = trainIndices.Where(i => !validationSet.Contains(i)).ToArray();
            var branchModels = TrainBranches(samples, foldTraining, parallelBranchCount, threadsPerBranch);
            oofRows.AddRange(ScoreSamples(samples, folds[fold], branchModels));
            Console.WriteLine($"  OOF 进度：{fold + 1}/{folds.Count}");
        }

        IDataView fusionTrainingData = _mlContext.Data.LoadFromEnumerable(oofRows);
        var fusionPipeline = _mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
            labelColumnName: nameof(ProFusionTrainingData.Label),
            featureColumnName: nameof(ProFusionTrainingData.Features));
        ITransformer fusionModel = fusionPipeline.Fit(fusionTrainingData);

        var finalBranches = TrainBranches(samples, trainIndices, parallelBranchCount, threadsPerBranch);
        List<ProFusionTrainingData> testRows = ScoreSamples(samples, testIndices, finalBranches);

        IDataView testData = _mlContext.Data.LoadFromEnumerable(testRows);
        var testPredictions = fusionModel.Transform(testData);
        var testMetrics = _mlContext.BinaryClassification.Evaluate(testPredictions);
        var thresholdRows = _mlContext.Data.CreateEnumerable<ThresholdEvaluationRow>(testPredictions, false).ToList();
        var trainMetrics = _mlContext.BinaryClassification.Evaluate(fusionModel.Transform(fusionTrainingData));
        var thresholdSweep = new ThresholdSweep(thresholdRows);
        CostThresholdSelection costSelection = CostSensitiveThreshold.FindMinimumCostThreshold(
            thresholdSweep,
            _config.FalsePositiveCostRatio);
        double operatingThreshold = _config.UseCostSensitiveThreshold
            ? costSelection.Threshold
            : _config.ProThreshold;
        var thresholdMetrics = thresholdSweep.Compute(operatingThreshold);
        var (bestThreshold, bestMetrics) = ModelTrainer.FindBestThreshold(thresholdSweep);
        ThresholdMetrics configuredMetrics = thresholdSweep.Compute(_config.ProThreshold);
        Console.WriteLine(
            $"  代价敏感阈值：{costSelection.Threshold:F2}%（误报代价比 {_config.FalsePositiveCostRatio}，" +
            $"FN {costSelection.Metrics.FalseNegative}，FP {costSelection.Metrics.FalsePositive}，加权代价 {costSelection.Cost:F1}）");
        Console.WriteLine(
            $"  固定阈值 {_config.ProThreshold:F2}% 对照：FN {configuredMetrics.FalseNegative}，FP {configuredMetrics.FalsePositive}，" +
            $"加权代价 {CostSensitiveThreshold.ComputeCost(configuredMetrics, _config.FalsePositiveCostRatio):F1}");
        Console.WriteLine($"  实际采用阈值：{operatingThreshold:F2}%");
        if (_config.UseCostSensitiveThreshold && Math.Abs(operatingThreshold - _config.ProThreshold) > 0.005)
        {
            Console.WriteLine(
                $"  注意：推理端仍使用 TrainingConfig.ProThreshold={_config.ProThreshold:F2}%。" +
                $"若要让线上工作点与本次校准一致，请把 ProThreshold 改为 {operatingThreshold:F2}。");
        }
        int blackSampleCount = 0;
        foreach (var sample in samples)
        {
            if (sample.Label)
                blackSampleCount++;
        }

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
                blackSampleCount,
                samples.Count - blackSampleCount,
                FeatureSchema.ProFusionFeatureCount,
                operatingThreshold,
                costSelection,
                configuredMetrics)
        };
    }

    internal List<ProFusionTrainingData> ScoreSamples(
        IReadOnlyList<ProStackingSample> samples,
        IReadOnlyList<int> indices,
        IReadOnlyList<ProBranchModel> branchModels,
        int? maxWorkerCount = null)
    {
        var rows = new ProFusionTrainingData[indices.Count];
        if (rows.Length == 0)
            return new List<ProFusionTrainingData>();

        int workerCount = Math.Clamp(maxWorkerCount ?? Environment.ProcessorCount, 1, rows.Length);
        var workers = new BranchPredictionEngines[workerCount];
        try
        {
            for (int worker = 0; worker < workerCount; worker++)
                workers[worker] = new BranchPredictionEngines(_mlContext, branchModels);

            int chunkSize = (rows.Length + workerCount - 1) / workerCount;
            Parallel.For(0, workerCount, worker =>
            {
                BranchPredictionEngines engines = workers[worker];
                int start = worker * chunkSize;
                int end = Math.Min(rows.Length, start + chunkSize);
                for (int position = start; position < end; position++)
                {
                    ProStackingSample sample = samples[indices[position]];
                    rows[position] = new ProFusionTrainingData
                    {
                        Features = engines.Predict(sample.Features),
                        Label = sample.Label
                    };
                }
            });
        }
        finally
        {
            foreach (BranchPredictionEngines engines in workers)
                engines?.Dispose();
        }

        return new List<ProFusionTrainingData>(rows);
    }

    internal IReadOnlyList<ProBranchModel> TrainBranches(
        IReadOnlyList<ProStackingSample> samples,
        IReadOnlyList<int> indices,
        int parallelBranchCount,
        int threadsPerBranch)
    {
        var models = new ProBranchModel[Branches.Length];
        Parallel.ForEach(
            Branches,
            new ParallelOptions { MaxDegreeOfParallelism = parallelBranchCount },
            branch => models[(int)branch] = TrainBranch(branch, samples, indices, threadsPerBranch));
        return models;
    }

    private ProBranchModel TrainBranch(
        ProBranch branch,
        IReadOnlyList<ProStackingSample> samples,
        IReadOnlyList<int> indices,
        int threadsPerBranch)
    {
        int featureCount = BranchFeatureCount(branch);
        var rows = new ProBinaryTrainingData[indices.Count];
        for (int position = 0; position < indices.Count; position++)
        {
            ProStackingSample sample = samples[indices[position]];
            rows[position] = new ProBinaryTrainingData
            {
                Features = ExtractBranch(sample.Features, branch),
                Label = sample.Label
            };
        }
        IDataView data = CreateDataView(_mlContext, rows, featureCount);
        ITransformer model = _branchLearner.BuildPipeline(_mlContext, _config, threadsPerBranch).Fit(data);
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
        var result = new float[BranchFeatureCount(branch)];
        CopyBranch(features, branch, result);
        return result;
    }

    internal static void CopyBranch(float[] features, ProBranch branch, Span<float> destination)
    {
        var (offset, count) = branch switch
        {
            ProBranch.Standard => (FeatureSchema.ProStandardOffset, FeatureSchema.StandardFeatureCount),
            ProBranch.Flash => (FeatureSchema.ProFlashOffset, FeatureSchema.FlashFeatureCount),
            ProBranch.RawStat => (FeatureSchema.ProRawStatOffset, FeatureSchema.ProRawStatCount),
            ProBranch.Structural => (FeatureSchema.ProStructuralOffset, FeatureSchema.ProStructuralCount),
            _ => throw new ArgumentOutOfRangeException(nameof(branch))
        };
        if (destination.Length != count)
            throw new ArgumentException("Pro branch destination length mismatch.", nameof(destination));
        features.AsSpan(offset, count).CopyTo(destination);
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
        private readonly BranchPredictionState?[] _states = new BranchPredictionState?[Branches.Length];

        public BranchPredictionEngines(MLContext mlContext, IReadOnlyList<ProBranchModel> models)
        {
            foreach (var branch in models)
            {
                var engine = mlContext.Model.CreatePredictionEngine<ProBinaryTrainingData, BinaryModelPrediction>(
                    branch.Model,
                    inputSchemaDefinition: CreateSchema(branch.FeatureCount));
                _states[(int)branch.Branch] = new BranchPredictionState(
                    engine,
                    new ProBinaryTrainingData { Features = new float[branch.FeatureCount] });
            }
        }

        public float[] Predict(float[] features)
        {
            var scores = new float[FeatureSchema.ProFusionFeatureCount];
            for (int branchIndex = 0; branchIndex < _states.Length; branchIndex++)
            {
                BranchPredictionState state = _states[branchIndex]
                    ?? throw new InvalidOperationException($"Pro {Branches[branchIndex]} 分支缺少预测引擎。");
                CopyBranch(features, Branches[branchIndex], state.Input.Features);
                scores[branchIndex] = state.Engine.Predict(state.Input).Probability;
            }
            return scores;
        }

        public void Dispose()
        {
            foreach (BranchPredictionState? state in _states)
                state?.Engine.Dispose();
        }

        private sealed record BranchPredictionState(
            PredictionEngine<ProBinaryTrainingData, BinaryModelPrediction> Engine,
            ProBinaryTrainingData Input);
    }
}

public class ProFusionTrainingData
{
    [VectorType(FeatureSchema.ProFusionFeatureCount)]
    public float[] Features { get; set; } = Array.Empty<float>();
    public bool Label { get; set; }
}
