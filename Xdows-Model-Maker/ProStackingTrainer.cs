using Microsoft.ML;
using Microsoft.ML.Data;
using Xdows_Model_Config;

namespace Xdows_Model_Maker;

internal sealed record ProStackingSample(float[] Features, bool Label)
{
    /// <summary>可选的分组键（如恶意软件家族）：同组样本不会被切分到训练/测试两侧。</summary>
    public string? GroupKey { get; init; }

    /// <summary>可选的时间排序键（如 PE 时间戳），供 <see cref="ProDatasetSplitMode.TimeOrdered"/> 使用。</summary>
    public long? TimeKey { get; init; }

    /// <summary>文件内容的 SHA-256 摘要，用于剔除完全重复的样本；缺失时该样本不参与去重。</summary>
    public string? ContentHash { get; init; }
}

internal sealed record ProBranchModel(ProBranch Branch, int FeatureCount, ITransformer Model, IDataView TrainingData);

internal sealed class ProStackingTrainingResult
{
    public required ITransformer FusionModel { get; init; }
    public required IDataView FusionTrainingData { get; init; }
    public required IReadOnlyList<ProBranchModel> BranchModels { get; init; }
    public required ProTrainingEvaluation Evaluation { get; init; }

    public void SaveArtifacts(MLContext mlContext, string modelPath, string? onnxPath, double fixedThreshold)
    {
        mlContext.Model.Save(FusionModel, FusionTrainingData.Schema, modelPath);
        foreach (var branch in BranchModels)
            mlContext.Model.Save(branch.Model, branch.TrainingData.Schema, ProBranches.PathFor(branch.Branch, modelPath));

        if (string.IsNullOrWhiteSpace(onnxPath))
            return;

        ExportToOnnx(mlContext, FusionModel, FusionTrainingData, onnxPath);
        foreach (var branch in BranchModels)
            ExportToOnnx(mlContext, branch.Model, branch.TrainingData, ProBranches.PathFor(branch.Branch, onnxPath));

        var manifest = BuildManifest(onnxPath, fixedThreshold);
        manifest.Save(onnxPath);
        Console.WriteLine($"  Pro 模型清单已保存至: {ProModelManifest.ResolvePath(onnxPath)}");
    }

    private ProModelManifest BuildManifest(string fusionOnnxPath, double fixedThreshold)
    {
        return new ProModelManifest
        {
            SchemaVersion = ProModelManifest.CurrentSchemaVersion,
            ModelVersion = typeof(ProStackingTrainingResult).Assembly.GetName().Version?.ToString() ?? "unknown",
            FeatureSchemaVersion = FeatureSchema.Version,
            FeatureCount = FeatureSchema.ProHybridFeatureCount,
            FeatureHash = ProBranches.ComputeFeatureLayoutHash(),
            FusionModelFileName = Path.GetFileName(fusionOnnxPath),
            FusionInputCount = BranchModels.Count,
            Branches = BranchModels
                .OrderBy(branch => (int)branch.Branch)
                .Select(branch => new ProBranchManifestEntry
                {
                    Name = branch.Branch.ToString(),
                    FileName = ProBranches.FileNameFor(branch.Branch, fusionOnnxPath),
                    InputDimension = branch.FeatureCount,
                    FeatureOffset = ProBranches.Offset(branch.Branch),
                    FeatureCount = branch.FeatureCount
                })
                .ToList(),
            HashAlgorithm = ImportHashConfig.Algorithm,
            HashSeed = ImportHashConfig.Seed,
            DllHashDimensions = ImportHashConfig.DllHashDimensions,
            ApiHashDimensions = ImportHashConfig.ApiHashDimensions,
            Threshold = fixedThreshold,
            RecommendedThreshold = Evaluation.OperatingThreshold,
            GeneratedAt = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss")
        };
    }

    private static void ExportToOnnx(MLContext mlContext, ITransformer model, IDataView data, string path)
    {
        using var stream = File.Create(path);
        mlContext.Model.ConvertToOnnx(model, data, stream);
    }
}

internal sealed class ProStackingTrainer
{
    private static readonly ProBranch[] Branches = ProBranches.All.ToArray();
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
        ProDatasetSplitResult split = ProDatasetSplitter.Split(samples, _config);
        IReadOnlyList<ProStackingSample> data = split.Samples;
        IReadOnlyList<int> trainIndices = split.TrainIndices;
        IReadOnlyList<int> testIndices = split.TestIndices;

        int minorityTrainCount = Math.Min(trainIndices.Count(i => data[i].Label), trainIndices.Count(i => !data[i].Label));
        int foldCount = Math.Min(RequestedFoldCount, minorityTrainCount);
        if (foldCount < 2)
            throw new InvalidOperationException("Pro Stacking 至少需要每类 3 个有效样本。");

        Console.WriteLine($"  Pro 架构：{Branches.Length} 路 GBDT + Logistic Regression 融合，OOF={foldCount} 折");
        int parallelBranchCount = data.Count >= MinimumSamplesForParallelBranches
            ? _maxParallelBranchCount
            : 1;
        int trainingThreadCount = TrainingHardware.ResolveTrainingThreadCount(_config.TrainingThreadCount);
        int threadsPerBranch = Math.Max(1, trainingThreadCount / parallelBranchCount);
        Console.WriteLine($"  Pro 并行度：{parallelBranchCount} 个分支，LightGBM 每分支线程：{threadsPerBranch}");
        var folds = CreateStratifiedFolds(data, trainIndices, foldCount, (_config.RandomSeed ?? 43846) + 1);
        ProDatasetReporter.Print(split, foldCount);
        var oofRows = new List<ProFusionTrainingData>(trainIndices.Count);

        for (int fold = 0; fold < folds.Count; fold++)
        {
            var validationSet = folds[fold].ToHashSet();
            var foldTraining = trainIndices.Where(i => !validationSet.Contains(i)).ToArray();
            var branchModels = TrainBranches(data, foldTraining, parallelBranchCount, threadsPerBranch);
            oofRows.AddRange(ScoreSamples(data, folds[fold], branchModels));
            Console.WriteLine($"  OOF 进度：{fold + 1}/{folds.Count}");
        }

        IDataView fusionTrainingData = _mlContext.Data.LoadFromEnumerable(oofRows);
        var fusionPipeline = _mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
            labelColumnName: nameof(ProFusionTrainingData.Label),
            featureColumnName: nameof(ProFusionTrainingData.Features));
        ITransformer fusionModel = fusionPipeline.Fit(fusionTrainingData);

        var finalBranches = TrainBranches(data, trainIndices, parallelBranchCount, threadsPerBranch);
        List<ProFusionTrainingData> testRows = ScoreSamples(data, testIndices, finalBranches);

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
        foreach (var sample in data)
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
                data.Count,
                blackSampleCount,
                data.Count - blackSampleCount,
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
        return ProBranches.Extract(features, branch);
    }

    internal static void CopyBranch(float[] features, ProBranch branch, Span<float> destination)
    {
        ProBranches.Copy(features, branch, destination);
    }

    internal static int BranchFeatureCount(ProBranch branch) => ProBranches.FeatureCount(branch);

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
