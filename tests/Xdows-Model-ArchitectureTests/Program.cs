using System.Diagnostics;
using Microsoft.ML;
using Xdows_Model_Config;
using Xdows_Model_Invoker;
using Xdows_Model_Maker;

if (args.Length > 0 && args[0].Equals("--benchmark-pro-prepare", StringComparison.OrdinalIgnoreCase))
{
    RunProPreparationBenchmark(
        args.Length > 1 ? int.Parse(args[1]) : 200,
        args.Length > 2 ? args[2] : @"D:\Code\Model\Files\Black",
        args.Length > 3 ? args[3] : @"D:\Code\Model\Files\White");
    return;
}

if (args.Length > 0 && args[0].Equals("--benchmark-standard-train", StringComparison.OrdinalIgnoreCase))
{
    RunStandardTrainingBenchmark(
        args.Length > 1 ? int.Parse(args[1]) : 200,
        args.Length > 2 ? args[2] : @"D:\Code\Model\Files\Black",
        args.Length > 3 ? args[3] : @"D:\Code\Model\Files\White");
    return;
}

if (args.Length > 0 && args[0].Equals("--benchmark-threshold-sweep", StringComparison.OrdinalIgnoreCase))
{
    RunThresholdSweepBenchmark(args.Length > 1 ? int.Parse(args[1]) : 50_000);
    return;
}

if (args.Length > 0 && args[0].Equals("--benchmark-pro-train", StringComparison.OrdinalIgnoreCase))
{
    RunProTrainingBenchmark(
        args.Length > 1 ? int.Parse(args[1]) : 2_000,
        args.Length > 2 ? int.Parse(args[2]) : 1,
        args.Length > 3 ? int.Parse(args[3]) : 40);
    return;
}

AssertModelModeContract();
AssertDecision(1, 96, AdaptiveIntermediateDecision.FinalSafe, "Flash high-confidence safe exit");
AssertDecision(4, 96, AdaptiveIntermediateDecision.FinalSafe, "Flash safe-exit boundary");
AssertDecision(4.0001f, 96, AdaptiveIntermediateDecision.Escalate, "Flash value above safe-exit boundary");
AssertDecision(50, 96, AdaptiveIntermediateDecision.Escalate, "Flash uncertainty escalation");
AssertDecision(99, 96, AdaptiveIntermediateDecision.Escalate, "Flash suspicious result must reach Pro");
AssertDecision(8, 92, AdaptiveIntermediateDecision.FinalSafe, "Standard safe-exit boundary");
AssertDecision(8.0001f, 92, AdaptiveIntermediateDecision.Escalate, "Standard value above safe-exit boundary");
AssertDecision(99, 92, AdaptiveIntermediateDecision.Escalate, "Standard suspicious result must reach Pro");

string peSamplePath = Path.Combine(Environment.SystemDirectory, "notepad.exe");
AssertAdaptiveInvoker(peSamplePath);
byte[] peBytes = File.ReadAllBytes(peSamplePath);
float[] standardFeatures = FeatureExtractor.ExtractFromBytes(peBytes).ToFloatArray();
float[] flashFeatures = FlashFeatureExtractor.ExtractFromBytes(peBytes).ToFloatArray();
float[] composed = AdaptiveFeatureComposer.ComposePro(peBytes, standardFeatures, flashFeatures);
float[] expected = ProHybridFeatureExtractor.ExtractFromBytes(peBytes).ToFloatArray();
if (composed.Length != expected.Length)
    throw new InvalidOperationException("Adaptive Pro composition length mismatch.");
for (int i = 0; i < composed.Length; i++)
{
    if (Math.Abs(composed[i] - expected[i]) > 0.00001f)
        throw new InvalidOperationException($"Adaptive Pro composition differs at feature {i}.");
}

Console.WriteLine("PASS: Adaptive intermediate stages cannot create a positive verdict.");

AssertStandardThresholdSelection();
AssertStandardStratifiedSplit();
AssertProFeatureCacheReuse();
AssertProBranchCopy();
AssertThresholdSweepEquivalence();
AssertProParallelScoringEquivalence();
AssertTrainingThreadResolution();
Console.WriteLine("PASS: Standard training policy preserves class balance and optimizes recall under an FPR cap.");
Console.WriteLine("PASS: Pro training reuses prepared features and branch copies preserve feature values.");
Console.WriteLine("PASS: Threshold sweep matches the exhaustive per-row confusion matrix.");
Console.WriteLine("PASS: Parallel Pro branch scoring reproduces the single-threaded fusion features.");
Console.WriteLine("PASS: LightGBM thread resolution prefers physical cores and honors explicit overrides.");

static void AssertModelModeContract()
{
    if ((int)ModelMode.Standard != 0 ||
        (int)ModelMode.Flash != 1 ||
        (int)ModelMode.Pro != 2 ||
        (int)ModelMode.Adaptive != 3)
    {
        throw new InvalidOperationException("Model mode values no longer match the native ABI contract.");
    }

    Console.WriteLine("PASS: Adaptive model mode matches the native ABI contract.");
}

static void AssertAdaptiveInvoker(string samplePath)
{
    ModelInvoker.InitializeAdaptive();
    try
    {
        if (!ModelInvoker.IsInitialized ||
            !ModelInvoker.IsAdaptiveMode ||
            ModelInvoker.CurrentMode != ModelMode.Adaptive)
        {
            throw new InvalidOperationException("ModelInvoker did not enter Adaptive mode.");
        }

        var (_, probability) = ModelInvoker.ScanFile(samplePath);
        if (!float.IsFinite(probability) || probability < 0 || probability > 100)
            throw new InvalidOperationException($"Adaptive probability is invalid: {probability}.");
    }
    finally
    {
        ModelInvoker.UnloadModel();
    }

    Console.WriteLine("PASS: ModelInvoker initializes and scans through Adaptive mode.");
}

static void AssertDecision(float probability, double threshold, AdaptiveIntermediateDecision expected, string scenario)
{
    var actual = AdaptiveDecisionPolicy.EvaluateIntermediate(probability, threshold);
    if (actual != expected)
        throw new InvalidOperationException($"{scenario}: expected {expected}, got {actual}.");
}

static void AssertStandardThresholdSelection()
{
    var rows = new List<ThresholdEvaluationRow>
    {
        new() { Label = true, Probability = 0.95f },
        new() { Label = true, Probability = 0.85f },
        new() { Label = false, Probability = 0.91f },
        new() { Label = false, Probability = 0.80f },
        new() { Label = false, Probability = 0.30f },
        new() { Label = false, Probability = 0.20f }
    };

    var result = StandardTrainingPolicy.FindThresholdAtMaximumFalsePositiveRate(rows, 0.25);
    if (result.Metrics.TruePositiveRate != 1.0 || result.Metrics.FalsePositiveRate > 0.25)
        throw new InvalidOperationException("Standard threshold selection did not maximize recall under the FPR cap.");
}

static void AssertStandardStratifiedSplit()
{
    var rows = Enumerable.Range(0, 20)
        .Select(index => new BinaryTrainingData
        {
            Features = new float[FileFeatures.FeatureCount],
            Label = index < 10
        })
        .ToList();

    var split = StandardTrainingPolicy.CreateStratifiedHoldout(rows, 0.2, 43846);
    if (split.Test.Count != 4 || split.Test.Count(row => row.Label) != 2 || split.Test.Count(row => !row.Label) != 2)
        throw new InvalidOperationException("Standard holdout is not stratified by class.");
}

static void AssertProFeatureCacheReuse()
{
    var preparedFeatures = new float[FeatureSchema.ProHybridFeatureCount];
    var data = new List<FileData>
    {
        new()
        {
            FilePath = @"Z:\path-that-must-not-be-read.exe",
            Label = true,
            ProFeatures = preparedFeatures,
            ProFeaturesAttempted = true
        }
    };

    ProFeatureCache cache = ProFeatureCache.Build(data);
    if (cache.FailedCount != 0 || cache.Entries.Count != 1 ||
        !ReferenceEquals(preparedFeatures, cache.Entries[0].Features))
    {
        throw new InvalidOperationException("Pro feature cache did not reuse the prepared feature vector.");
    }
}

static void AssertProBranchCopy()
{
    float[] features = Enumerable.Range(0, FeatureSchema.ProHybridFeatureCount).Select(index => (float)index).ToArray();
    foreach (ProBranch branch in Enum.GetValues<ProBranch>())
    {
        float[] expected = ProStackingTrainer.ExtractBranch(features, branch);
        var actual = new float[expected.Length];
        ProStackingTrainer.CopyBranch(features, branch, actual);
        if (!expected.SequenceEqual(actual))
            throw new InvalidOperationException($"Pro {branch} branch copy changed feature values.");
    }
}

static void AssertTrainingThreadResolution()
{
    int logicalCoreCount = Math.Max(1, Environment.ProcessorCount);
    int physicalCoreCount = TrainingHardware.PhysicalCoreCount;
    if (physicalCoreCount < 1 || physicalCoreCount > logicalCoreCount)
        throw new InvalidOperationException($"Physical core count {physicalCoreCount} is outside 1..{logicalCoreCount}.");

    if (TrainingHardware.ResolveTrainingThreadCount(null) != physicalCoreCount)
        throw new InvalidOperationException("Default LightGBM thread count must fall back to the physical core count.");

    if (TrainingHardware.ResolveTrainingThreadCount(2) != Math.Min(2, logicalCoreCount))
        throw new InvalidOperationException("Explicit LightGBM thread count was not honored.");

    if (TrainingHardware.ResolveTrainingThreadCount(logicalCoreCount * 4) != logicalCoreCount)
        throw new InvalidOperationException("LightGBM thread count must be clamped to the logical core count.");

    if (TrainingHardware.ResolveTrainingThreadCount(0) != physicalCoreCount ||
        TrainingHardware.ResolveTrainingThreadCount(-1) != physicalCoreCount)
    {
        throw new InvalidOperationException("Non-positive thread overrides must fall back to the physical core count.");
    }

    Console.WriteLine($"  INFO logicalCores={logicalCoreCount} physicalCores={physicalCoreCount}");
}

static void AssertProParallelScoringEquivalence()
{
    var random = new Random(43846);
    var samples = new List<ProStackingSample>(400);
    for (int index = 0; index < 400; index++)
    {
        bool label = (index & 1) == 0;
        var features = new float[FeatureSchema.ProHybridFeatureCount];
        for (int featureIndex = 0; featureIndex < features.Length; featureIndex++)
            features[featureIndex] = (float)(random.NextDouble() + (label ? 0.05 : 0));
        samples.Add(new ProStackingSample(features, label));
    }

    var config = new TrainingConfig
    {
        ProNumberOfIterations = 5,
        ProNumberOfLeaves = 7,
        ProMinimumExampleCountPerLeaf = 5,
        ProMaxParallelBranches = 1
    };
    var trainer = new ProStackingTrainer(new MLContext(seed: config.RandomSeed), config, new ProGbdtLearner());
    int[] indices = Enumerable.Range(0, samples.Count).ToArray();
    IReadOnlyList<ProBranchModel> branchModels = trainer.TrainBranches(samples, indices, 1, 1);

    List<ProFusionTrainingData> serial = trainer.ScoreSamples(samples, indices, branchModels, maxWorkerCount: 1);
    List<ProFusionTrainingData> parallel = trainer.ScoreSamples(samples, indices, branchModels);
    if (serial.Count != parallel.Count)
        throw new InvalidOperationException("Parallel Pro scoring returned a different row count.");

    for (int index = 0; index < serial.Count; index++)
    {
        if (serial[index].Label != parallel[index].Label ||
            !serial[index].Features.SequenceEqual(parallel[index].Features))
        {
            throw new InvalidOperationException($"Parallel Pro scoring changed fusion row {index}.");
        }
    }
}

static void AssertThresholdSweepEquivalence()
{
    var random = new Random(7);
    var rows = new List<ThresholdEvaluationRow>();
    for (int index = 0; index < 500; index++)
    {
        bool label = index % 3 == 0;
        rows.Add(new ThresholdEvaluationRow
        {
            Label = label,
            Probability = (float)(random.NextDouble() * (label ? 1.0 : 0.9))
        });
    }
    rows.Add(new ThresholdEvaluationRow { Label = true, Probability = 0f });
    rows.Add(new ThresholdEvaluationRow { Label = false, Probability = 1f });
    rows.Add(new ThresholdEvaluationRow { Label = true, Probability = float.NaN });
    rows.Add(new ThresholdEvaluationRow { Label = false, Probability = float.NaN });

    for (double threshold = 0; threshold <= 100.0001; threshold += 0.1)
    {
        ThresholdMetrics expected = ComputeThresholdMetricsExhaustively(rows, threshold);
        ThresholdMetrics actual = ModelTrainer.ComputeThresholdMetrics(rows, threshold);
        if (expected != actual)
            throw new InvalidOperationException($"Threshold sweep disagrees with the exhaustive scan at {threshold:F1}.");
    }
}

static ThresholdMetrics ComputeThresholdMetricsExhaustively(List<ThresholdEvaluationRow> rows, double threshold)
{
    long truePositive = 0;
    long falseNegative = 0;
    long falsePositive = 0;
    long trueNegative = 0;

    foreach (ThresholdEvaluationRow row in rows)
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

    return ModelTrainer.CreateThresholdMetrics(truePositive, falseNegative, falsePositive, trueNegative);
}

static void RunStandardTrainingBenchmark(int sampleCountPerClass, string sourceBlack, string sourceWhite)
{
    string benchmarkRoot = Path.Combine(Path.GetTempPath(), $"xdows-model-standard-benchmark-{Guid.NewGuid():N}");
    string blackFolder = Path.Combine(benchmarkRoot, "Black");
    string whiteFolder = Path.Combine(benchmarkRoot, "White");
    Directory.CreateDirectory(blackFolder);
    Directory.CreateDirectory(whiteFolder);

    try
    {
        CopySamples(sourceBlack, blackFolder, sampleCountPerClass, skipCount: 0);
        CopySamples(sourceWhite, whiteFolder, sampleCountPerClass, skipCount: sampleCountPerClass);

        List<FileData> data = DataLoader.LoadData(blackFolder, whiteFolder, mode: DataLoadMode.Both);
        var config = new TrainingConfig
        {
            BlackFolder = blackFolder,
            WhiteFolder = whiteFolder,
            NumberOfIterations = 60,
            FlashNumberOfIterations = 60,
            ModelPath = Path.Combine(benchmarkRoot, "Xdows-Model.zip"),
            OnnxPath = Path.Combine(benchmarkRoot, "Xdows-Model.onnx"),
            FlashModelPath = Path.Combine(benchmarkRoot, "Xdows-Model-Flash.zip"),
            FlashOnnxPath = Path.Combine(benchmarkRoot, "Xdows-Model-Flash.onnx")
        };

        var trainer = new ModelTrainer(config);
        var stopwatch = Stopwatch.StartNew();
        trainer.TrainModel(data);
        double standardSeconds = stopwatch.Elapsed.TotalSeconds;
        stopwatch.Restart();
        trainer.TrainFlashModel(data);
        double flashSeconds = stopwatch.Elapsed.TotalSeconds;

        Console.WriteLine(
            $"BENCHMARK StandardTrain samples={data.Count} standard={standardSeconds:F3}s flash={flashSeconds:F3}s " +
            $"standardOnnx={Sha256OfFile(config.OnnxPath)} flashOnnx={Sha256OfFile(config.FlashOnnxPath)}");
    }
    finally
    {
        Directory.Delete(benchmarkRoot, recursive: true);
    }
}

static string Sha256OfFile(string path)
{
    using System.Security.Cryptography.SHA256 sha = System.Security.Cryptography.SHA256.Create();
    using FileStream stream = File.OpenRead(path);
    return Convert.ToHexString(sha.ComputeHash(stream))[..16];
}

static void RunThresholdSweepBenchmark(int rowCount)
{
    var random = new Random(43846);
    var rows = new List<ThresholdEvaluationRow>(rowCount);
    for (int index = 0; index < rowCount; index++)
    {
        bool label = (index & 1) == 0;
        rows.Add(new ThresholdEvaluationRow
        {
            Label = label,
            Probability = (float)Math.Clamp(random.NextDouble() + (label ? 0.15 : -0.15), 0, 1)
        });
    }

    var stopwatch = Stopwatch.StartNew();
    (double threshold, ThresholdMetrics metrics) best = ModelTrainer.FindBestThreshold(rows);
    stopwatch.Stop();

    var exhaustiveStopwatch = Stopwatch.StartNew();
    double exhaustiveBestThreshold = 50;
    ThresholdMetrics? exhaustiveBestMetrics = null;
    for (double threshold = 50; threshold <= 99.9; threshold += 0.1)
    {
        ThresholdMetrics metrics = ComputeThresholdMetricsExhaustively(rows, threshold);
        if (exhaustiveBestMetrics == null || metrics.F1Score > exhaustiveBestMetrics.F1Score + 0.000001)
        {
            exhaustiveBestThreshold = threshold;
            exhaustiveBestMetrics = metrics;
        }
    }
    exhaustiveStopwatch.Stop();

    Console.WriteLine(
        $"BENCHMARK ThresholdSweep rows={rowCount} total={stopwatch.Elapsed.TotalSeconds:F3}s " +
        $"exhaustive={exhaustiveStopwatch.Elapsed.TotalSeconds:F3}s " +
        $"bestThreshold={best.threshold:F1} f1={best.metrics.F1Score:F8} " +
        $"exhaustiveBestThreshold={exhaustiveBestThreshold:F1}");
}

static void RunProPreparationBenchmark(int sampleCountPerClass, string sourceBlack, string sourceWhite)
{
    string benchmarkRoot = Path.Combine(Path.GetTempPath(), $"xdows-model-pro-benchmark-{Guid.NewGuid():N}");
    string blackFolder = Path.Combine(benchmarkRoot, "Black");
    string whiteFolder = Path.Combine(benchmarkRoot, "White");
    Directory.CreateDirectory(blackFolder);
    Directory.CreateDirectory(whiteFolder);

    try
    {
        CopySamples(sourceBlack, blackFolder, sampleCountPerClass, skipCount: 0);
        CopySamples(sourceWhite, whiteFolder, sampleCountPerClass, skipCount: sampleCountPerClass);

        var stopwatch = Stopwatch.StartNew();
        List<FileData> data = DataLoader.LoadData(blackFolder, whiteFolder, mode: DataLoadMode.ProOnly);
        ProFeatureCache cache = ProFeatureCache.Build(data);
        stopwatch.Stop();

        Console.WriteLine(
            $"BENCHMARK ProPrepare samples={data.Count} total={stopwatch.Elapsed.TotalSeconds:F3}s " +
            $"cache={cache.Elapsed.TotalSeconds:F3}s failed={cache.FailedCount}");
    }
    finally
    {
        Directory.Delete(benchmarkRoot, recursive: true);
    }
}

static void CopySamples(string sourceFolder, string destinationFolder, int count, int skipCount)
{
    int index = 0;
    foreach (string sourcePath in Directory.EnumerateFiles(sourceFolder).Order().Skip(skipCount).Take(count))
    {
        string destinationPath = Path.Combine(destinationFolder, $"{index++:D6}-{Path.GetFileName(sourcePath)}");
        File.Copy(sourcePath, destinationPath);
    }
}

static void RunProTrainingBenchmark(int sampleCount, int parallelBranches, int iterations)
{
    var random = new Random(43846);
    var samples = new List<ProStackingSample>(sampleCount);
    for (int sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
    {
        bool label = (sampleIndex & 1) == 0;
        var features = new float[FeatureSchema.ProHybridFeatureCount];
        for (int featureIndex = 0; featureIndex < features.Length; featureIndex++)
            features[featureIndex] = (float)(random.NextDouble() + (label ? 0.05 : 0));
        samples.Add(new ProStackingSample(features, label));
    }

    var config = new TrainingConfig
    {
        ProNumberOfIterations = iterations,
        ProNumberOfLeaves = 31,
        ProMinimumExampleCountPerLeaf = 10,
        ProMaxParallelBranches = parallelBranches
    };
    var stopwatch = Stopwatch.StartNew();
    ProStackingTrainingResult result = new ProStackingTrainer(
        new MLContext(seed: config.RandomSeed),
        config,
        new ProGbdtLearner()).Train(samples);
    stopwatch.Stop();
    Console.WriteLine(
        $"BENCHMARK ProTrain samples={sampleCount} iterations={config.ProNumberOfIterations} " +
        $"parallelBranches={parallelBranches} total={stopwatch.Elapsed.TotalSeconds:F3}s " +
        $"testAuc={result.Evaluation.TestAuc:F8} trainAuc={result.Evaluation.TrainAuc:F8}");
}
