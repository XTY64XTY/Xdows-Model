using Xdows_Model_Invoker;
using Xdows_Model_Maker;

AssertDecision(1, 96, AdaptiveIntermediateDecision.FinalSafe, "Flash high-confidence safe exit");
AssertDecision(4, 96, AdaptiveIntermediateDecision.FinalSafe, "Flash safe-exit boundary");
AssertDecision(4.0001f, 96, AdaptiveIntermediateDecision.Escalate, "Flash value above safe-exit boundary");
AssertDecision(50, 96, AdaptiveIntermediateDecision.Escalate, "Flash uncertainty escalation");
AssertDecision(99, 96, AdaptiveIntermediateDecision.Escalate, "Flash suspicious result must reach Pro");
AssertDecision(8, 92, AdaptiveIntermediateDecision.FinalSafe, "Standard safe-exit boundary");
AssertDecision(8.0001f, 92, AdaptiveIntermediateDecision.Escalate, "Standard value above safe-exit boundary");
AssertDecision(99, 92, AdaptiveIntermediateDecision.Escalate, "Standard suspicious result must reach Pro");

byte[] peBytes = File.ReadAllBytes(Path.Combine(Environment.SystemDirectory, "notepad.exe"));
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
Console.WriteLine("PASS: Standard training policy preserves class balance and optimizes recall under an FPR cap.");

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
