using Xdows_Model_Invoker;

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

static void AssertDecision(float probability, double threshold, AdaptiveIntermediateDecision expected, string scenario)
{
    var actual = AdaptiveDecisionPolicy.EvaluateIntermediate(probability, threshold);
    if (actual != expected)
        throw new InvalidOperationException($"{scenario}: expected {expected}, got {actual}.");
}
