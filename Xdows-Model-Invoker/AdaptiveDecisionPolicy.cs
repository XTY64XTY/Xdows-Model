namespace Xdows_Model_Invoker;

internal enum AdaptiveIntermediateDecision
{
    Escalate,
    FinalSafe
}

internal static class AdaptiveDecisionPolicy
{
    public static AdaptiveIntermediateDecision EvaluateIntermediate(float probability, double threatThreshold)
    {
        if (probability <= 100.0 - threatThreshold)
            return AdaptiveIntermediateDecision.FinalSafe;
        return AdaptiveIntermediateDecision.Escalate;
    }
}
