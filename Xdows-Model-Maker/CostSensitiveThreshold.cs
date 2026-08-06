namespace Xdows_Model_Maker;

public sealed record CostThresholdSelection(
    double Threshold,
    ThresholdMetrics Metrics,
    double Cost);

/// <summary>
/// 按「一个误报等于多少个漏报」的代价比挑选判毒阈值。
/// F1 对误报与漏报等权，与实际排行榜/运营偏好不一致，因此这里直接最小化加权错误代价。
/// </summary>
internal static class CostSensitiveThreshold
{
    public const double MinimumThreshold = 0.1;
    public const double MaximumThreshold = 99.9;

    public static CostThresholdSelection FindMinimumCostThreshold(
        IReadOnlyList<ThresholdEvaluationRow> rows,
        double falsePositiveCostRatio)
    {
        return FindMinimumCostThreshold(new ThresholdSweep(rows), falsePositiveCostRatio);
    }

    public static CostThresholdSelection FindMinimumCostThreshold(
        ThresholdSweep sweep,
        double falsePositiveCostRatio)
    {
        if (!double.IsFinite(falsePositiveCostRatio) || falsePositiveCostRatio <= 0)
            throw new ArgumentOutOfRangeException(nameof(falsePositiveCostRatio), "误报代价比必须是正有限值。");

        double bestThreshold = MaximumThreshold;
        ThresholdMetrics? bestMetrics = null;
        double bestCost = double.PositiveInfinity;

        foreach (double threshold in sweep.CandidateThresholds(MinimumThreshold, MaximumThreshold))
        {
            ThresholdMetrics metrics = sweep.Compute(threshold);
            double cost = ComputeCost(metrics, falsePositiveCostRatio);
            if (bestMetrics == null ||
                cost < bestCost - 0.000000001 ||
                (Math.Abs(cost - bestCost) <= 0.000000001 && threshold > bestThreshold))
            {
                bestThreshold = threshold;
                bestMetrics = metrics;
                bestCost = cost;
            }
        }

        bestMetrics ??= sweep.Compute(bestThreshold);
        return new CostThresholdSelection(bestThreshold, bestMetrics, ComputeCost(bestMetrics, falsePositiveCostRatio));
    }

    /// <summary>
    /// 加权错误代价：漏报权重 1，误报权重为代价比。数值越小越好。
    /// </summary>
    public static double ComputeCost(ThresholdMetrics metrics, double falsePositiveCostRatio)
    {
        return metrics.FalseNegative + falsePositiveCostRatio * metrics.FalsePositive;
    }

    /// <summary>
    /// 代价比对应的贝叶斯最优概率阈值，用于校验模型概率标定是否与代价一致。
    /// </summary>
    public static double BayesOptimalThreshold(double falsePositiveCostRatio)
    {
        if (!double.IsFinite(falsePositiveCostRatio) || falsePositiveCostRatio <= 0)
            throw new ArgumentOutOfRangeException(nameof(falsePositiveCostRatio), "误报代价比必须是正有限值。");
        return 100.0 * falsePositiveCostRatio / (1.0 + falsePositiveCostRatio);
    }
}
