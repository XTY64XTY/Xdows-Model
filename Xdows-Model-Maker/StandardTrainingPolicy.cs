namespace Xdows_Model_Maker;

internal sealed record StandardTrainingSplit(
    IReadOnlyList<BinaryTrainingData> Train,
    IReadOnlyList<BinaryTrainingData> Test);

internal sealed record StandardThresholdSelection(
    double Threshold,
    ThresholdMetrics Metrics);

internal static class StandardTrainingPolicy
{
    public static StandardTrainingSplit CreateStratifiedHoldout(
        IReadOnlyList<BinaryTrainingData> rows,
        double testFraction,
        int seed)
    {
        if (testFraction <= 0 || testFraction >= 1)
            throw new ArgumentOutOfRangeException(nameof(testFraction), "Test fraction must be between 0 and 1.");

        var random = new Random(seed);
        var train = new List<BinaryTrainingData>();
        var test = new List<BinaryTrainingData>();

        foreach (bool label in new[] { false, true })
        {
            BinaryTrainingData[] labelRows = rows
                .Where(row => row.Label == label)
                .OrderBy(_ => random.Next())
                .ToArray();
            if (labelRows.Length < 2)
                throw new InvalidOperationException("Standard training requires at least two samples from each class.");

            int testCount = Math.Clamp((int)Math.Round(labelRows.Length * testFraction), 1, labelRows.Length - 1);
            test.AddRange(labelRows.Take(testCount));
            train.AddRange(labelRows.Skip(testCount));
        }

        return new StandardTrainingSplit(
            train.OrderBy(_ => random.Next()).ToArray(),
            test.OrderBy(_ => random.Next()).ToArray());
    }

    public static StandardThresholdSelection FindThresholdAtMaximumFalsePositiveRate(
        IReadOnlyList<ThresholdEvaluationRow> rows,
        double maximumFalsePositiveRate)
    {
        if (maximumFalsePositiveRate < 0 || maximumFalsePositiveRate > 1)
            throw new ArgumentOutOfRangeException(nameof(maximumFalsePositiveRate));

        List<ThresholdEvaluationRow> materializedRows = rows as List<ThresholdEvaluationRow> ?? rows.ToList();
        double bestThreshold = 100.0;
        ThresholdMetrics bestMetrics = ModelTrainer.ComputeThresholdMetrics(materializedRows, bestThreshold);

        for (double threshold = 50.0; threshold <= 100.0; threshold += 0.1)
        {
            ThresholdMetrics metrics = ModelTrainer.ComputeThresholdMetrics(materializedRows, threshold);
            if (metrics.FalsePositiveRate > maximumFalsePositiveRate + 0.000000001)
                continue;

            if (metrics.TruePositiveRate > bestMetrics.TruePositiveRate + 0.000000001 ||
                (Math.Abs(metrics.TruePositiveRate - bestMetrics.TruePositiveRate) <= 0.000000001 &&
                 metrics.FalsePositiveRate < bestMetrics.FalsePositiveRate - 0.000000001))
            {
                bestThreshold = threshold;
                bestMetrics = metrics;
            }
        }

        return new StandardThresholdSelection(bestThreshold, bestMetrics);
    }
}
