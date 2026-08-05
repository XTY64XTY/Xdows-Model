namespace Xdows_Model_Maker;

internal sealed class ThresholdSweep
{
    private readonly double[] _positiveScores;
    private readonly double[] _negativeScores;
    private readonly long _positiveNeverPredicted;
    private readonly long _negativeNeverPredicted;

    public ThresholdSweep(IReadOnlyList<ThresholdEvaluationRow> rows)
    {
        var positives = new List<double>();
        var negatives = new List<double>();
        long positiveNeverPredicted = 0;
        long negativeNeverPredicted = 0;

        foreach (var row in rows)
        {
            double score = row.Probability * 100;
            if (double.IsNaN(score))
            {
                if (row.Label)
                    positiveNeverPredicted++;
                else
                    negativeNeverPredicted++;
                continue;
            }

            if (row.Label)
                positives.Add(score);
            else
                negatives.Add(score);
        }

        _positiveScores = positives.ToArray();
        _negativeScores = negatives.ToArray();
        Array.Sort(_positiveScores);
        Array.Sort(_negativeScores);
        _positiveNeverPredicted = positiveNeverPredicted;
        _negativeNeverPredicted = negativeNeverPredicted;
    }

    public ThresholdMetrics Compute(double threshold)
    {
        long truePositive = CountAtLeast(_positiveScores, threshold);
        long falseNegative = _positiveScores.Length - truePositive + _positiveNeverPredicted;
        long falsePositive = CountAtLeast(_negativeScores, threshold);
        long trueNegative = _negativeScores.Length - falsePositive + _negativeNeverPredicted;
        return ModelTrainer.CreateThresholdMetrics(truePositive, falseNegative, falsePositive, trueNegative);
    }

    private static long CountAtLeast(double[] sortedScores, double threshold)
    {
        int low = 0;
        int high = sortedScores.Length;
        while (low < high)
        {
            int middle = low + ((high - low) >> 1);
            if (sortedScores[middle] >= threshold)
                high = middle;
            else
                low = middle + 1;
        }
        return sortedScores.Length - low;
    }
}
