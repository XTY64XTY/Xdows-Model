using Xdows_Model_Config;

namespace Xdows_Model_Maker;

internal static class TrainingDatasetReporter
{
    public static void Print(IReadOnlyList<FileData> data)
    {
        if (data.Count == 0)
            return;

        int blackCount = 0;
        foreach (FileData sample in data)
        {
            if (sample.Label)
                blackCount++;
        }
        int whiteCount = data.Count - blackCount;

        Console.WriteLine("\n=== 训练数据基线报告 ===");
        Console.WriteLine($"总样本: {data.Count}");
        Console.WriteLine($"黑样本: {blackCount} ({Ratio(blackCount, data.Count):P2})");
        Console.WriteLine($"白样本: {whiteCount} ({Ratio(whiteCount, data.Count):P2})");

        var lastWriteYears = new int?[data.Count];
        var peTimestampYearArray = new int?[data.Count];
        Parallel.For(0, data.Count, new ParallelOptions
        {
            MaxDegreeOfParallelism = Math.Max(1, Environment.ProcessorCount)
        }, index =>
        {
            string path = data[index].FilePath;
            lastWriteYears[index] = TryGetLastWriteYear(path);
            peTimestampYearArray[index] = TryGetPeTimestampYear(path);
        });
        var peTimestampYears = new List<int?>(peTimestampYearArray);

        PrintDistribution("文件修改年份分布", lastWriteYears);
        PrintDistribution("PE 时间戳年份分布", peTimestampYears);
        PrintOldSampleWarning(peTimestampYears);
        Console.WriteLine("========================\n");
    }

    private static void PrintDistribution(string title, IEnumerable<int?> years)
    {
        var counts = new SortedDictionary<int, int>();
        int unknown = 0;

        foreach (var year in years)
        {
            if (year.HasValue)
                counts[year.Value] = counts.TryGetValue(year.Value, out int count) ? count + 1 : 1;
            else
                unknown++;
        }

        Console.WriteLine(title + ":");
        foreach (var (year, count) in counts)
            Console.WriteLine($"  {year}: {count}");
        if (unknown > 0)
            Console.WriteLine($"  unknown: {unknown}");
    }

    private static void PrintOldSampleWarning(IReadOnlyList<int?> peTimestampYears)
    {
        int knownCount = 0;
        int year2020Count = 0;
        foreach (int? year in peTimestampYears)
        {
            if (!year.HasValue)
                continue;
            knownCount++;
            if (year.Value == 2020)
                year2020Count++;
        }

        if (knownCount == 0)
            return;

        double ratio = Ratio(year2020Count, knownCount);
        if (ratio >= 0.1)
        {
            Console.WriteLine($"警告：PE 时间戳为 2020 年的样本占已识别时间戳 {ratio:P2}。");
            Console.WriteLine("这部分旧样本可能让模型拟合过期分布；建议评估剔除副本，但保留原始文件。");
        }
    }

    private static int? TryGetLastWriteYear(string path)
    {
        try
        {
            return File.GetLastWriteTimeUtc(path).Year;
        }
        catch
        {
            return null;
        }
    }

    private static int? TryGetPeTimestampYear(string path)
    {
        try
        {
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 4096, FileOptions.SequentialScan);
            if (stream.Length < 64)
                return null;

            Span<byte> dosHeader = stackalloc byte[64];
            if (stream.Read(dosHeader) < dosHeader.Length)
                return null;

            int peHeaderOffset = BitConverter.ToInt32(dosHeader[0x3c..0x40]);
            if (peHeaderOffset < 0 || peHeaderOffset + 12 > stream.Length)
                return null;

            stream.Position = peHeaderOffset + 8;
            Span<byte> timestampBytes = stackalloc byte[4];
            if (stream.Read(timestampBytes) < timestampBytes.Length)
                return null;

            uint timestamp = BitConverter.ToUInt32(timestampBytes);
            if (timestamp == 0)
                return null;

            int year = DateTimeOffset.FromUnixTimeSeconds(timestamp).UtcDateTime.Year;
            return year is >= 1980 and <= 2100 ? year : null;
        }
        catch
        {
            return null;
        }
    }

    private static double Ratio(int value, int total) => total > 0 ? (double)value / total : 0;
}
