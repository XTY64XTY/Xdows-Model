using System.Diagnostics;
using Xdows_Model_Invoker;

namespace Xdows_Model_Maker;

internal sealed class ProFeatureCache
{
    private readonly List<ProFeatureCacheEntry> _entries;

    private ProFeatureCache(List<ProFeatureCacheEntry> entries, int failedCount, TimeSpan elapsed)
    {
        _entries = entries;
        FailedCount = failedCount;
        Elapsed = elapsed;
    }

    public IReadOnlyList<ProFeatureCacheEntry> Entries => _entries;
    public int FailedCount { get; }
    public TimeSpan Elapsed { get; }

    public static ProFeatureCache Build(IReadOnlyList<FileData> fileData)
    {
        var stopwatch = Stopwatch.StartNew();
        var entries = new ProFeatureCacheEntry?[fileData.Count];
        int failedCount = 0;
        int processedCount = 0;
        int progressInterval = Math.Max(100, fileData.Count / 100);
        var progressLock = new object();

        Console.WriteLine("正在构建 Pro 特征缓存...");

        Parallel.For(0, fileData.Count, new ParallelOptions
        {
            MaxDegreeOfParallelism = Math.Max(1, Environment.ProcessorCount)
        }, i =>
        {
            try
            {
                var fd = fileData[i];
                float[] features;
                if (fd.ProFeatures is { } precomputedFeatures &&
                    precomputedFeatures.Length == ProHybridFileFeatures.FeatureCount)
                {
                    features = precomputedFeatures;
                }
                else if (fd.ProFeaturesAttempted)
                {
                    throw new InvalidDataException("Pro features were not produced during data loading.");
                }
                else
                {
                    byte[] bytes = File.ReadAllBytes(fd.FilePath);
                    features = ProHybridFeatureExtractor.ExtractFromBytes(bytes).ToFloatArray();
                }

                entries[i] = new ProFeatureCacheEntry(fd.FilePath, fd.Label, features);
            }
            catch
            {
                Interlocked.Increment(ref failedCount);
            }

            int processed = Interlocked.Increment(ref processedCount);
            if (processed % progressInterval == 0 || processed == fileData.Count)
            {
                lock (progressLock)
                    Console.Write($"\rPro 缓存构建进度: {processed}/{fileData.Count}");
            }
        });

        stopwatch.Stop();
        var validEntries = entries.OfType<ProFeatureCacheEntry>().ToList();
        Console.WriteLine();
        Console.WriteLine($"Pro 特征缓存完成：有效 {validEntries.Count}，失败 {failedCount}，耗时 {stopwatch.Elapsed.TotalSeconds:F2}s");

        return new ProFeatureCache(validEntries, failedCount, stopwatch.Elapsed);
    }
}

internal sealed class ProFeatureCacheEntry
{
    public ProFeatureCacheEntry(
        string filePath,
        bool label,
        float[] features)
    {
        FilePath = filePath;
        Label = label;
        Features = features;
    }

    public string FilePath { get; }
    public bool Label { get; }
    public float[] Features { get; }
}
