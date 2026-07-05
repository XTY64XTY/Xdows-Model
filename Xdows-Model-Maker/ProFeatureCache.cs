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
        var entries = new List<ProFeatureCacheEntry>(fileData.Count);
        int failedCount = 0;

        Console.WriteLine("正在构建 Pro 特征缓存...");

        for (int i = 0; i < fileData.Count; i++)
        {
            var fd = fileData[i];
            try
            {
                byte[] bytes = File.ReadAllBytes(fd.FilePath);
                var standardFeatures = FeatureExtractor.ExtractFromBytes(bytes).ToFloatArray();
                var flashFeatures = FlashFeatureExtractor.ExtractFromBytes(bytes).ToFloatArray();
                var structuralFeatures = ProHybridFeatureExtractor.ExtractStructuralFeatures(bytes);
                var rawStatFeatures = ProRawStatExtractor.ExtractFromBytes(bytes).ToFloatArray();

                entries.Add(new ProFeatureCacheEntry(
                    fd.FilePath,
                    fd.Label,
                    standardFeatures,
                    flashFeatures,
                    structuralFeatures,
                    rawStatFeatures));
            }
            catch
            {
                failedCount++;
            }

            if ((i + 1) % 100 == 0 || i + 1 == fileData.Count)
                Console.Write($"\rPro 缓存构建进度: {i + 1}/{fileData.Count}");
        }

        stopwatch.Stop();
        Console.WriteLine();
        Console.WriteLine($"Pro 特征缓存完成：有效 {entries.Count}，失败 {failedCount}，耗时 {stopwatch.Elapsed.TotalSeconds:F2}s");

        return new ProFeatureCache(entries, failedCount, stopwatch.Elapsed);
    }
}

internal sealed class ProFeatureCacheEntry
{
    private readonly float[] _standardFeatures;
    private readonly float[] _flashFeatures;
    private readonly float[] _structuralFeatures;
    private readonly float[] _rawStatFeatures;

    public ProFeatureCacheEntry(
        string filePath,
        bool label,
        float[] standardFeatures,
        float[] flashFeatures,
        float[] structuralFeatures,
        float[] rawStatFeatures)
    {
        FilePath = filePath;
        Label = label;
        _standardFeatures = standardFeatures;
        _flashFeatures = flashFeatures;
        _structuralFeatures = structuralFeatures;
        _rawStatFeatures = rawStatFeatures;
    }

    public string FilePath { get; }
    public bool Label { get; }

    public float[] CreateFeatures()
    {
        var features = new float[ProHybridFileFeatures.FeatureCount];
        int offset = 0;

        Array.Copy(_standardFeatures, 0, features, offset, _standardFeatures.Length);
        offset += _standardFeatures.Length;
        Array.Copy(_flashFeatures, 0, features, offset, _flashFeatures.Length);
        offset += _flashFeatures.Length;
        Array.Copy(_rawStatFeatures, 0, features, offset, _rawStatFeatures.Length);
        offset += _rawStatFeatures.Length;
        Array.Copy(_structuralFeatures, 0, features, offset, _structuralFeatures.Length);

        return features;
    }
}
