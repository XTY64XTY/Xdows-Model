using System.Diagnostics;

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

    public static ProFeatureCache Build(IReadOnlyList<FileData> fileData, int maxBytesPerSection)
    {
        int normalizedMaxBytes = Math.Max(1, maxBytesPerSection);
        var stopwatch = Stopwatch.StartNew();
        var entries = new List<ProFeatureCacheEntry>(fileData.Count);
        int failedCount = 0;

        Console.WriteLine($"正在构建 Pro 特征缓存，最大 Raw 每段 {normalizedMaxBytes} 字节...");

        for (int i = 0; i < fileData.Count; i++)
        {
            var fd = fileData[i];
            try
            {
                byte[] bytes = File.ReadAllBytes(fd.FilePath);
                var standardFeatures = FeatureExtractor.ExtractFromBytes(bytes).ToFloatArray();
                var flashFeatures = FlashFeatureExtractor.ExtractFromBytes(bytes).ToFloatArray();
                var structuralFeatures = ProHybridFeatureExtractor.ExtractStructuralFeatures(bytes);
                var rawWindows = ProRawWindowCache.Create(bytes, normalizedMaxBytes);

                entries.Add(new ProFeatureCacheEntry(
                    fd.FilePath,
                    fd.Label,
                    standardFeatures,
                    flashFeatures,
                    structuralFeatures,
                    rawWindows));
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
    private readonly ProRawWindowCache _rawWindows;

    public ProFeatureCacheEntry(
        string filePath,
        bool label,
        float[] standardFeatures,
        float[] flashFeatures,
        float[] structuralFeatures,
        ProRawWindowCache rawWindows)
    {
        FilePath = filePath;
        Label = label;
        _standardFeatures = standardFeatures;
        _flashFeatures = flashFeatures;
        _structuralFeatures = structuralFeatures;
        _rawWindows = rawWindows;
    }

    public string FilePath { get; }
    public bool Label { get; }

    public float[] CreateFeatures(int bytesPerSection)
    {
        int featureCount = ProHybridFileFeatures.GetFeatureCount(bytesPerSection);
        var features = new float[featureCount];
        int offset = 0;

        Array.Copy(_standardFeatures, 0, features, offset, _standardFeatures.Length);
        offset += _standardFeatures.Length;
        Array.Copy(_flashFeatures, 0, features, offset, _flashFeatures.Length);
        offset += _flashFeatures.Length;
        _rawWindows.CopyTo(bytesPerSection, features, offset);
        offset += ProHybridFileFeatures.GetRawFeatureCount(bytesPerSection);
        Array.Copy(_structuralFeatures, 0, features, offset, _structuralFeatures.Length);

        return features;
    }
}

internal sealed class ProRawWindowCache
{
    private ProRawWindowCache(int fileSize, byte[] head, int midStart, byte[] mid, int tailStart, byte[] tail)
    {
        FileSize = fileSize;
        Head = head;
        MidStart = midStart;
        Mid = mid;
        TailStart = tailStart;
        Tail = tail;
    }

    private int FileSize { get; }
    private byte[] Head { get; }
    private int MidStart { get; }
    private byte[] Mid { get; }
    private int TailStart { get; }
    private byte[] Tail { get; }

    public static ProRawWindowCache Create(byte[] bytes, int maxBytesPerSection)
    {
        int sectionSize = Math.Max(1, maxBytesPerSection);
        int fileSize = bytes.Length;

        int headLength = Math.Min(sectionSize, fileSize);
        int midStart = Math.Max(0, fileSize / 2 - sectionSize / 2);
        int midLength = Math.Min(sectionSize, fileSize - midStart);
        int tailStart = Math.Max(0, fileSize - sectionSize);
        int tailLength = fileSize - tailStart;

        return new ProRawWindowCache(
            fileSize,
            Slice(bytes, 0, headLength),
            midStart,
            Slice(bytes, midStart, midLength),
            tailStart,
            Slice(bytes, tailStart, tailLength));
    }

    public void CopyTo(int bytesPerSection, float[] destination, int destinationOffset)
    {
        int sectionSize = Math.Max(1, bytesPerSection);
        CopyBytes(Head, 0, destination, destinationOffset, sectionSize);

        int candidateMidStart = Math.Max(0, FileSize / 2 - sectionSize / 2);
        CopyBytes(Mid, candidateMidStart - MidStart, destination, destinationOffset + sectionSize, sectionSize);

        int candidateTailStart = Math.Max(0, FileSize - sectionSize);
        CopyBytes(Tail, candidateTailStart - TailStart, destination, destinationOffset + sectionSize * 2, sectionSize);
    }

    private static byte[] Slice(byte[] bytes, int start, int length)
    {
        if (length <= 0)
            return Array.Empty<byte>();

        var result = new byte[length];
        Array.Copy(bytes, start, result, 0, length);
        return result;
    }

    private static void CopyBytes(byte[] source, int sourceOffset, float[] destination, int destinationOffset, int maxLength)
    {
        if (maxLength <= 0 || source.Length == 0)
            return;

        int offset = Math.Max(0, sourceOffset);
        int length = Math.Min(maxLength, source.Length - offset);
        if (length <= 0)
            return;

        for (int i = 0; i < length; i++)
            destination[destinationOffset + i] = source[offset + i];
    }
}
