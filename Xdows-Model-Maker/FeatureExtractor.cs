using System.Runtime.CompilerServices;

namespace Xdows_Model_Maker;

internal static class ByteAnalysisHelper
{
    public static bool IsPeFile(byte[] bytes)
    {
        if (bytes.Length < 2 || bytes[0] != 'M' || bytes[1] != 'Z')
            return false;

        if (bytes.Length < 64)
            return false;

        int peOffset = BitConverter.ToInt32(bytes, 60);
        if (peOffset + 4 > bytes.Length)
            return false;

        return bytes[peOffset] == 'P' && bytes[peOffset + 1] == 'E';
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsWhitespace(byte b) => b == 9 || b == 10 || b == 13 || b == 32;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsPrintable(byte b) => b >= 32 && b <= 126;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsLetter(byte b) => (b >= 65 && b <= 90) || (b >= 97 && b <= 122);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsDigit(byte b) => b >= 48 && b <= 57;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsHighByte(byte b) => b >= 0x80;

    public static void ComputeCommonStats(byte[] bytes, long[] byteCounts,
        out int printableCount, out int controlCount, out int whitespaceCount,
        out int letterCount, out int digitCount, out int maxZeroRun, out int highByteCount)
    {
        printableCount = 0;
        controlCount = 0;
        whitespaceCount = 0;
        letterCount = 0;
        digitCount = 0;
        maxZeroRun = 0;
        highByteCount = 0;
        int currentZeroRun = 0;

        foreach (var b in bytes)
        {
            byteCounts[b]++;

            if (IsHighByte(b))
                highByteCount++;

            if (IsWhitespace(b))
                whitespaceCount++;

            if (IsPrintable(b))
            {
                printableCount++;
                if (IsLetter(b))
                    letterCount++;
                else if (IsDigit(b))
                    digitCount++;
            }
            else
            {
                controlCount++;
            }

            if (b == 0)
            {
                currentZeroRun++;
                if (currentZeroRun > maxZeroRun)
                    maxZeroRun = currentZeroRun;
            }
            else
            {
                currentZeroRun = 0;
            }
        }
    }

    public static void ComputeCommonStatsSpan(ReadOnlySpan<byte> bytes, Span<long> byteCounts,
        out int printableCount, out int controlCount, out int whitespaceCount,
        out int letterCount, out int digitCount, out int maxZeroRun, out int highByteCount,
        out int zeroRunCount, out long totalZeroRunLength)
    {
        printableCount = 0;
        controlCount = 0;
        whitespaceCount = 0;
        letterCount = 0;
        digitCount = 0;
        maxZeroRun = 0;
        highByteCount = 0;
        zeroRunCount = 0;
        totalZeroRunLength = 0;
        int currentZeroRun = 0;

        for (int i = 0; i < bytes.Length; i++)
        {
            byte b = bytes[i];
            byteCounts[b]++;

            if (IsHighByte(b))
                highByteCount++;

            if (IsWhitespace(b))
                whitespaceCount++;

            if (IsPrintable(b))
            {
                printableCount++;
                if (IsLetter(b))
                    letterCount++;
                else if (IsDigit(b))
                    digitCount++;
            }
            else
            {
                controlCount++;
            }

            if (b == 0)
            {
                currentZeroRun++;
                if (currentZeroRun > maxZeroRun)
                    maxZeroRun = currentZeroRun;
            }
            else
            {
                if (currentZeroRun > 0)
                {
                    zeroRunCount++;
                    totalZeroRunLength += currentZeroRun;
                }
                currentZeroRun = 0;
            }
        }

        if (currentZeroRun > 0)
        {
            zeroRunCount++;
            totalZeroRunLength += currentZeroRun;
        }
    }

    public static double ComputeEntropy(Span<long> byteCounts, int totalBytes)
    {
        double entropy = 0;
        for (int i = 0; i < 256; i++)
        {
            long count = byteCounts[i];
            if (count > 0)
            {
                double p = (double)count / totalBytes;
                entropy -= p * Math.Log(p, 2);
            }
        }
        return entropy;
    }

    public static void ComputeStatsSummary(Span<long> byteCounts, int totalBytes,
        out int uniqueBytes, out double mostCommonByteRatio, out double zeroByteRatio)
    {
        uniqueBytes = 0;
        long maxCount = 0;
        for (int i = 0; i < 256; i++)
        {
            if (byteCounts[i] > 0)
                uniqueBytes++;
            if (byteCounts[i] > maxCount)
                maxCount = byteCounts[i];
        }
        mostCommonByteRatio = (double)maxCount / totalBytes;
        zeroByteRatio = (double)byteCounts[0] / totalBytes;
    }

    public static void ComputeByteMoments(Span<long> byteCounts, int totalBytes,
        out double mean, out double variance, out double skewness, out double kurtosis)
    {
        mean = 0;
        for (int i = 0; i < 256; i++)
            mean += i * (double)byteCounts[i] / totalBytes;

        variance = 0;
        double m3 = 0;
        double m4 = 0;
        for (int i = 0; i < 256; i++)
        {
            double p = (double)byteCounts[i] / totalBytes;
            double diff = i - mean;
            double diff2 = diff * diff;
            variance += diff2 * p;
            m3 += diff2 * diff * p;
            m4 += diff2 * diff2 * p;
        }

        double stdDev = Math.Sqrt(variance);
        skewness = stdDev > 0 ? m3 / (stdDev * stdDev * stdDev) : 0;
        kurtosis = variance > 0 ? m4 / (variance * variance) - 3 : 0;
    }

    public static double ComputeRegionEntropy(ReadOnlySpan<byte> bytes, int start, int length)
    {
        int actualLength = Math.Min(length, bytes.Length - start);
        if (actualLength <= 0) return 0;

        Span<long> counts = stackalloc long[256];
        counts.Clear();

        for (int i = start; i < start + actualLength; i++)
            counts[bytes[i]]++;

        return ComputeEntropy(counts, actualLength);
    }
}

public class FeatureExtractor
{
    public static FileFeatures ExtractFeatures(string filePath)
    {
        var bytes = File.ReadAllBytes(filePath);
        return ExtractFromBytes(bytes);
    }

    public static async Task<FileFeatures> ExtractFeaturesAsync(string filePath)
    {
        var bytes = await File.ReadAllBytesAsync(filePath);
        return ExtractFromBytes(bytes);
    }

    public static FileFeatures ExtractFromBytes(byte[] bytes)
    {
        var features = new FileFeatures { FileSize = bytes.Length };

        if (bytes.Length == 0)
            return features;

        Span<long> byteCounts = stackalloc long[256];
        ByteAnalysisHelper.ComputeCommonStatsSpan(bytes, byteCounts,
            out int printableCount, out int controlCount, out int whitespaceCount,
            out int letterCount, out int digitCount, out int maxZeroRun, out int highByteCount,
            out _, out _);

        int uniqueBytes = 0;
        long maxCount = 0, minCount = long.MaxValue;
        int mostCommonByte = 0, leastCommonByte = 0;

        for (int i = 0; i < 256; i++)
        {
            features.ByteFrequency[i] = (double)byteCounts[i] / bytes.Length;

            long c = byteCounts[i];
            if (c > 0) uniqueBytes++;
            if (c > maxCount) { maxCount = c; mostCommonByte = i; }
            if (c < minCount) { minCount = c; leastCommonByte = i; }
        }

        features.UniqueBytes = uniqueBytes;
        features.MostCommonByte = mostCommonByte;
        features.MostCommonByteRatio = (double)maxCount / bytes.Length;
        features.LeastCommonByte = leastCommonByte;
        features.LeastCommonByteRatio = (double)minCount / bytes.Length;
        features.ZeroByteRatio = (double)byteCounts[0] / bytes.Length;
        features.HighEntropyRatio = (double)highByteCount / bytes.Length;
        features.Entropy = ByteAnalysisHelper.ComputeEntropy(byteCounts, bytes.Length);

        ExtractBlockEntropyOptimized(bytes, features);

        features.PrintableCharRatio = (double)printableCount / bytes.Length;
        features.ControlCharRatio = (double)controlCount / bytes.Length;
        features.WhitespaceRatio = (double)whitespaceCount / bytes.Length;
        features.LetterRatio = (double)letterCount / bytes.Length;
        features.DigitRatio = (double)digitCount / bytes.Length;
        features.MaxZeroByteRun = maxZeroRun;

        return features;
    }

    private static void ExtractBlockEntropyOptimized(byte[] bytes, FileFeatures features)
    {
        const int blockSize = 256;
        int numBlocks = (bytes.Length + blockSize - 1) / blockSize;

        if (numBlocks == 0)
        {
            features.MinBlockEntropy = 0;
            features.MaxBlockEntropy = 0;
            features.MeanBlockEntropy = 0;
            features.BlockEntropyVariance = 0;
            features.MinEntropyBlockPosition = 0;
            features.MaxEntropyBlockPosition = 0;
            features.FirstBlockEntropy = 0;
            features.LastBlockEntropy = 0;
            return;
        }

        double minEntropy = double.MaxValue;
        double maxEntropy = double.MinValue;
        double totalEntropy = 0;
        double totalEntropySquared = 0;
        int minEntropyBlockIdx = 0;
        int maxEntropyBlockIdx = 0;
        double firstBlockEntropy = 0;
        double lastBlockEntropy = 0;

        var blockByteCounts = new long[256];

        for (int blockIdx = 0; blockIdx < numBlocks; blockIdx++)
        {
            int start = blockIdx * blockSize;
            int end = Math.Min(start + blockSize, bytes.Length);
            int currentBlockSize = end - start;

            blockByteCounts.AsSpan().Clear();

            for (int i = start; i < end; i++)
                blockByteCounts[bytes[i]]++;

            double blockEntropy = ByteAnalysisHelper.ComputeEntropy(blockByteCounts, currentBlockSize);

            if (blockIdx == 0) firstBlockEntropy = blockEntropy;
            if (blockIdx == numBlocks - 1) lastBlockEntropy = blockEntropy;

            if (blockEntropy < minEntropy)
            {
                minEntropy = blockEntropy;
                minEntropyBlockIdx = blockIdx;
            }
            if (blockEntropy > maxEntropy)
            {
                maxEntropy = blockEntropy;
                maxEntropyBlockIdx = blockIdx;
            }
            totalEntropy += blockEntropy;
            totalEntropySquared += blockEntropy * blockEntropy;
        }

        double meanEntropy = totalEntropy / numBlocks;
        double variance = (totalEntropySquared / numBlocks) - (meanEntropy * meanEntropy);
        if (variance < 0) variance = 0;

        features.MinBlockEntropy = minEntropy;
        features.MaxBlockEntropy = maxEntropy;
        features.MeanBlockEntropy = meanEntropy;
        features.BlockEntropyVariance = variance;
        features.MinEntropyBlockPosition = numBlocks > 1 ? (double)minEntropyBlockIdx / (numBlocks - 1) : 0;
        features.MaxEntropyBlockPosition = numBlocks > 1 ? (double)maxEntropyBlockIdx / (numBlocks - 1) : 0;
        features.FirstBlockEntropy = firstBlockEntropy;
        features.LastBlockEntropy = lastBlockEntropy;
    }

    public static bool IsPeFile(byte[] bytes) => ByteAnalysisHelper.IsPeFile(bytes);
}

public class FileFeatures
{
    public const int FeatureCount = 279;

    public double[] ByteFrequency { get; set; } = new double[256];
    public long FileSize { get; set; }
    public double Entropy { get; set; }
    public double MinBlockEntropy { get; set; }
    public double MaxBlockEntropy { get; set; }
    public double MeanBlockEntropy { get; set; }
    public double BlockEntropyVariance { get; set; }
    public double MinEntropyBlockPosition { get; set; }
    public double MaxEntropyBlockPosition { get; set; }
    public double FirstBlockEntropy { get; set; }
    public double LastBlockEntropy { get; set; }
    public int UniqueBytes { get; set; }
    public int MostCommonByte { get; set; }
    public double MostCommonByteRatio { get; set; }
    public int LeastCommonByte { get; set; }
    public double LeastCommonByteRatio { get; set; }
    public double PrintableCharRatio { get; set; }
    public double ControlCharRatio { get; set; }
    public double WhitespaceRatio { get; set; }
    public double LetterRatio { get; set; }
    public double DigitRatio { get; set; }
    public int MaxZeroByteRun { get; set; }

    public double ZeroByteRatio { get; set; }
    public double HighEntropyRatio { get; set; }

    public float[] ToFloatArray()
    {
        var features = new float[FeatureCount];
        int idx = 0;

        for (int i = 0; i < 256; i++)
            features[idx++] = (float)ByteFrequency[i];

        features[idx++] = (float)Math.Log(FileSize + 1);
        features[idx++] = (float)Entropy;
        features[idx++] = (float)MinBlockEntropy;
        features[idx++] = (float)MaxBlockEntropy;
        features[idx++] = (float)MeanBlockEntropy;
        features[idx++] = (float)BlockEntropyVariance;
        features[idx++] = (float)MinEntropyBlockPosition;
        features[idx++] = (float)MaxEntropyBlockPosition;
        features[idx++] = (float)FirstBlockEntropy;
        features[idx++] = (float)LastBlockEntropy;
        features[idx++] = UniqueBytes;
        features[idx++] = MostCommonByte;
        features[idx++] = (float)MostCommonByteRatio;
        features[idx++] = LeastCommonByte;
        features[idx++] = (float)LeastCommonByteRatio;
        features[idx++] = (float)PrintableCharRatio;
        features[idx++] = (float)ControlCharRatio;
        features[idx++] = (float)WhitespaceRatio;
        features[idx++] = (float)LetterRatio;
        features[idx++] = (float)DigitRatio;
        features[idx++] = MaxZeroByteRun;
        features[idx++] = (float)ZeroByteRatio;
        features[idx++] = (float)HighEntropyRatio;

        return features;
    }
}

public class FlashFileFeatures
{
    public const int FeatureCount = 20;

    public long FileSize { get; set; }
    public double Entropy { get; set; }
    public double ZeroByteRatio { get; set; }
    public double HighEntropyRatio { get; set; }
    public double PrintableCharRatio { get; set; }
    public double ControlCharRatio { get; set; }
    public double WhitespaceRatio { get; set; }
    public double LetterRatio { get; set; }
    public double DigitRatio { get; set; }
    public int UniqueBytes { get; set; }
    public double MostCommonByteRatio { get; set; }
    public int MaxZeroByteRun { get; set; }
    public double MeanByteValue { get; set; }
    public double ByteValueVariance { get; set; }
    public double ByteDistributionSkewness { get; set; }
    public double ByteDistributionKurtosis { get; set; }
    public double MeanZeroRunLength { get; set; }
    public int ZeroRunCount { get; set; }
    public double HeadBlockEntropy { get; set; }
    public double TailBlockEntropy { get; set; }

    public float[] ToFloatArray()
    {
        return new float[FeatureCount]
        {
            (float)Math.Log(FileSize + 1),
            (float)Entropy,
            (float)ZeroByteRatio,
            (float)HighEntropyRatio,
            (float)PrintableCharRatio,
            (float)ControlCharRatio,
            (float)WhitespaceRatio,
            (float)LetterRatio,
            (float)DigitRatio,
            UniqueBytes,
            (float)MostCommonByteRatio,
            MaxZeroByteRun,
            (float)MeanByteValue,
            (float)ByteValueVariance,
            (float)ByteDistributionSkewness,
            (float)ByteDistributionKurtosis,
            (float)MeanZeroRunLength,
            ZeroRunCount,
            (float)HeadBlockEntropy,
            (float)TailBlockEntropy
        };
    }
}

public class FlashFeatureExtractor
{
    private const int FlashSampleSize = 1024 * 1024;

    public static FlashFileFeatures ExtractFeatures(string filePath)
    {
        var fileInfo = new FileInfo(filePath);
        long fileSize = fileInfo.Length;

        if (fileSize == 0)
            return new FlashFileFeatures { FileSize = 0 };

        int bytesToRead = (int)Math.Min(fileSize, FlashSampleSize);
        byte[] buffer = new byte[bytesToRead];

        using (var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read, bytesToRead, FileOptions.SequentialScan))
        {
            int totalRead = 0;
            while (totalRead < bytesToRead)
            {
                int read = fs.Read(buffer, totalRead, bytesToRead - totalRead);
                if (read == 0) break;
                totalRead += read;
            }

            if (totalRead < bytesToRead)
                Array.Resize(ref buffer, totalRead);
        }

        return ExtractFromBytes(buffer, fileSize);
    }

    public static async Task<FlashFileFeatures> ExtractFeaturesAsync(string filePath)
    {
        var fileInfo = new FileInfo(filePath);
        long fileSize = fileInfo.Length;

        if (fileSize == 0)
            return new FlashFileFeatures { FileSize = 0 };

        int bytesToRead = (int)Math.Min(fileSize, FlashSampleSize);
        byte[] buffer = new byte[bytesToRead];

        using (var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read, bytesToRead, FileOptions.SequentialScan))
        {
            int totalRead = 0;
            while (totalRead < bytesToRead)
            {
                int read = await fs.ReadAsync(buffer, totalRead, bytesToRead - totalRead);
                if (read == 0) break;
                totalRead += read;
            }

            if (totalRead < bytesToRead)
                Array.Resize(ref buffer, totalRead);
        }

        return ExtractFromBytes(buffer, fileSize);
    }

    public static FlashFileFeatures ExtractFromBytes(byte[] bytes)
    {
        return ExtractFromBytes(bytes, bytes.Length);
    }

    public static FlashFileFeatures ExtractFromBytes(byte[] bytes, long actualFileSize)
    {
        var features = new FlashFileFeatures { FileSize = actualFileSize };

        if (bytes.Length == 0)
            return features;

        Span<long> byteCounts = stackalloc long[256];

        ByteAnalysisHelper.ComputeCommonStatsSpan(bytes, byteCounts,
            out int printableCount, out int controlCount, out int whitespaceCount,
            out int letterCount, out int digitCount, out int maxZeroRun, out int highByteCount,
            out int zeroRunCount, out long totalZeroRunLength);

        ByteAnalysisHelper.ComputeStatsSummary(byteCounts, bytes.Length,
            out int uniqueBytes, out double mostCommonByteRatio, out double zeroByteRatio);

        ByteAnalysisHelper.ComputeByteMoments(byteCounts, bytes.Length,
            out double meanByteValue, out double byteValueVariance,
            out double skewness, out double kurtosis);

        features.UniqueBytes = uniqueBytes;
        features.MostCommonByteRatio = mostCommonByteRatio;
        features.ZeroByteRatio = zeroByteRatio;
        features.HighEntropyRatio = (double)highByteCount / bytes.Length;
        features.Entropy = ByteAnalysisHelper.ComputeEntropy(byteCounts, bytes.Length);
        features.PrintableCharRatio = (double)printableCount / bytes.Length;
        features.ControlCharRatio = (double)controlCount / bytes.Length;
        features.WhitespaceRatio = (double)whitespaceCount / bytes.Length;
        features.LetterRatio = (double)letterCount / bytes.Length;
        features.DigitRatio = (double)digitCount / bytes.Length;
        features.MaxZeroByteRun = maxZeroRun;
        features.MeanByteValue = meanByteValue;
        features.ByteValueVariance = byteValueVariance;
        features.ByteDistributionSkewness = skewness;
        features.ByteDistributionKurtosis = kurtosis;
        features.MeanZeroRunLength = zeroRunCount > 0 ? (double)totalZeroRunLength / zeroRunCount : 0;
        features.ZeroRunCount = zeroRunCount;
        features.HeadBlockEntropy = ByteAnalysisHelper.ComputeRegionEntropy(bytes, 0, 512);
        features.TailBlockEntropy = ByteAnalysisHelper.ComputeRegionEntropy(bytes, Math.Max(0, bytes.Length - 512), 512);

        return features;
    }

    public static bool IsPeFile(byte[] bytes) => ByteAnalysisHelper.IsPeFile(bytes);
}

public class ProFileFeatures
{
    public const int SectionCount = 3;

    public int BytesPerSection { get; }
    public int FeatureCount => SectionCount * BytesPerSection;
    public float[] RawBytes { get; }

    public ProFileFeatures(int bytesPerSection)
    {
        BytesPerSection = bytesPerSection;
        RawBytes = new float[FeatureCount];
    }

    public float[] ToFloatArray()
    {
        var result = new float[FeatureCount];
        Array.Copy(RawBytes, result, FeatureCount);
        return result;
    }
}

public class ProFeatureExtractor
{
    private const int PeCheckSize = 512;

    public static ProFileFeatures ExtractFeatures(string filePath, int bytesPerSection)
    {
        var fileInfo = new FileInfo(filePath);
        long fileSize = fileInfo.Length;

        if (fileSize == 0)
            throw new NotSupportedException("文件为空");

        if (fileSize < 64)
            throw new NotSupportedException("文件过小，无法进行PE格式验证");

        using var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read, 4096, FileOptions.SequentialScan);

        int checkSize = (int)Math.Min(PeCheckSize, fileSize);
        byte[] checkBuf = new byte[checkSize];
        int checkRead = 0;
        while (checkRead < checkSize)
        {
            int n = fs.Read(checkBuf, checkRead, checkSize - checkRead);
            if (n == 0) break;
            checkRead += n;
        }
        if (!ByteAnalysisHelper.IsPeFile(checkBuf))
            throw new NotSupportedException("不支持该文件类型");

        var features = new ProFileFeatures(bytesPerSection);
        int sectionSize = bytesPerSection;

        ReadSection(fs, 0, (int)Math.Min(sectionSize, fileSize), features.RawBytes, 0);
        long midStart = Math.Max(0, fileSize / 2 - sectionSize / 2);
        int midLen = (int)(Math.Min(midStart + sectionSize, fileSize) - midStart);
        ReadSection(fs, midStart, midLen, features.RawBytes, sectionSize);
        long tailStart = Math.Max(0, fileSize - sectionSize);
        int tailLen = (int)(fileSize - tailStart);
        ReadSection(fs, tailStart, tailLen, features.RawBytes, sectionSize * 2);

        return features;
    }

    private static void ReadSection(FileStream fs, long start, int length, float[] buffer, int bufferOffset)
    {
        if (length <= 0) return;
        fs.Position = start;
        byte[] temp = new byte[length];
        int totalRead = 0;
        while (totalRead < length)
        {
            int n = fs.Read(temp, totalRead, length - totalRead);
            if (n == 0) break;
            totalRead += n;
        }
        for (int i = 0; i < totalRead; i++)
            buffer[bufferOffset + i] = temp[i];
    }

    public static ProFileFeatures ExtractFromBytes(byte[] bytes, int bytesPerSection)
    {
        if (bytes.Length == 0)
            throw new NotSupportedException("文件为空");

        if (bytes.Length < 64)
            throw new NotSupportedException("文件过小，无法进行PE格式验证");

        var features = new ProFileFeatures(bytesPerSection);

        int sectionSize = bytesPerSection;
        int totalFeatures = ProFileFeatures.SectionCount * sectionSize;

        long fileSize = bytes.Length;

        int headStart = 0;
        int headEnd = Math.Min(sectionSize, (int)fileSize);

        int midStart = Math.Max(0, (int)(fileSize / 2) - sectionSize / 2);
        int midEnd = Math.Min(midStart + sectionSize, (int)fileSize);

        int tailStart = Math.Max(0, (int)fileSize - sectionSize);
        int tailEnd = (int)fileSize;

        int offset = 0;

        for (int i = headStart; i < headEnd && offset < totalFeatures; i++)
            features.RawBytes[offset++] = bytes[i];
        offset = sectionSize;

        for (int i = midStart; i < midEnd && offset < sectionSize * 2; i++)
            features.RawBytes[offset++] = bytes[i];
        offset = sectionSize * 2;

        for (int i = tailStart; i < tailEnd && offset < totalFeatures; i++)
            features.RawBytes[offset++] = bytes[i];

        return features;
    }

    public static bool IsPeFile(byte[] bytes) => ByteAnalysisHelper.IsPeFile(bytes);
}

public class FileData
{
    public string FilePath { get; set; } = string.Empty;
    public FileFeatures Features { get; set; } = new FileFeatures();
    public FlashFileFeatures FlashFeatures { get; set; } = new FlashFileFeatures();
    public bool Label { get; set; }
}
