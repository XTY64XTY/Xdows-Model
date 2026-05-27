using System;
using System.IO;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace Xdows_Model_Invoker;

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

    public static bool IsPeFileHeader(ReadOnlySpan<byte> header)
    {
        if (header.Length < 2 || header[0] != 'M' || header[1] != 'Z')
            return false;

        if (header.Length < 64)
            return false;

        int peOffset = BitConverter.ToInt32(header.Slice(60, 4));
        if (peOffset + 4 > header.Length)
            return false;

        return header[peOffset] == 'P' && header[peOffset + 1] == 'E';
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
        ComputeCommonStatsSpan(bytes, byteCounts, out printableCount, out controlCount, out whitespaceCount,
            out letterCount, out digitCount, out maxZeroRun, out highByteCount,
            out zeroRunCount, out totalZeroRunLength, out _, out _, out _);
    }

    public static void ComputeCommonStatsSpan(ReadOnlySpan<byte> bytes, Span<long> byteCounts,
        out int printableCount, out int controlCount, out int whitespaceCount,
        out int letterCount, out int digitCount, out int maxZeroRun, out int highByteCount,
        out int zeroRunCount, out long totalZeroRunLength,
        out int maxNonZeroRun, out long totalNonZeroRunLength, out int nonZeroRunCount)
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
        maxNonZeroRun = 0;
        totalNonZeroRunLength = 0;
        nonZeroRunCount = 0;
        int currentZeroRun = 0;
        int currentNonZeroRun = 0;

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
                if (currentNonZeroRun > 0)
                {
                    nonZeroRunCount++;
                    totalNonZeroRunLength += currentNonZeroRun;
                    if (currentNonZeroRun > maxNonZeroRun)
                        maxNonZeroRun = currentNonZeroRun;
                    currentNonZeroRun = 0;
                }
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
                    currentZeroRun = 0;
                }
                currentNonZeroRun++;
            }
        }

        if (currentZeroRun > 0)
        {
            zeroRunCount++;
            totalZeroRunLength += currentZeroRun;
        }

        if (currentNonZeroRun > 0)
        {
            nonZeroRunCount++;
            totalNonZeroRunLength += currentNonZeroRun;
            if (currentNonZeroRun > maxNonZeroRun)
                maxNonZeroRun = currentNonZeroRun;
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

    public static void ComputeByteHistogram32(Span<long> byteCounts, int totalBytes, float[] histogram32)
    {
        for (int bin = 0; bin < 32; bin++)
        {
            long sum = 0;
            for (int j = 0; j < 8; j++)
                sum += byteCounts[bin * 8 + j];
            histogram32[bin] = totalBytes > 0 ? (float)sum / totalBytes : 0f;
        }
    }

    public static void ComputeByteRangeRatios(Span<long> byteCounts, int totalBytes,
        out double lowByteRatio, out double printableAsciiRatio, out double extendedAsciiRatio)
    {
        if (totalBytes == 0)
        {
            lowByteRatio = printableAsciiRatio = extendedAsciiRatio = 0;
            return;
        }

        long lowBytes = 0;
        for (int i = 0x00; i <= 0x1F; i++) lowBytes += byteCounts[i];

        long printableAscii = 0;
        for (int i = 0x20; i <= 0x7E; i++) printableAscii += byteCounts[i];

        long extendedAscii = 0;
        for (int i = 0x80; i <= 0xFF; i++) extendedAscii += byteCounts[i];

        lowByteRatio = (double)lowBytes / totalBytes;
        printableAsciiRatio = (double)printableAscii / totalBytes;
        extendedAsciiRatio = (double)extendedAscii / totalBytes;
    }

    public static void ComputeBlockEntropyStats(byte[] regionBytes, int blockSize, int maxRegionSize,
        out double minEntropy, out double maxEntropy, out double meanEntropy, out double variance)
    {
        minEntropy = double.MaxValue;
        maxEntropy = double.MinValue;
        meanEntropy = 0;
        variance = 0;

        int analysisLen = Math.Min(regionBytes.Length, maxRegionSize);
        if (analysisLen <= 0) { minEntropy = maxEntropy = 0; return; }

        int numBlocks = (analysisLen + blockSize - 1) / blockSize;
        var blockCounts = new long[256];
        double totalEntropy = 0;
        double totalEntropySq = 0;

        for (int blockIdx = 0; blockIdx < numBlocks; blockIdx++)
        {
            int start = blockIdx * blockSize;
            int end = Math.Min(start + blockSize, analysisLen);
            int currentBlockSize = end - start;

            blockCounts.AsSpan().Clear();
            for (int i = start; i < end; i++)
                blockCounts[regionBytes[i]]++;

            double blockEnt = ComputeEntropy(blockCounts.AsSpan(0, 256), currentBlockSize);

            if (blockEnt < minEntropy) minEntropy = blockEnt;
            if (blockEnt > maxEntropy) maxEntropy = blockEnt;
            totalEntropy += blockEnt;
            totalEntropySq += blockEnt * blockEnt;
        }

        meanEntropy = totalEntropy / numBlocks;
        double var = (totalEntropySq / numBlocks) - (meanEntropy * meanEntropy);
        variance = var < 0 ? 0 : var;
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

public class FlashFileFeatures
{
    public const int FeatureCount = 68;

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

    public float[] ByteHistogram32 { get; set; } = new float[32];

    public double LowByteRatio { get; set; }
    public double PrintableAsciiRatio { get; set; }
    public double ExtendedAsciiRatio { get; set; }
    public int MaxNonZeroByteRun { get; set; }
    public double MeanNonZeroRunLength { get; set; }

    public double HeadBlockEntropyMin { get; set; }
    public double HeadBlockEntropyMax { get; set; }
    public double HeadBlockEntropyMean { get; set; }
    public double HeadBlockEntropyVar { get; set; }
    public double TailBlockEntropyMin { get; set; }
    public double TailBlockEntropyMax { get; set; }
    public double TailBlockEntropyMean { get; set; }
    public double TailBlockEntropyVar { get; set; }

    public short PeNumberOfSections { get; set; }
    public uint PeTimeDateStamp { get; set; }
    public ushort PeCharacteristics { get; set; }
    public uint PeSizeOfHeaders { get; set; }
    public ushort PeOptionalMagic { get; set; }

    public float[] ToFloatArray()
    {
        var features = new float[FeatureCount];
        int idx = 0;

        features[idx++] = (float)Math.Log(FileSize + 1);
        features[idx++] = (float)Entropy;
        features[idx++] = (float)ZeroByteRatio;
        features[idx++] = (float)HighEntropyRatio;
        features[idx++] = (float)PrintableCharRatio;
        features[idx++] = (float)ControlCharRatio;
        features[idx++] = (float)WhitespaceRatio;
        features[idx++] = (float)LetterRatio;
        features[idx++] = (float)DigitRatio;
        features[idx++] = UniqueBytes;
        features[idx++] = (float)MostCommonByteRatio;
        features[idx++] = MaxZeroByteRun;
        features[idx++] = (float)MeanByteValue;
        features[idx++] = (float)ByteValueVariance;
        features[idx++] = (float)ByteDistributionSkewness;
        features[idx++] = (float)ByteDistributionKurtosis;
        features[idx++] = (float)MeanZeroRunLength;
        features[idx++] = ZeroRunCount;

        for (int i = 0; i < 32; i++)
            features[idx++] = ByteHistogram32[i];

        features[idx++] = (float)LowByteRatio;
        features[idx++] = (float)PrintableAsciiRatio;
        features[idx++] = (float)ExtendedAsciiRatio;
        features[idx++] = MaxNonZeroByteRun;
        features[idx++] = (float)MeanNonZeroRunLength;

        features[idx++] = (float)HeadBlockEntropyMin;
        features[idx++] = (float)HeadBlockEntropyMax;
        features[idx++] = (float)HeadBlockEntropyMean;
        features[idx++] = (float)HeadBlockEntropyVar;
        features[idx++] = (float)TailBlockEntropyMin;
        features[idx++] = (float)TailBlockEntropyMax;
        features[idx++] = (float)TailBlockEntropyMean;
        features[idx++] = (float)TailBlockEntropyVar;

        features[idx++] = PeNumberOfSections;
        features[idx++] = (float)PeTimeDateStamp;
        features[idx++] = PeCharacteristics;
        features[idx++] = (float)PeSizeOfHeaders;
        features[idx++] = PeOptionalMagic;

        return features;
    }
}

public class FlashFeatureExtractor
{
    private const int FlashRegionSize = 512 * 1024;
    private const int PeHeaderSize = 1024;
    private const int BlockEntropyBlockSize = 4096;
    private const int BlockEntropyRegionSize = 128 * 1024;

    public static FlashFileFeatures ExtractFeatures(string filePath)
    {
        var fileInfo = new FileInfo(filePath);
        long fileSize = fileInfo.Length;

        if (fileSize == 0)
            return new FlashFileFeatures { FileSize = 0 };

        if (fileSize < 64)
            throw new NotSupportedException("文件过小，无法进行PE格式验证");

        byte[] headBuf, tailBuf;
        int headRead, tailRead;

        using (var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read, 4096, FileOptions.SequentialScan))
        {
            headBuf = new byte[(int)Math.Min(fileSize, FlashRegionSize)];
            headRead = fs.Read(headBuf, 0, headBuf.Length);
            if (headRead < 64 || !ByteAnalysisHelper.IsPeFileHeader(headBuf.AsSpan(0, headRead)))
                throw new NotSupportedException("不支持该文件类型");

            while (headRead < headBuf.Length)
            {
                int read = fs.Read(headBuf, headRead, headBuf.Length - headRead);
                if (read == 0) break;
                headRead += read;
            }
            if (headRead < headBuf.Length) Array.Resize(ref headBuf, headRead);

            tailBuf = new byte[(int)Math.Min(fileSize, FlashRegionSize)];
            long tailPos = Math.Max(0, fileSize - FlashRegionSize);
            if (tailPos >= headRead)
            {
                fs.Position = tailPos;
                tailRead = fs.Read(tailBuf, 0, tailBuf.Length);
                while (tailRead < tailBuf.Length)
                {
                    int read = fs.Read(tailBuf, tailRead, tailBuf.Length - tailRead);
                    if (read == 0) break;
                    tailRead += read;
                }
            }
            else
            {
                int overlapStart = (int)tailPos;
                int copyLen = Math.Min(headRead - overlapStart, tailBuf.Length);
                Array.Copy(headBuf, overlapStart, tailBuf, 0, copyLen);
                tailRead = copyLen;
            }
            if (tailRead < tailBuf.Length) Array.Resize(ref tailBuf, tailRead);
        }

        return ExtractFromRegions(headBuf, tailBuf, fileSize);
    }

    public static async Task<FlashFileFeatures> ExtractFeaturesAsync(string filePath)
    {
        var fileInfo = new FileInfo(filePath);
        long fileSize = fileInfo.Length;

        if (fileSize == 0)
            return new FlashFileFeatures { FileSize = 0 };

        if (fileSize < 64)
            throw new NotSupportedException("文件过小，无法进行PE格式验证");

        byte[] headBuf, tailBuf;
        int headRead, tailRead;

        using (var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read, 4096, FileOptions.SequentialScan))
        {
            headBuf = new byte[(int)Math.Min(fileSize, FlashRegionSize)];
            headRead = await fs.ReadAsync(headBuf, 0, headBuf.Length);
            if (headRead < 64 || !ByteAnalysisHelper.IsPeFileHeader(headBuf.AsSpan(0, headRead)))
                throw new NotSupportedException("不支持该文件类型");

            while (headRead < headBuf.Length)
            {
                int read = await fs.ReadAsync(headBuf, headRead, headBuf.Length - headRead);
                if (read == 0) break;
                headRead += read;
            }
            if (headRead < headBuf.Length) Array.Resize(ref headBuf, headRead);

            tailBuf = new byte[(int)Math.Min(fileSize, FlashRegionSize)];
            long tailPos = Math.Max(0, fileSize - FlashRegionSize);
            if (tailPos >= headRead)
            {
                fs.Position = tailPos;
                tailRead = await fs.ReadAsync(tailBuf, 0, tailBuf.Length);
                while (tailRead < tailBuf.Length)
                {
                    int read = await fs.ReadAsync(tailBuf, tailRead, tailBuf.Length - tailRead);
                    if (read == 0) break;
                    tailRead += read;
                }
            }
            else
            {
                int overlapStart = (int)tailPos;
                int copyLen = Math.Min(headRead - overlapStart, tailBuf.Length);
                Array.Copy(headBuf, overlapStart, tailBuf, 0, copyLen);
                tailRead = copyLen;
            }
            if (tailRead < tailBuf.Length) Array.Resize(ref tailBuf, tailRead);
        }

        return ExtractFromRegions(headBuf, tailBuf, fileSize);
    }

    public static FlashFileFeatures ExtractFromBytes(byte[] bytes)
    {
        if (bytes.Length < 64 || !ByteAnalysisHelper.IsPeFile(bytes))
            throw new NotSupportedException("不支持该文件类型");

        int headLen = Math.Min(bytes.Length, FlashRegionSize);
        var headBuf = new byte[headLen];
        Array.Copy(bytes, headBuf, headLen);

        int tailStart = Math.Max(0, bytes.Length - FlashRegionSize);
        int tailLen = bytes.Length - tailStart;
        var tailBuf = new byte[tailLen];
        Array.Copy(bytes, tailStart, tailBuf, 0, tailLen);

        return ExtractFromRegions(headBuf, tailBuf, bytes.Length);
    }

    internal static FlashFileFeatures ExtractFromRegions(byte[] headBytes, byte[] tailBytes, long actualFileSize)
    {
        var features = new FlashFileFeatures { FileSize = actualFileSize };

        if (headBytes.Length == 0)
            return features;

        Span<long> byteCounts = stackalloc long[256];

        ByteAnalysisHelper.ComputeCommonStatsSpan(headBytes, byteCounts,
            out int printableCount, out int controlCount, out int whitespaceCount,
            out int letterCount, out int digitCount, out int maxZeroRun, out int highByteCount,
            out int zeroRunCount, out long totalZeroRunLength,
            out int maxNonZeroRun, out long totalNonZeroRunLength, out int nonZeroRunCount);

        ByteAnalysisHelper.ComputeStatsSummary(byteCounts, headBytes.Length,
            out int uniqueBytes, out double mostCommonByteRatio, out double zeroByteRatio);

        ByteAnalysisHelper.ComputeByteMoments(byteCounts, headBytes.Length,
            out double meanByteValue, out double byteValueVariance,
            out double skewness, out double kurtosis);

        ByteAnalysisHelper.ComputeByteHistogram32(byteCounts, headBytes.Length, features.ByteHistogram32);
        ByteAnalysisHelper.ComputeByteRangeRatios(byteCounts, headBytes.Length,
            out double lowByteRatio, out double printableAsciiRatio, out double extendedAsciiRatio);

        features.UniqueBytes = uniqueBytes;
        features.MostCommonByteRatio = mostCommonByteRatio;
        features.ZeroByteRatio = zeroByteRatio;
        features.HighEntropyRatio = (double)highByteCount / headBytes.Length;
        features.Entropy = ByteAnalysisHelper.ComputeEntropy(byteCounts, headBytes.Length);
        features.PrintableCharRatio = (double)printableCount / headBytes.Length;
        features.ControlCharRatio = (double)controlCount / headBytes.Length;
        features.WhitespaceRatio = (double)whitespaceCount / headBytes.Length;
        features.LetterRatio = (double)letterCount / headBytes.Length;
        features.DigitRatio = (double)digitCount / headBytes.Length;
        features.MaxZeroByteRun = maxZeroRun;
        features.MeanByteValue = meanByteValue;
        features.ByteValueVariance = byteValueVariance;
        features.ByteDistributionSkewness = skewness;
        features.ByteDistributionKurtosis = kurtosis;
        features.MeanZeroRunLength = zeroRunCount > 0 ? (double)totalZeroRunLength / zeroRunCount : 0;
        features.ZeroRunCount = zeroRunCount;

        features.LowByteRatio = lowByteRatio;
        features.PrintableAsciiRatio = printableAsciiRatio;
        features.ExtendedAsciiRatio = extendedAsciiRatio;
        features.MaxNonZeroByteRun = maxNonZeroRun;
        features.MeanNonZeroRunLength = nonZeroRunCount > 0 ? (double)totalNonZeroRunLength / nonZeroRunCount : 0;

        ByteAnalysisHelper.ComputeBlockEntropyStats(headBytes, BlockEntropyBlockSize, BlockEntropyRegionSize,
            out double hMin, out double hMax, out double hMean, out double hVar);
        features.HeadBlockEntropyMin = hMin;
        features.HeadBlockEntropyMax = hMax;
        features.HeadBlockEntropyMean = hMean;
        features.HeadBlockEntropyVar = hVar;

        if (tailBytes.Length > 0 && !ReferenceEquals(tailBytes, headBytes))
        {
            ByteAnalysisHelper.ComputeBlockEntropyStats(tailBytes, BlockEntropyBlockSize, BlockEntropyRegionSize,
                out double tMin, out double tMax, out double tMean, out double tVar);
            features.TailBlockEntropyMin = tMin;
            features.TailBlockEntropyMax = tMax;
            features.TailBlockEntropyMean = tMean;
            features.TailBlockEntropyVar = tVar;
        }
        else
        {
            features.TailBlockEntropyMin = hMin;
            features.TailBlockEntropyMax = hMax;
            features.TailBlockEntropyMean = hMean;
            features.TailBlockEntropyVar = hVar;
        }

        ParsePeHeader(headBytes, features);

        return features;
    }

    private static void ParsePeHeader(byte[] headerBytes, FlashFileFeatures features)
    {
        features.PeNumberOfSections = 0;
        features.PeTimeDateStamp = 0;
        features.PeCharacteristics = 0;
        features.PeSizeOfHeaders = 0;
        features.PeOptionalMagic = 0;

        if (headerBytes.Length < 64) return;

        int peOffset = BitConverter.ToInt32(headerBytes, 60);
        if (peOffset < 0 || peOffset + 24 > headerBytes.Length || headerBytes[peOffset] != 'P' || headerBytes[peOffset + 1] != 'E')
            return;

        features.PeNumberOfSections = BitConverter.ToInt16(headerBytes, peOffset + 6);
        features.PeTimeDateStamp = BitConverter.ToUInt32(headerBytes, peOffset + 8);
        features.PeCharacteristics = BitConverter.ToUInt16(headerBytes, peOffset + 22);

        int optHeaderOffset = peOffset + 24;
        if (optHeaderOffset + 2 > headerBytes.Length) return;

        features.PeOptionalMagic = BitConverter.ToUInt16(headerBytes, optHeaderOffset);

        bool isPe32 = features.PeOptionalMagic == 0x10b;
        int sizeOfHeadersFieldOffset = isPe32 ? optHeaderOffset + 60 : optHeaderOffset + 84;

        if (sizeOfHeadersFieldOffset + 4 <= headerBytes.Length)
            features.PeSizeOfHeaders = BitConverter.ToUInt32(headerBytes, sizeOfHeadersFieldOffset);
    }

    public static bool IsPeFile(byte[] bytes) => ByteAnalysisHelper.IsPeFile(bytes);

    public static bool IsPeFileFromPath(string filePath)
    {
        try
        {
            var fileInfo = new FileInfo(filePath);
            if (fileInfo.Length < 64)
                return false;

            byte[] header = new byte[PeHeaderSize];
            using var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read, PeHeaderSize, FileOptions.SequentialScan);
            int read = fs.Read(header, 0, Math.Min((int)fileInfo.Length, PeHeaderSize));
            if (read < 64)
                return false;

            return ByteAnalysisHelper.IsPeFileHeader(header.AsSpan(0, read));
        }
        catch
        {
            return false;
        }
    }
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
