using System;
using System.IO;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using Xdows_Model_Config;

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

    internal static readonly byte[] ByteClass = CreateByteClass();

    private static byte[] CreateByteClass()
    {
        var classes = new byte[256];
        for (int i = 0; i < 256; i++)
        {
            byte b = (byte)i;
            byte c = 0;
            if (b >= 0x80) c |= 0x01;
            if (b == 9 || b == 10 || b == 13 || b == 32) c |= 0x02;
            if (b >= 32 && b <= 126)
            {
                c |= 0x04;
                if ((b >= 65 && b <= 90) || (b >= 97 && b <= 122)) c |= 0x08;
                else if (b >= 48 && b <= 57) c |= 0x10;
            }
            classes[i] = c;
        }
        return classes;
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

        ReadOnlySpan<byte> clsSpan = ByteClass;
        for (int i = 0; i < bytes.Length; i++)
        {
            byte b = bytes[i];
            byteCounts[b]++;

            byte cls = clsSpan[b];
            highByteCount += cls & 1;
            whitespaceCount += (cls >> 1) & 1;
            int printable = (cls >> 2) & 1;
            printableCount += printable;
            controlCount += printable ^ 1;
            letterCount += (cls >> 3) & printable;
            digitCount += (cls >> 4) & printable;

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
                entropy -= p * Math.Log2(p);
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

    public static unsafe double ComputeRegionEntropy(ReadOnlySpan<byte> bytes, int start, int length)
    {
        int actualLength = Math.Min(length, bytes.Length - start);
        if (actualLength <= 0) return 0;

        Span<long> counts = stackalloc long[256];
        counts.Clear();

        fixed (byte* pBytes = bytes)
        {
            byte* p = pBytes + start;
            byte* end = p + actualLength;
            while (p < end)
                counts[*p++]++;
        }

        return ComputeEntropy(counts, actualLength);
    }

    public static void ComputeByteHistogram32(Span<long> byteCounts, int totalBytes, Span<float> histogram32)
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

    public static void ComputeBlockEntropyStats(ReadOnlySpan<byte> regionBytes, int blockSize, int maxRegionSize,
        out double minEntropy, out double maxEntropy, out double meanEntropy, out double variance)
    {
        minEntropy = double.MaxValue;
        maxEntropy = double.MinValue;
        meanEntropy = 0;
        variance = 0;

        int analysisLen = Math.Min(regionBytes.Length, maxRegionSize);
        if (analysisLen <= 0) { minEntropy = maxEntropy = 0; return; }

        int numBlocks = (analysisLen + blockSize - 1) / blockSize;
        Span<long> blockCounts = stackalloc long[256];
        double totalEntropy = 0;
        double totalEntropySq = 0;

        for (int blockIdx = 0; blockIdx < numBlocks; blockIdx++)
        {
            int start = blockIdx * blockSize;
            int end = Math.Min(start + blockSize, analysisLen);
            int currentBlockSize = end - start;

            blockCounts.Clear();
            unsafe
            {
                fixed (byte* pBytes = regionBytes)
                fixed (long* pCounts = blockCounts)
                {
                    byte* p = pBytes + start;
                    byte* endP = p + currentBlockSize;
                    while (p < endP)
                        pCounts[*p++]++;
                }
            }

            double blockEnt = ComputeEntropy(blockCounts, currentBlockSize);

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
            out int zeroRunCount, out long totalZeroRunLength,
            out int maxNonZeroRun, out long totalNonZeroRunLength, out int nonZeroRunCount);

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

        ByteAnalysisHelper.ComputeByteMoments(byteCounts, bytes.Length,
            out double meanByteValue, out double byteValueVariance,
            out double skewness, out double kurtosis);

        ByteAnalysisHelper.ComputeByteRangeRatios(byteCounts, bytes.Length,
            out double lowByteRatio, out double printableAsciiRatio, out double extendedAsciiRatio);

        ByteAnalysisHelper.ComputeBlockEntropyStats(bytes, 4096, 128 * 1024,
            out double headBlockEntropyMin, out double headBlockEntropyMax,
            out double headBlockEntropyMean, out double headBlockEntropyVar);

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

        ParsePeHeader(bytes, features);

        features.HeadBlockEntropyMin = headBlockEntropyMin;
        features.HeadBlockEntropyMax = headBlockEntropyMax;
        features.HeadBlockEntropyMean = headBlockEntropyMean;
        features.HeadBlockEntropyVar = headBlockEntropyVar;

        return features;
    }

    internal static void ExtractBlockEntropyOptimized(byte[] bytes, FileFeatures features)
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

        Span<long> blockByteCounts = stackalloc long[256];

        for (int blockIdx = 0; blockIdx < numBlocks; blockIdx++)
        {
            int start = blockIdx * blockSize;
            int end = Math.Min(start + blockSize, bytes.Length);
            int currentBlockSize = end - start;

            blockByteCounts.Clear();

            unsafe
            {
                fixed (byte* pBytes = bytes)
                fixed (long* pCounts = blockByteCounts)
                {
                    byte* p = pBytes + start;
                    byte* endP = p + currentBlockSize;
                    while (p < endP)
                        pCounts[*p++]++;
                }
            }

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

    internal static void ParsePeHeader(ReadOnlySpan<byte> headerBytes, FileFeatures features)
    {
        features.PeNumberOfSections = 0;
        features.PeTimeDateStamp = 0;
        features.PeCharacteristics = 0;
        features.PeSizeOfHeaders = 0;
        features.PeOptionalMagic = 0;

        if (headerBytes.Length < 64) return;

        int peOffset = BitConverter.ToInt32(headerBytes.Slice(60, 4));
        if (peOffset < 0 || peOffset + 24 > headerBytes.Length || headerBytes[peOffset] != 'P' || headerBytes[peOffset + 1] != 'E')
            return;

        features.PeNumberOfSections = BitConverter.ToInt16(headerBytes.Slice(peOffset + 6, 2));
        features.PeTimeDateStamp = BitConverter.ToUInt32(headerBytes.Slice(peOffset + 8, 4));
        features.PeCharacteristics = BitConverter.ToUInt16(headerBytes.Slice(peOffset + 22, 2));

        int optHeaderOffset = peOffset + 24;
        if (optHeaderOffset + 2 > headerBytes.Length) return;

        features.PeOptionalMagic = BitConverter.ToUInt16(headerBytes.Slice(optHeaderOffset, 2));

        bool isPe32 = features.PeOptionalMagic == 0x10b;
        int sizeOfHeadersFieldOffset = isPe32 ? optHeaderOffset + 60 : optHeaderOffset + 84;

        if (sizeOfHeadersFieldOffset + 4 <= headerBytes.Length)
            features.PeSizeOfHeaders = BitConverter.ToUInt32(headerBytes.Slice(sizeOfHeadersFieldOffset, 4));
    }

    public static bool IsPeFile(byte[] bytes) => ByteAnalysisHelper.IsPeFile(bytes);
}

public class FlashFileFeatures
{
    public const int FeatureCount = FeatureSchema.FlashFeatureCount;

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

    public void WriteTo(Span<float> destination)
    {
        int idx = 0;

        destination[idx++] = (float)Math.Log(FileSize + 1);
        destination[idx++] = (float)Entropy;
        destination[idx++] = (float)ZeroByteRatio;
        destination[idx++] = (float)HighEntropyRatio;
        destination[idx++] = (float)PrintableCharRatio;
        destination[idx++] = (float)ControlCharRatio;
        destination[idx++] = (float)WhitespaceRatio;
        destination[idx++] = (float)LetterRatio;
        destination[idx++] = (float)DigitRatio;
        destination[idx++] = UniqueBytes;
        destination[idx++] = (float)MostCommonByteRatio;
        destination[idx++] = MaxZeroByteRun;
        destination[idx++] = (float)MeanByteValue;
        destination[idx++] = (float)ByteValueVariance;
        destination[idx++] = (float)ByteDistributionSkewness;
        destination[idx++] = (float)ByteDistributionKurtosis;
        destination[idx++] = (float)MeanZeroRunLength;
        destination[idx++] = ZeroRunCount;

        for (int i = 0; i < 32; i++)
            destination[idx++] = ByteHistogram32[i];

        destination[idx++] = (float)LowByteRatio;
        destination[idx++] = (float)PrintableAsciiRatio;
        destination[idx++] = (float)ExtendedAsciiRatio;
        destination[idx++] = MaxNonZeroByteRun;
        destination[idx++] = (float)MeanNonZeroRunLength;

        destination[idx++] = (float)HeadBlockEntropyMin;
        destination[idx++] = (float)HeadBlockEntropyMax;
        destination[idx++] = (float)HeadBlockEntropyMean;
        destination[idx++] = (float)HeadBlockEntropyVar;
        destination[idx++] = (float)TailBlockEntropyMin;
        destination[idx++] = (float)TailBlockEntropyMax;
        destination[idx++] = (float)TailBlockEntropyMean;
        destination[idx++] = (float)TailBlockEntropyVar;

        destination[idx++] = PeNumberOfSections;
        destination[idx++] = (float)PeTimeDateStamp;
        destination[idx++] = PeCharacteristics;
        destination[idx++] = (float)PeSizeOfHeaders;
        destination[idx++] = PeOptionalMagic;
    }
}

public class FlashFeatureExtractor
{
    internal const int FlashRegionSize = 512 * 1024;
    private const int PeHeaderSize = 1024;
    internal const int BlockEntropyBlockSize = 4096;
    internal const int BlockEntropyRegionSize = 128 * 1024;

    public static FlashFileFeatures ExtractFeatures(string filePath)
    {
        var fileInfo = new FileInfo(filePath);
        long fileSize = fileInfo.Length;

        if (fileSize == 0)
            return new FlashFileFeatures { FileSize = 0 };

        if (fileSize < 64)
            throw new NotSupportedException("文件过小，无法进行PE格式验证");

        byte[] headBuf;
        int headRead;

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

            if (fileSize <= FlashRegionSize)
                return ExtractFromRegions(headBuf.AsSpan(0, headRead), headBuf.AsSpan(0, headRead), fileSize);

            var tailBuf = new byte[FlashRegionSize];
            long tailPos = fileSize - FlashRegionSize;
            fs.Position = tailPos;
            int tailRead = fs.Read(tailBuf, 0, tailBuf.Length);
            while (tailRead < tailBuf.Length)
            {
                int read = fs.Read(tailBuf, tailRead, tailBuf.Length - tailRead);
                if (read == 0) break;
                tailRead += read;
            }
            if (tailRead < tailBuf.Length) Array.Resize(ref tailBuf, tailRead);

            return ExtractFromRegions(headBuf.AsSpan(0, headRead), tailBuf.AsSpan(0, tailRead), fileSize);
        }
    }

    public static async Task<FlashFileFeatures> ExtractFeaturesAsync(string filePath)
    {
        var fileInfo = new FileInfo(filePath);
        long fileSize = fileInfo.Length;

        if (fileSize == 0)
            return new FlashFileFeatures { FileSize = 0 };

        if (fileSize < 64)
            throw new NotSupportedException("文件过小，无法进行PE格式验证");

        byte[] headBuf;
        int headRead;

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

            if (fileSize <= FlashRegionSize)
                return ExtractFromRegions(headBuf.AsSpan(0, headRead), headBuf.AsSpan(0, headRead), fileSize);

            var tailBuf = new byte[FlashRegionSize];
            long tailPos = fileSize - FlashRegionSize;
            fs.Position = tailPos;
            int tailRead = await fs.ReadAsync(tailBuf, 0, tailBuf.Length);
            while (tailRead < tailBuf.Length)
            {
                int read = await fs.ReadAsync(tailBuf, tailRead, tailBuf.Length - tailRead);
                if (read == 0) break;
                tailRead += read;
            }
            if (tailRead < tailBuf.Length) Array.Resize(ref tailBuf, tailRead);

            return ExtractFromRegions(headBuf.AsSpan(0, headRead), tailBuf.AsSpan(0, tailRead), fileSize);
        }
    }

    public static FlashFileFeatures ExtractFromBytes(byte[] bytes)
    {
        if (bytes.Length < 64 || !ByteAnalysisHelper.IsPeFile(bytes))
            throw new NotSupportedException("不支持该文件类型");

        if (bytes.Length <= FlashRegionSize)
            return ExtractFromRegions(bytes, bytes, bytes.Length);

        ReadOnlySpan<byte> headSpan = bytes.AsSpan(0, FlashRegionSize);
        ReadOnlySpan<byte> tailSpan = bytes.AsSpan(bytes.Length - FlashRegionSize, FlashRegionSize);
        return ExtractFromRegions(headSpan, tailSpan, bytes.Length);
    }

    internal static FlashFileFeatures ExtractFromRegions(ReadOnlySpan<byte> headBytes, ReadOnlySpan<byte> tailBytes, long actualFileSize)
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

        if (tailBytes.Length > 0 && tailBytes != headBytes)
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

    internal static void ParsePeHeader(ReadOnlySpan<byte> headerBytes, FlashFileFeatures features)
    {
        features.PeNumberOfSections = 0;
        features.PeTimeDateStamp = 0;
        features.PeCharacteristics = 0;
        features.PeSizeOfHeaders = 0;
        features.PeOptionalMagic = 0;

        if (headerBytes.Length < 64) return;

        int peOffset = BitConverter.ToInt32(headerBytes.Slice(60, 4));
        if (peOffset < 0 || peOffset + 24 > headerBytes.Length || headerBytes[peOffset] != 'P' || headerBytes[peOffset + 1] != 'E')
            return;

        features.PeNumberOfSections = BitConverter.ToInt16(headerBytes.Slice(peOffset + 6, 2));
        features.PeTimeDateStamp = BitConverter.ToUInt32(headerBytes.Slice(peOffset + 8, 4));
        features.PeCharacteristics = BitConverter.ToUInt16(headerBytes.Slice(peOffset + 22, 2));

        int optHeaderOffset = peOffset + 24;
        if (optHeaderOffset + 2 > headerBytes.Length) return;

        features.PeOptionalMagic = BitConverter.ToUInt16(headerBytes.Slice(optHeaderOffset, 2));

        bool isPe32 = features.PeOptionalMagic == 0x10b;
        int sizeOfHeadersFieldOffset = isPe32 ? optHeaderOffset + 60 : optHeaderOffset + 84;

        if (sizeOfHeadersFieldOffset + 4 <= headerBytes.Length)
            features.PeSizeOfHeaders = BitConverter.ToUInt32(headerBytes.Slice(sizeOfHeadersFieldOffset, 4));
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
    public const int FeatureCount = FeatureSchema.StandardFeatureCount;

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

    public double MeanByteValue { get; set; }
    public double ByteValueVariance { get; set; }
    public double ByteDistributionSkewness { get; set; }
    public double ByteDistributionKurtosis { get; set; }
    public double MeanZeroRunLength { get; set; }
    public int ZeroRunCount { get; set; }
    public double LowByteRatio { get; set; }
    public double PrintableAsciiRatio { get; set; }
    public double ExtendedAsciiRatio { get; set; }
    public int MaxNonZeroByteRun { get; set; }
    public double MeanNonZeroRunLength { get; set; }
    public short PeNumberOfSections { get; set; }
    public uint PeTimeDateStamp { get; set; }
    public ushort PeCharacteristics { get; set; }
    public uint PeSizeOfHeaders { get; set; }
    public ushort PeOptionalMagic { get; set; }
    public double HeadBlockEntropyMin { get; set; }
    public double HeadBlockEntropyMax { get; set; }
    public double HeadBlockEntropyMean { get; set; }
    public double HeadBlockEntropyVar { get; set; }

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
        features[idx++] = (float)MeanByteValue;
        features[idx++] = (float)ByteValueVariance;
        features[idx++] = (float)ByteDistributionSkewness;
        features[idx++] = (float)ByteDistributionKurtosis;
        features[idx++] = (float)MeanZeroRunLength;
        features[idx++] = ZeroRunCount;
        features[idx++] = (float)LowByteRatio;
        features[idx++] = (float)PrintableAsciiRatio;
        features[idx++] = (float)ExtendedAsciiRatio;
        features[idx++] = MaxNonZeroByteRun;
        features[idx++] = (float)MeanNonZeroRunLength;
        features[idx++] = PeNumberOfSections;
        features[idx++] = (float)PeTimeDateStamp;
        features[idx++] = PeCharacteristics;
        features[idx++] = (float)PeSizeOfHeaders;
        features[idx++] = PeOptionalMagic;
        features[idx++] = (float)HeadBlockEntropyMin;
        features[idx++] = (float)HeadBlockEntropyMax;
        features[idx++] = (float)HeadBlockEntropyMean;
        features[idx++] = (float)HeadBlockEntropyVar;

        return features;
    }

    public void WriteTo(Span<float> destination)
    {
        int idx = 0;

        for (int i = 0; i < 256; i++)
            destination[idx++] = (float)ByteFrequency[i];

        destination[idx++] = (float)Math.Log(FileSize + 1);
        destination[idx++] = (float)Entropy;
        destination[idx++] = (float)MinBlockEntropy;
        destination[idx++] = (float)MaxBlockEntropy;
        destination[idx++] = (float)MeanBlockEntropy;
        destination[idx++] = (float)BlockEntropyVariance;
        destination[idx++] = (float)MinEntropyBlockPosition;
        destination[idx++] = (float)MaxEntropyBlockPosition;
        destination[idx++] = (float)FirstBlockEntropy;
        destination[idx++] = (float)LastBlockEntropy;
        destination[idx++] = UniqueBytes;
        destination[idx++] = MostCommonByte;
        destination[idx++] = (float)MostCommonByteRatio;
        destination[idx++] = LeastCommonByte;
        destination[idx++] = (float)LeastCommonByteRatio;
        destination[idx++] = (float)PrintableCharRatio;
        destination[idx++] = (float)ControlCharRatio;
        destination[idx++] = (float)WhitespaceRatio;
        destination[idx++] = (float)LetterRatio;
        destination[idx++] = (float)DigitRatio;
        destination[idx++] = MaxZeroByteRun;
        destination[idx++] = (float)ZeroByteRatio;
        destination[idx++] = (float)HighEntropyRatio;
        destination[idx++] = (float)MeanByteValue;
        destination[idx++] = (float)ByteValueVariance;
        destination[idx++] = (float)ByteDistributionSkewness;
        destination[idx++] = (float)ByteDistributionKurtosis;
        destination[idx++] = (float)MeanZeroRunLength;
        destination[idx++] = ZeroRunCount;
        destination[idx++] = (float)LowByteRatio;
        destination[idx++] = (float)PrintableAsciiRatio;
        destination[idx++] = (float)ExtendedAsciiRatio;
        destination[idx++] = MaxNonZeroByteRun;
        destination[idx++] = (float)MeanNonZeroRunLength;
        destination[idx++] = PeNumberOfSections;
        destination[idx++] = (float)PeTimeDateStamp;
        destination[idx++] = PeCharacteristics;
        destination[idx++] = (float)PeSizeOfHeaders;
        destination[idx++] = PeOptionalMagic;
        destination[idx++] = (float)HeadBlockEntropyMin;
        destination[idx++] = (float)HeadBlockEntropyMax;
        destination[idx++] = (float)HeadBlockEntropyMean;
        destination[idx++] = (float)HeadBlockEntropyVar;
    }
}

public class ProRawStatFeatures
{
    public const int FeaturesPerSection = 40;
    public const int SectionCount = 3;
    public const int TotalCount = FeatureSchema.ProRawStatCount;
    public const int SectionSize = 512;

    public float[] Features { get; } = new float[TotalCount];

    public float[] ToFloatArray()
    {
        var result = new float[TotalCount];
        Array.Copy(Features, result, TotalCount);
        return result;
    }

    public void WriteTo(Span<float> destination)
    {
        Features.AsSpan().CopyTo(destination);
    }
}

public static class ProRawStatExtractor
{
    public static ProRawStatFeatures ExtractFromBytes(byte[] bytes)
    {
        var result = new ProRawStatFeatures();
        int fileSize = bytes.Length;
        int sectionSize = ProRawStatFeatures.SectionSize;

        int headLength = Math.Min(sectionSize, fileSize);
        ExtractSectionStats(bytes, 0, headLength, fileSize, result.Features, 0);

        int midStart = Math.Max(0, fileSize / 2 - sectionSize / 2);
        int midLength = Math.Min(sectionSize, fileSize - midStart);
        ExtractSectionStats(bytes, midStart, midLength, fileSize, result.Features, ProRawStatFeatures.FeaturesPerSection);

        int tailStart = Math.Max(0, fileSize - sectionSize);
        int tailLength = fileSize - tailStart;
        ExtractSectionStats(bytes, tailStart, tailLength, fileSize, result.Features, ProRawStatFeatures.FeaturesPerSection * 2);

        return result;
    }

    private static void ExtractSectionStats(byte[] bytes, int start, int length, int fileSize, float[] destination, int offset)
    {
        if (length <= 0 || start >= bytes.Length)
            return;

        int actualLength = Math.Min(length, bytes.Length - start);
        if (actualLength <= 0)
            return;

        Span<long> byteCounts = stackalloc long[256];
        ByteAnalysisHelper.ComputeCommonStatsSpan(bytes.AsSpan(start, actualLength), byteCounts,
            out int printableCount, out _, out _, out int letterCount, out int digitCount,
            out int maxZeroRun, out int highByteCount, out _, out _, out _, out _, out _);

        destination[offset + 0] = (float)ByteAnalysisHelper.ComputeEntropy(byteCounts, actualLength);
        ByteAnalysisHelper.ComputeByteHistogram32(byteCounts, actualLength, destination.AsSpan(offset + 1, 32));

        destination[offset + 33] = (float)printableCount / actualLength;
        destination[offset + 34] = (float)byteCounts[0] / actualLength;
        destination[offset + 35] = (float)highByteCount / actualLength;
        destination[offset + 36] = (float)letterCount / actualLength;
        destination[offset + 37] = (float)digitCount / actualLength;
        destination[offset + 38] = (float)maxZeroRun / actualLength;
        destination[offset + 39] = fileSize > 0 ? (float)actualLength / fileSize : 0f;
    }
}

public class ProHybridFileFeatures
{
    public const int StructuralFeatureCount = FeatureSchema.ProStructuralCount;
    public const int RawStatFeatureCount = ProRawStatFeatures.TotalCount;
    public const int FixedFeatureCount = FileFeatures.FeatureCount + FlashFileFeatures.FeatureCount + StructuralFeatureCount;
    public static int FeatureCount => FeatureSchema.ProHybridFeatureCount;

    public int FeatureLength { get; }
    public float[] Features { get; }

    public ProHybridFileFeatures()
    {
        FeatureLength = FeatureCount;
        Features = new float[FeatureLength];
    }

    public static bool IsProFeatureCount(int featureCount)
    {
        return featureCount == FeatureCount;
    }

    public float[] ToFloatArray()
    {
        var result = new float[FeatureLength];
        Array.Copy(Features, result, FeatureLength);
        return result;
    }
}

public static class ProHybridFeatureExtractor
{
    public static ProHybridFileFeatures ExtractFeatures(string filePath)
    {
        var bytes = File.ReadAllBytes(filePath);
        return ExtractFromBytes(bytes);
    }

    public static ProHybridFileFeatures ExtractFromBytes(byte[] bytes)
    {
        if (bytes.Length == 0)
            throw new NotSupportedException("文件为空");

        if (bytes.Length < 64 || !ByteAnalysisHelper.IsPeFile(bytes))
            throw new NotSupportedException("不支持该文件类型");

        var result = new ProHybridFileFeatures();
        var span = result.Features.AsSpan();
        int idx = 0;

        FeatureExtractor.ExtractFromBytes(bytes).WriteTo(span.Slice(idx, FileFeatures.FeatureCount));
        idx += FileFeatures.FeatureCount;

        FlashFeatureExtractor.ExtractFromBytes(bytes).WriteTo(span.Slice(idx, FlashFileFeatures.FeatureCount));
        idx += FlashFileFeatures.FeatureCount;

        ProRawStatExtractor.ExtractFromBytes(bytes).WriteTo(span.Slice(idx, ProRawStatFeatures.TotalCount));
        idx += ProRawStatFeatures.TotalCount;

        ExtractStructuralFeatures(bytes).AsSpan().CopyTo(span.Slice(idx, ProHybridFileFeatures.StructuralFeatureCount));

        return result;
    }


    private static double ComputeSectionEntropy(byte[] bytes, int start, int length)
    {
        const int MaxSectionEntropyBytes = 128 * 1024;
        int actualLength = Math.Min(length, MaxSectionEntropyBytes);
        if (actualLength <= 0)
            return 0;

        return ByteAnalysisHelper.ComputeRegionEntropy(bytes, start, actualLength);
    }

    public static float[] ExtractStructuralFeatures(byte[] bytes)
    {
        var features = new float[ProHybridFileFeatures.StructuralFeatureCount];
        if (!TryReadPeLayout(bytes, out var layout))
            return features;

        int sectionCount = Math.Max(0, layout.SectionCount);
        int parsedSections = 0;
        int executableCount = 0;
        int writableCount = 0;
        int readableCount = 0;
        int codeCount = 0;
        int initializedDataCount = 0;
        int uninitializedDataCount = 0;
        int suspiciousRwxCount = 0;
        int zeroRawCount = 0;
        int entrySectionIndex = -1;
        long lastSectionEnd = 0;

        double entropySum = 0;
        double entropySquaredSum = 0;
        double minEntropy = double.MaxValue;
        double maxEntropy = 0;
        double rawSizeSum = 0;
        double maxRawSize = 0;
        double virtualSizeSum = 0;
        double maxVirtualSize = 0;
        double rawVirtualRatioSum = 0;
        double maxRawVirtualRatio = 0;

        for (int i = 0; i < sectionCount; i++)
        {
            int sectionOffset = layout.SectionTableOffset + i * 40;
            if (sectionOffset < 0 || sectionOffset + 40 > bytes.Length)
                break;

            uint virtualSize = ReadUInt32(bytes, sectionOffset + 8);
            uint virtualAddress = ReadUInt32(bytes, sectionOffset + 12);
            uint rawSize = ReadUInt32(bytes, sectionOffset + 16);
            uint rawPointer = ReadUInt32(bytes, sectionOffset + 20);
            uint characteristics = ReadUInt32(bytes, sectionOffset + 36);

            parsedSections++;

            bool executable = (characteristics & 0x20000000) != 0;
            bool readable = (characteristics & 0x40000000) != 0;
            bool writable = (characteristics & 0x80000000) != 0;

            if (executable) executableCount++;
            if (readable) readableCount++;
            if (writable) writableCount++;
            if ((characteristics & 0x00000020) != 0) codeCount++;
            if ((characteristics & 0x00000040) != 0) initializedDataCount++;
            if ((characteristics & 0x00000080) != 0) uninitializedDataCount++;
            if (executable && writable) suspiciousRwxCount++;
            if (rawSize == 0) zeroRawCount++;

            uint effectiveVirtualSize = Math.Max(virtualSize, rawSize);
            if (entrySectionIndex < 0 &&
                layout.AddressOfEntryPoint >= virtualAddress &&
                layout.AddressOfEntryPoint < virtualAddress + effectiveVirtualSize)
            {
                entrySectionIndex = i;
            }

            int availableRawSize = 0;
            if (rawPointer < bytes.Length)
                availableRawSize = (int)Math.Min(rawSize, bytes.Length - rawPointer);

            double entropy = availableRawSize > 0
                ? ComputeSectionEntropy(bytes, (int)rawPointer, availableRawSize)
                : 0;

            entropySum += entropy;
            entropySquaredSum += entropy * entropy;
            minEntropy = Math.Min(minEntropy, entropy);
            maxEntropy = Math.Max(maxEntropy, entropy);

            rawSizeSum += rawSize;
            maxRawSize = Math.Max(maxRawSize, rawSize);
            virtualSizeSum += virtualSize;
            maxVirtualSize = Math.Max(maxVirtualSize, virtualSize);

            double rawVirtualRatio = virtualSize > 0 ? (double)rawSize / virtualSize : 0;
            rawVirtualRatioSum += rawVirtualRatio;
            maxRawVirtualRatio = Math.Max(maxRawVirtualRatio, rawVirtualRatio);

            long sectionEnd = rawPointer + rawSize;
            if (sectionEnd > lastSectionEnd)
                lastSectionEnd = sectionEnd;
        }

        int denominator = Math.Max(parsedSections, 1);
        double meanEntropy = parsedSections > 0 ? entropySum / parsedSections : 0;
        double entropyVariance = parsedSections > 0
            ? Math.Max(0, entropySquaredSum / parsedSections - meanEntropy * meanEntropy)
            : 0;
        if (minEntropy == double.MaxValue)
            minEntropy = 0;

        long overlayBytes = Math.Max(0, bytes.Length - Math.Max(lastSectionEnd, layout.SizeOfHeaders));
        double overlayRatio = bytes.Length > 0 ? (double)overlayBytes / bytes.Length : 0;
        double entryRatio = layout.SizeOfImage > 0 ? (double)layout.AddressOfEntryPoint / layout.SizeOfImage : 0;

        int idx = 0;
        features[idx++] = sectionCount;
        features[idx++] = (float)executableCount / denominator;
        features[idx++] = (float)writableCount / denominator;
        features[idx++] = (float)readableCount / denominator;
        features[idx++] = (float)codeCount / denominator;
        features[idx++] = (float)initializedDataCount / denominator;
        features[idx++] = (float)uninitializedDataCount / denominator;
        features[idx++] = suspiciousRwxCount;
        features[idx++] = (float)zeroRawCount / denominator;
        features[idx++] = entrySectionIndex >= 0 && sectionCount > 0 ? (float)(entrySectionIndex + 1) / sectionCount : 0;
        features[idx++] = (float)meanEntropy;
        features[idx++] = (float)minEntropy;
        features[idx++] = (float)maxEntropy;
        features[idx++] = (float)entropyVariance;
        features[idx++] = (float)Math.Log(rawSizeSum / denominator + 1);
        features[idx++] = (float)Math.Log(maxRawSize + 1);
        features[idx++] = (float)Math.Log(virtualSizeSum / denominator + 1);
        features[idx++] = (float)Math.Log(maxVirtualSize + 1);
        features[idx++] = (float)(rawVirtualRatioSum / denominator);
        features[idx++] = (float)maxRawVirtualRatio;
        features[idx++] = (float)Math.Log(layout.SizeOfImage + 1);
        features[idx++] = (float)Math.Log(layout.SizeOfCode + 1);
        features[idx++] = (float)Math.Log(layout.SizeOfInitializedData + 1);
        features[idx++] = (float)Math.Log(layout.SizeOfUninitializedData + 1);
        features[idx++] = layout.Subsystem;
        features[idx++] = layout.DllCharacteristics;
        features[idx++] = layout.Characteristics;
        features[idx++] = bytes.Length > 0 ? (float)layout.PeOffset / bytes.Length : 0;
        features[idx++] = bytes.Length > 0 ? (float)layout.SizeOfHeaders / bytes.Length : 0;
        features[idx++] = (float)entryRatio;
        features[idx++] = overlayBytes > 0 ? 1 : 0;
        features[idx++] = (float)overlayRatio;

        return features;
    }

    private static bool TryReadPeLayout(byte[] bytes, out PeLayout layout)
    {
        layout = default;
        if (bytes.Length < 64)
            return false;

        int peOffset = BitConverter.ToInt32(bytes, 60);
        if (peOffset < 0 || peOffset + 24 > bytes.Length || bytes[peOffset] != 'P' || bytes[peOffset + 1] != 'E')
            return false;

        ushort characteristics = ReadUInt16(bytes, peOffset + 22);
        ushort sectionCount = ReadUInt16(bytes, peOffset + 6);
        ushort optionalHeaderSize = ReadUInt16(bytes, peOffset + 20);
        int optionalHeaderOffset = peOffset + 24;
        int sectionTableOffset = optionalHeaderOffset + optionalHeaderSize;

        if (optionalHeaderOffset + 2 > bytes.Length)
            return false;

        ushort magic = ReadUInt16(bytes, optionalHeaderOffset);
        bool pe32 = magic == 0x10b;
        uint addressOfEntryPoint = ReadUInt32(bytes, optionalHeaderOffset + 16);
        uint sizeOfCode = ReadUInt32(bytes, optionalHeaderOffset + 4);
        uint sizeOfInitializedData = ReadUInt32(bytes, optionalHeaderOffset + 8);
        uint sizeOfUninitializedData = ReadUInt32(bytes, optionalHeaderOffset + 12);
        uint sizeOfImage = ReadUInt32(bytes, optionalHeaderOffset + 56);
        uint sizeOfHeaders = ReadUInt32(bytes, optionalHeaderOffset + 60);
        ushort subsystem = ReadUInt16(bytes, optionalHeaderOffset + (pe32 ? 68 : 88));
        ushort dllCharacteristics = ReadUInt16(bytes, optionalHeaderOffset + (pe32 ? 70 : 90));

        layout = new PeLayout(
            peOffset,
            sectionTableOffset,
            sectionCount,
            characteristics,
            addressOfEntryPoint,
            sizeOfImage,
            sizeOfHeaders,
            sizeOfCode,
            sizeOfInitializedData,
            sizeOfUninitializedData,
            subsystem,
            dllCharacteristics);
        return true;
    }

    private static ushort ReadUInt16(byte[] bytes, int offset)
    {
        return offset >= 0 && offset + 2 <= bytes.Length ? BitConverter.ToUInt16(bytes, offset) : (ushort)0;
    }

    private static uint ReadUInt32(byte[] bytes, int offset)
    {
        return offset >= 0 && offset + 4 <= bytes.Length ? BitConverter.ToUInt32(bytes, offset) : 0;
    }

    private readonly record struct PeLayout(
        int PeOffset,
        int SectionTableOffset,
        int SectionCount,
        ushort Characteristics,
        uint AddressOfEntryPoint,
        uint SizeOfImage,
        uint SizeOfHeaders,
        uint SizeOfCode,
        uint SizeOfInitializedData,
        uint SizeOfUninitializedData,
        ushort Subsystem,
        ushort DllCharacteristics);
}
