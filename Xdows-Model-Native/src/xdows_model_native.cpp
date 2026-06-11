#include "xdows_model_native.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <Windows.h>
#include <objbase.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace
{
    constexpr int kStandardFeatureCount = 299;
    constexpr int kFlashFeatureCount = 68;
    constexpr int kProRawBytesPerSection = 512;
    constexpr int kProRawFeatureCount = 3 * kProRawBytesPerSection;
    constexpr int kProStructuralFeatureCount = 32;
    constexpr int kProHybridFeatureCount =
        kStandardFeatureCount + kFlashFeatureCount + kProRawFeatureCount + kProStructuralFeatureCount;
    constexpr size_t kFlashRegionSize = 512ULL * 1024ULL;
    constexpr size_t kBlockEntropyRegionSize = 128ULL * 1024ULL;

    struct CommonStats
    {
        std::array<long long, 256> Counts{};
        int PrintableCount = 0;
        int ControlCount = 0;
        int WhitespaceCount = 0;
        int LetterCount = 0;
        int DigitCount = 0;
        int MaxZeroRun = 0;
        int HighByteCount = 0;
        int ZeroRunCount = 0;
        long long TotalZeroRunLength = 0;
        int MaxNonZeroRun = 0;
        long long TotalNonZeroRunLength = 0;
        int NonZeroRunCount = 0;
    };

    struct PeLayout
    {
        int PeOffset = 0;
        int SectionTableOffset = 0;
        int SectionCount = 0;
        std::uint16_t Characteristics = 0;
        std::uint32_t AddressOfEntryPoint = 0;
        std::uint32_t SizeOfImage = 0;
        std::uint32_t SizeOfHeaders = 0;
        std::uint32_t SizeOfCode = 0;
        std::uint32_t SizeOfInitializedData = 0;
        std::uint32_t SizeOfUninitializedData = 0;
        std::uint16_t Subsystem = 0;
        std::uint16_t DllCharacteristics = 0;
    };

    struct NativeSession
    {
        int Mode = XdowsModelNativeModeStandard;
        int FeatureCount = kStandardFeatureCount;
        std::filesystem::path ModelPath;
        Ort::Env Env;
        Ort::SessionOptions Options;
        std::unique_ptr<Ort::Session> Session;

        NativeSession(int mode, int featureCount, const std::filesystem::path& modelPath)
            : Mode(mode),
              FeatureCount(featureCount),
              ModelPath(modelPath),
              Env(ORT_LOGGING_LEVEL_WARNING, "XdowsModelNative")
        {
            Options.SetIntraOpNumThreads(1);
            Options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
            Session = std::make_unique<Ort::Session>(Env, ModelPath.c_str(), Options);
        }
    };

    bool IsWhitespace(std::uint8_t b)
    {
        return b == 9 || b == 10 || b == 13 || b == 32;
    }

    bool IsPrintable(std::uint8_t b)
    {
        return b >= 32 && b <= 126;
    }

    bool IsLetter(std::uint8_t b)
    {
        return (b >= 65 && b <= 90) || (b >= 97 && b <= 122);
    }

    bool IsDigit(std::uint8_t b)
    {
        return b >= 48 && b <= 57;
    }

    std::uint16_t ReadUInt16(const std::vector<std::uint8_t>& bytes, size_t offset)
    {
        if (offset + 2 > bytes.size())
            return 0;

        return static_cast<std::uint16_t>(bytes[offset] | (bytes[offset + 1] << 8));
    }

    std::uint32_t ReadUInt32(const std::vector<std::uint8_t>& bytes, size_t offset)
    {
        if (offset + 4 > bytes.size())
            return 0;

        return static_cast<std::uint32_t>(bytes[offset]) |
            (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) |
            (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) |
            (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
    }

    std::int32_t ReadInt32(const std::vector<std::uint8_t>& bytes, size_t offset)
    {
        return static_cast<std::int32_t>(ReadUInt32(bytes, offset));
    }

    bool IsPeFile(const std::vector<std::uint8_t>& bytes)
    {
        if (bytes.size() < 64 || bytes[0] != 'M' || bytes[1] != 'Z')
            return false;

        std::int32_t peOffset = ReadInt32(bytes, 60);
        if (peOffset < 0 || static_cast<size_t>(peOffset) + 4 > bytes.size())
            return false;

        return bytes[peOffset] == 'P' && bytes[peOffset + 1] == 'E';
    }

    bool ContainsAscii(const std::vector<std::uint8_t>& bytes, const std::string& needle)
    {
        if (needle.empty() || bytes.size() < needle.size())
            return false;

        return std::search(
            bytes.begin(),
            bytes.end(),
            needle.begin(),
            needle.end(),
            [](std::uint8_t a, char b)
            {
                char ca = static_cast<char>(a);
                char cb = b;
                if (ca >= 'A' && ca <= 'Z')
                    ca = static_cast<char>(ca - 'A' + 'a');
                if (cb >= 'A' && cb <= 'Z')
                    cb = static_cast<char>(cb - 'A' + 'a');
                return ca == cb;
            }) != bytes.end();
    }

    bool ReadAllBytes(const std::filesystem::path& path, std::vector<std::uint8_t>& bytes)
    {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        if (!stream)
            return false;

        std::streamsize size = stream.tellg();
        if (size <= 0)
            return false;

        stream.seekg(0, std::ios::beg);
        bytes.resize(static_cast<size_t>(size));
        return stream.read(reinterpret_cast<char*>(bytes.data()), size).good();
    }

    CommonStats ComputeCommonStats(const std::uint8_t* data, size_t length)
    {
        CommonStats stats;
        int currentZeroRun = 0;
        int currentNonZeroRun = 0;

        for (size_t i = 0; i < length; i++)
        {
            std::uint8_t b = data[i];
            stats.Counts[b]++;

            if (b >= 0x80)
                stats.HighByteCount++;

            if (IsWhitespace(b))
                stats.WhitespaceCount++;

            if (IsPrintable(b))
            {
                stats.PrintableCount++;
                if (IsLetter(b))
                    stats.LetterCount++;
                else if (IsDigit(b))
                    stats.DigitCount++;
            }
            else
            {
                stats.ControlCount++;
            }

            if (b == 0)
            {
                if (currentNonZeroRun > 0)
                {
                    stats.NonZeroRunCount++;
                    stats.TotalNonZeroRunLength += currentNonZeroRun;
                    stats.MaxNonZeroRun = std::max(stats.MaxNonZeroRun, currentNonZeroRun);
                    currentNonZeroRun = 0;
                }

                currentZeroRun++;
                stats.MaxZeroRun = std::max(stats.MaxZeroRun, currentZeroRun);
            }
            else
            {
                if (currentZeroRun > 0)
                {
                    stats.ZeroRunCount++;
                    stats.TotalZeroRunLength += currentZeroRun;
                    currentZeroRun = 0;
                }

                currentNonZeroRun++;
            }
        }

        if (currentZeroRun > 0)
        {
            stats.ZeroRunCount++;
            stats.TotalZeroRunLength += currentZeroRun;
        }

        if (currentNonZeroRun > 0)
        {
            stats.NonZeroRunCount++;
            stats.TotalNonZeroRunLength += currentNonZeroRun;
            stats.MaxNonZeroRun = std::max(stats.MaxNonZeroRun, currentNonZeroRun);
        }

        return stats;
    }

    double ComputeEntropy(const std::array<long long, 256>& counts, size_t totalBytes)
    {
        if (totalBytes == 0)
            return 0;

        double entropy = 0;
        double total = static_cast<double>(totalBytes);
        for (long long count : counts)
        {
            if (count <= 0)
                continue;

            double p = static_cast<double>(count) / total;
            entropy -= p * std::log2(p);
        }
        return entropy;
    }

    double ComputeRegionEntropy(const std::vector<std::uint8_t>& bytes, size_t start, size_t length)
    {
        if (start >= bytes.size())
            return 0;

        size_t actualLength = std::min(length, bytes.size() - start);
        if (actualLength == 0)
            return 0;

        CommonStats stats = ComputeCommonStats(bytes.data() + start, actualLength);
        return ComputeEntropy(stats.Counts, actualLength);
    }

    void ComputeByteMoments(
        const std::array<long long, 256>& counts,
        size_t totalBytes,
        double& mean,
        double& variance,
        double& skewness,
        double& kurtosis)
    {
        mean = 0;
        variance = 0;
        skewness = 0;
        kurtosis = 0;

        if (totalBytes == 0)
            return;

        double total = static_cast<double>(totalBytes);
        for (int i = 0; i < 256; i++)
            mean += i * static_cast<double>(counts[i]) / total;

        double m3 = 0;
        double m4 = 0;
        for (int i = 0; i < 256; i++)
        {
            double p = static_cast<double>(counts[i]) / total;
            double diff = i - mean;
            double diff2 = diff * diff;
            variance += diff2 * p;
            m3 += diff2 * diff * p;
            m4 += diff2 * diff2 * p;
        }

        double stdDev = std::sqrt(variance);
        skewness = stdDev > 0 ? m3 / (stdDev * stdDev * stdDev) : 0;
        kurtosis = variance > 0 ? m4 / (variance * variance) - 3 : 0;
    }

    void ComputeByteRangeRatios(
        const std::array<long long, 256>& counts,
        size_t totalBytes,
        double& lowByteRatio,
        double& printableAsciiRatio,
        double& extendedAsciiRatio)
    {
        lowByteRatio = 0;
        printableAsciiRatio = 0;
        extendedAsciiRatio = 0;
        if (totalBytes == 0)
            return;

        long long lowBytes = 0;
        for (int i = 0x00; i <= 0x1F; i++)
            lowBytes += counts[i];

        long long printableAscii = 0;
        for (int i = 0x20; i <= 0x7E; i++)
            printableAscii += counts[i];

        long long extendedAscii = 0;
        for (int i = 0x80; i <= 0xFF; i++)
            extendedAscii += counts[i];

        double total = static_cast<double>(totalBytes);
        lowByteRatio = static_cast<double>(lowBytes) / total;
        printableAsciiRatio = static_cast<double>(printableAscii) / total;
        extendedAsciiRatio = static_cast<double>(extendedAscii) / total;
    }

    void ComputeBlockEntropyStats(
        const std::uint8_t* data,
        size_t length,
        size_t blockSize,
        size_t maxRegionSize,
        double& minEntropy,
        double& maxEntropy,
        double& meanEntropy,
        double& variance)
    {
        minEntropy = 0;
        maxEntropy = 0;
        meanEntropy = 0;
        variance = 0;

        size_t analysisLength = std::min(length, maxRegionSize);
        if (analysisLength == 0 || blockSize == 0)
            return;

        size_t blockCount = (analysisLength + blockSize - 1) / blockSize;
        double totalEntropy = 0;
        double totalEntropySq = 0;
        minEntropy = std::numeric_limits<double>::max();
        maxEntropy = std::numeric_limits<double>::lowest();

        for (size_t blockIndex = 0; blockIndex < blockCount; blockIndex++)
        {
            size_t start = blockIndex * blockSize;
            size_t end = std::min(start + blockSize, analysisLength);
            size_t currentBlockSize = end - start;
            CommonStats blockStats = ComputeCommonStats(data + start, currentBlockSize);
            double blockEntropy = ComputeEntropy(blockStats.Counts, currentBlockSize);

            minEntropy = std::min(minEntropy, blockEntropy);
            maxEntropy = std::max(maxEntropy, blockEntropy);
            totalEntropy += blockEntropy;
            totalEntropySq += blockEntropy * blockEntropy;
        }

        meanEntropy = totalEntropy / blockCount;
        double var = (totalEntropySq / blockCount) - (meanEntropy * meanEntropy);
        variance = var < 0 ? 0 : var;
    }

    void ComputeStandardBlockEntropy(
        const std::vector<std::uint8_t>& bytes,
        double& minEntropy,
        double& maxEntropy,
        double& meanEntropy,
        double& variance,
        double& minPosition,
        double& maxPosition,
        double& firstEntropy,
        double& lastEntropy)
    {
        constexpr size_t blockSize = 256;
        size_t blockCount = (bytes.size() + blockSize - 1) / blockSize;
        if (blockCount == 0)
        {
            minEntropy = maxEntropy = meanEntropy = variance = minPosition = maxPosition = firstEntropy = lastEntropy = 0;
            return;
        }

        minEntropy = std::numeric_limits<double>::max();
        maxEntropy = std::numeric_limits<double>::lowest();
        double totalEntropy = 0;
        double totalEntropySq = 0;
        size_t minIndex = 0;
        size_t maxIndex = 0;

        for (size_t blockIndex = 0; blockIndex < blockCount; blockIndex++)
        {
            size_t start = blockIndex * blockSize;
            size_t end = std::min(start + blockSize, bytes.size());
            size_t currentBlockSize = end - start;
            CommonStats blockStats = ComputeCommonStats(bytes.data() + start, currentBlockSize);
            double blockEntropy = ComputeEntropy(blockStats.Counts, currentBlockSize);

            if (blockIndex == 0)
                firstEntropy = blockEntropy;
            if (blockIndex == blockCount - 1)
                lastEntropy = blockEntropy;

            if (blockEntropy < minEntropy)
            {
                minEntropy = blockEntropy;
                minIndex = blockIndex;
            }

            if (blockEntropy > maxEntropy)
            {
                maxEntropy = blockEntropy;
                maxIndex = blockIndex;
            }

            totalEntropy += blockEntropy;
            totalEntropySq += blockEntropy * blockEntropy;
        }

        meanEntropy = totalEntropy / blockCount;
        double var = (totalEntropySq / blockCount) - (meanEntropy * meanEntropy);
        variance = var < 0 ? 0 : var;
        minPosition = blockCount > 1 ? static_cast<double>(minIndex) / (blockCount - 1) : 0;
        maxPosition = blockCount > 1 ? static_cast<double>(maxIndex) / (blockCount - 1) : 0;
    }

    void ParsePeHeader(const std::vector<std::uint8_t>& bytes, float* values)
    {
        values[0] = 0;
        values[1] = 0;
        values[2] = 0;
        values[3] = 0;
        values[4] = 0;

        if (bytes.size() < 64)
            return;

        std::int32_t peOffset = ReadInt32(bytes, 60);
        if (peOffset < 0 || static_cast<size_t>(peOffset) + 24 > bytes.size())
            return;

        if (bytes[peOffset] != 'P' || bytes[peOffset + 1] != 'E')
            return;

        values[0] = static_cast<float>(ReadUInt16(bytes, peOffset + 6));
        values[1] = static_cast<float>(ReadUInt32(bytes, peOffset + 8));
        values[2] = static_cast<float>(ReadUInt16(bytes, peOffset + 22));

        size_t optionalHeaderOffset = static_cast<size_t>(peOffset) + 24;
        if (optionalHeaderOffset + 2 > bytes.size())
            return;

        std::uint16_t magic = ReadUInt16(bytes, optionalHeaderOffset);
        values[4] = static_cast<float>(magic);

        bool pe32 = magic == 0x10b;
        size_t sizeOfHeadersOffset = optionalHeaderOffset + (pe32 ? 60 : 84);
        if (sizeOfHeadersOffset + 4 <= bytes.size())
            values[3] = static_cast<float>(ReadUInt32(bytes, sizeOfHeadersOffset));
    }

    bool TryReadPeLayout(const std::vector<std::uint8_t>& bytes, PeLayout& layout)
    {
        layout = {};
        if (bytes.size() < 64)
            return false;

        std::int32_t peOffset = ReadInt32(bytes, 60);
        if (peOffset < 0 || static_cast<size_t>(peOffset) + 24 > bytes.size())
            return false;

        if (bytes[peOffset] != 'P' || bytes[peOffset + 1] != 'E')
            return false;

        size_t optionalHeaderOffset = static_cast<size_t>(peOffset) + 24;
        if (optionalHeaderOffset + 92 > bytes.size())
            return false;

        std::uint16_t optionalHeaderSize = ReadUInt16(bytes, peOffset + 20);
        std::uint16_t magic = ReadUInt16(bytes, optionalHeaderOffset);
        bool pe32 = magic == 0x10b;

        layout.PeOffset = peOffset;
        layout.SectionTableOffset = static_cast<int>(optionalHeaderOffset + optionalHeaderSize);
        layout.SectionCount = ReadUInt16(bytes, peOffset + 6);
        layout.Characteristics = ReadUInt16(bytes, peOffset + 22);
        layout.AddressOfEntryPoint = ReadUInt32(bytes, optionalHeaderOffset + 16);
        layout.SizeOfCode = ReadUInt32(bytes, optionalHeaderOffset + 4);
        layout.SizeOfInitializedData = ReadUInt32(bytes, optionalHeaderOffset + 8);
        layout.SizeOfUninitializedData = ReadUInt32(bytes, optionalHeaderOffset + 12);
        layout.SizeOfImage = ReadUInt32(bytes, optionalHeaderOffset + 56);
        layout.SizeOfHeaders = ReadUInt32(bytes, optionalHeaderOffset + 60);
        layout.Subsystem = ReadUInt16(bytes, optionalHeaderOffset + (pe32 ? 68 : 88));
        layout.DllCharacteristics = ReadUInt16(bytes, optionalHeaderOffset + (pe32 ? 70 : 90));
        return true;
    }

    void AppendStandardFeatures(const std::vector<std::uint8_t>& bytes, std::vector<float>& features)
    {
        features.reserve(features.size() + kStandardFeatureCount);

        CommonStats stats = ComputeCommonStats(bytes.data(), bytes.size());
        double total = static_cast<double>(bytes.size());

        for (int i = 0; i < 256; i++)
            features.push_back(static_cast<float>(static_cast<double>(stats.Counts[i]) / total));

        int uniqueBytes = 0;
        long long maxCount = 0;
        long long minCount = std::numeric_limits<long long>::max();
        int mostCommonByte = 0;
        int leastCommonByte = 0;
        for (int i = 0; i < 256; i++)
        {
            long long count = stats.Counts[i];
            if (count > 0)
                uniqueBytes++;
            if (count > maxCount)
            {
                maxCount = count;
                mostCommonByte = i;
            }
            if (count < minCount)
            {
                minCount = count;
                leastCommonByte = i;
            }
        }

        double entropy = ComputeEntropy(stats.Counts, bytes.size());
        double minBlockEntropy = 0;
        double maxBlockEntropy = 0;
        double meanBlockEntropy = 0;
        double blockEntropyVariance = 0;
        double minEntropyBlockPosition = 0;
        double maxEntropyBlockPosition = 0;
        double firstBlockEntropy = 0;
        double lastBlockEntropy = 0;
        ComputeStandardBlockEntropy(
            bytes,
            minBlockEntropy,
            maxBlockEntropy,
            meanBlockEntropy,
            blockEntropyVariance,
            minEntropyBlockPosition,
            maxEntropyBlockPosition,
            firstBlockEntropy,
            lastBlockEntropy);

        double meanByteValue = 0;
        double byteValueVariance = 0;
        double skewness = 0;
        double kurtosis = 0;
        ComputeByteMoments(stats.Counts, bytes.size(), meanByteValue, byteValueVariance, skewness, kurtosis);

        double lowByteRatio = 0;
        double printableAsciiRatio = 0;
        double extendedAsciiRatio = 0;
        ComputeByteRangeRatios(stats.Counts, bytes.size(), lowByteRatio, printableAsciiRatio, extendedAsciiRatio);

        double headBlockEntropyMin = 0;
        double headBlockEntropyMax = 0;
        double headBlockEntropyMean = 0;
        double headBlockEntropyVar = 0;
        ComputeBlockEntropyStats(
            bytes.data(),
            bytes.size(),
            4096,
            kBlockEntropyRegionSize,
            headBlockEntropyMin,
            headBlockEntropyMax,
            headBlockEntropyMean,
            headBlockEntropyVar);

        float peValues[5]{};
        ParsePeHeader(bytes, peValues);

        features.push_back(static_cast<float>(std::log(total + 1.0)));
        features.push_back(static_cast<float>(entropy));
        features.push_back(static_cast<float>(minBlockEntropy));
        features.push_back(static_cast<float>(maxBlockEntropy));
        features.push_back(static_cast<float>(meanBlockEntropy));
        features.push_back(static_cast<float>(blockEntropyVariance));
        features.push_back(static_cast<float>(minEntropyBlockPosition));
        features.push_back(static_cast<float>(maxEntropyBlockPosition));
        features.push_back(static_cast<float>(firstBlockEntropy));
        features.push_back(static_cast<float>(lastBlockEntropy));
        features.push_back(static_cast<float>(uniqueBytes));
        features.push_back(static_cast<float>(mostCommonByte));
        features.push_back(static_cast<float>(static_cast<double>(maxCount) / total));
        features.push_back(static_cast<float>(leastCommonByte));
        features.push_back(static_cast<float>(static_cast<double>(minCount) / total));
        features.push_back(static_cast<float>(static_cast<double>(stats.PrintableCount) / total));
        features.push_back(static_cast<float>(static_cast<double>(stats.ControlCount) / total));
        features.push_back(static_cast<float>(static_cast<double>(stats.WhitespaceCount) / total));
        features.push_back(static_cast<float>(static_cast<double>(stats.LetterCount) / total));
        features.push_back(static_cast<float>(static_cast<double>(stats.DigitCount) / total));
        features.push_back(static_cast<float>(stats.MaxZeroRun));
        features.push_back(static_cast<float>(static_cast<double>(stats.Counts[0]) / total));
        features.push_back(static_cast<float>(static_cast<double>(stats.HighByteCount) / total));
        features.push_back(static_cast<float>(meanByteValue));
        features.push_back(static_cast<float>(byteValueVariance));
        features.push_back(static_cast<float>(skewness));
        features.push_back(static_cast<float>(kurtosis));
        features.push_back(static_cast<float>(stats.ZeroRunCount > 0
            ? static_cast<double>(stats.TotalZeroRunLength) / stats.ZeroRunCount
            : 0));
        features.push_back(static_cast<float>(stats.ZeroRunCount));
        features.push_back(static_cast<float>(lowByteRatio));
        features.push_back(static_cast<float>(printableAsciiRatio));
        features.push_back(static_cast<float>(extendedAsciiRatio));
        features.push_back(static_cast<float>(stats.MaxNonZeroRun));
        features.push_back(static_cast<float>(stats.NonZeroRunCount > 0
            ? static_cast<double>(stats.TotalNonZeroRunLength) / stats.NonZeroRunCount
            : 0));
        for (float peValue : peValues)
            features.push_back(peValue);
        features.push_back(static_cast<float>(headBlockEntropyMin));
        features.push_back(static_cast<float>(headBlockEntropyMax));
        features.push_back(static_cast<float>(headBlockEntropyMean));
        features.push_back(static_cast<float>(headBlockEntropyVar));
    }

    void AppendFlashFeatures(const std::vector<std::uint8_t>& bytes, std::vector<float>& features)
    {
        features.reserve(features.size() + kFlashFeatureCount);

        size_t headLength = std::min(bytes.size(), kFlashRegionSize);
        size_t tailStart = bytes.size() > kFlashRegionSize ? bytes.size() - kFlashRegionSize : 0;
        size_t tailLength = bytes.size() - tailStart;

        CommonStats stats = ComputeCommonStats(bytes.data(), headLength);
        double total = static_cast<double>(headLength);

        int uniqueBytes = 0;
        long long maxCount = 0;
        for (int i = 0; i < 256; i++)
        {
            if (stats.Counts[i] > 0)
                uniqueBytes++;
            if (stats.Counts[i] > maxCount)
                maxCount = stats.Counts[i];
        }

        double meanByteValue = 0;
        double byteValueVariance = 0;
        double skewness = 0;
        double kurtosis = 0;
        ComputeByteMoments(stats.Counts, headLength, meanByteValue, byteValueVariance, skewness, kurtosis);

        std::array<float, 32> histogram32{};
        for (int bin = 0; bin < 32; bin++)
        {
            long long sum = 0;
            for (int j = 0; j < 8; j++)
                sum += stats.Counts[bin * 8 + j];
            histogram32[bin] = total > 0 ? static_cast<float>(static_cast<double>(sum) / total) : 0;
        }

        double lowByteRatio = 0;
        double printableAsciiRatio = 0;
        double extendedAsciiRatio = 0;
        ComputeByteRangeRatios(stats.Counts, headLength, lowByteRatio, printableAsciiRatio, extendedAsciiRatio);

        double hMin = 0;
        double hMax = 0;
        double hMean = 0;
        double hVar = 0;
        ComputeBlockEntropyStats(bytes.data(), headLength, 4096, kBlockEntropyRegionSize, hMin, hMax, hMean, hVar);

        double tMin = 0;
        double tMax = 0;
        double tMean = 0;
        double tVar = 0;
        ComputeBlockEntropyStats(bytes.data() + tailStart, tailLength, 4096, kBlockEntropyRegionSize, tMin, tMax, tMean, tVar);

        float peValues[5]{};
        ParsePeHeader(bytes, peValues);

        features.push_back(static_cast<float>(std::log(static_cast<double>(bytes.size()) + 1.0)));
        features.push_back(static_cast<float>(ComputeEntropy(stats.Counts, headLength)));
        features.push_back(static_cast<float>(total > 0 ? static_cast<double>(stats.Counts[0]) / total : 0));
        features.push_back(static_cast<float>(total > 0 ? static_cast<double>(stats.HighByteCount) / total : 0));
        features.push_back(static_cast<float>(total > 0 ? static_cast<double>(stats.PrintableCount) / total : 0));
        features.push_back(static_cast<float>(total > 0 ? static_cast<double>(stats.ControlCount) / total : 0));
        features.push_back(static_cast<float>(total > 0 ? static_cast<double>(stats.WhitespaceCount) / total : 0));
        features.push_back(static_cast<float>(total > 0 ? static_cast<double>(stats.LetterCount) / total : 0));
        features.push_back(static_cast<float>(total > 0 ? static_cast<double>(stats.DigitCount) / total : 0));
        features.push_back(static_cast<float>(uniqueBytes));
        features.push_back(static_cast<float>(total > 0 ? static_cast<double>(maxCount) / total : 0));
        features.push_back(static_cast<float>(stats.MaxZeroRun));
        features.push_back(static_cast<float>(meanByteValue));
        features.push_back(static_cast<float>(byteValueVariance));
        features.push_back(static_cast<float>(skewness));
        features.push_back(static_cast<float>(kurtosis));
        features.push_back(static_cast<float>(stats.ZeroRunCount > 0
            ? static_cast<double>(stats.TotalZeroRunLength) / stats.ZeroRunCount
            : 0));
        features.push_back(static_cast<float>(stats.ZeroRunCount));

        for (float value : histogram32)
            features.push_back(value);

        features.push_back(static_cast<float>(lowByteRatio));
        features.push_back(static_cast<float>(printableAsciiRatio));
        features.push_back(static_cast<float>(extendedAsciiRatio));
        features.push_back(static_cast<float>(stats.MaxNonZeroRun));
        features.push_back(static_cast<float>(stats.NonZeroRunCount > 0
            ? static_cast<double>(stats.TotalNonZeroRunLength) / stats.NonZeroRunCount
            : 0));

        features.push_back(static_cast<float>(hMin));
        features.push_back(static_cast<float>(hMax));
        features.push_back(static_cast<float>(hMean));
        features.push_back(static_cast<float>(hVar));
        features.push_back(static_cast<float>(tMin));
        features.push_back(static_cast<float>(tMax));
        features.push_back(static_cast<float>(tMean));
        features.push_back(static_cast<float>(tVar));

        for (float peValue : peValues)
            features.push_back(peValue);
    }

    void AppendProRawFeatures(const std::vector<std::uint8_t>& bytes, std::vector<float>& features)
    {
        size_t base = features.size();
        features.resize(base + kProRawFeatureCount, 0.0f);

        auto copySection = [&](size_t sourceStart, size_t length, size_t targetOffset)
        {
            size_t maxLength = std::min(length, static_cast<size_t>(kProRawBytesPerSection));
            for (size_t i = 0; i < maxLength && sourceStart + i < bytes.size(); i++)
                features[base + targetOffset + i] = static_cast<float>(bytes[sourceStart + i]);
        };

        size_t fileSize = bytes.size();
        size_t headLength = std::min(fileSize, static_cast<size_t>(kProRawBytesPerSection));
        size_t midStart = fileSize / 2 > kProRawBytesPerSection / 2
            ? fileSize / 2 - kProRawBytesPerSection / 2
            : 0;
        size_t midLength = std::min(midStart + kProRawBytesPerSection, fileSize) - midStart;
        size_t tailStart = fileSize > kProRawBytesPerSection ? fileSize - kProRawBytesPerSection : 0;
        size_t tailLength = fileSize - tailStart;

        copySection(0, headLength, 0);
        copySection(midStart, midLength, kProRawBytesPerSection);
        copySection(tailStart, tailLength, kProRawBytesPerSection * 2);
    }

    void AppendProStructuralFeatures(const std::vector<std::uint8_t>& bytes, std::vector<float>& features)
    {
        std::array<float, kProStructuralFeatureCount> values{};
        PeLayout layout;
        if (!TryReadPeLayout(bytes, layout))
        {
            features.insert(features.end(), values.begin(), values.end());
            return;
        }

        int sectionCount = std::max(0, layout.SectionCount);
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
        long long lastSectionEnd = 0;

        double entropySum = 0;
        double entropySquaredSum = 0;
        double minEntropy = std::numeric_limits<double>::max();
        double maxEntropy = 0;
        double rawSizeSum = 0;
        double maxRawSize = 0;
        double virtualSizeSum = 0;
        double maxVirtualSize = 0;
        double rawVirtualRatioSum = 0;
        double maxRawVirtualRatio = 0;

        for (int i = 0; i < sectionCount; i++)
        {
            size_t sectionOffset = static_cast<size_t>(layout.SectionTableOffset) + static_cast<size_t>(i) * 40;
            if (sectionOffset + 40 > bytes.size())
                break;

            std::uint32_t virtualSize = ReadUInt32(bytes, sectionOffset + 8);
            std::uint32_t virtualAddress = ReadUInt32(bytes, sectionOffset + 12);
            std::uint32_t rawSize = ReadUInt32(bytes, sectionOffset + 16);
            std::uint32_t rawPointer = ReadUInt32(bytes, sectionOffset + 20);
            std::uint32_t characteristics = ReadUInt32(bytes, sectionOffset + 36);

            parsedSections++;

            bool executable = (characteristics & 0x20000000) != 0;
            bool readable = (characteristics & 0x40000000) != 0;
            bool writable = (characteristics & 0x80000000) != 0;

            if (executable)
                executableCount++;
            if (readable)
                readableCount++;
            if (writable)
                writableCount++;
            if ((characteristics & 0x00000020) != 0)
                codeCount++;
            if ((characteristics & 0x00000040) != 0)
                initializedDataCount++;
            if ((characteristics & 0x00000080) != 0)
                uninitializedDataCount++;
            if (executable && writable)
                suspiciousRwxCount++;
            if (rawSize == 0)
                zeroRawCount++;

            std::uint32_t effectiveVirtualSize = std::max(virtualSize, rawSize);
            if (entrySectionIndex < 0 &&
                layout.AddressOfEntryPoint >= virtualAddress &&
                layout.AddressOfEntryPoint < virtualAddress + effectiveVirtualSize)
            {
                entrySectionIndex = i;
            }

            size_t availableRawSize = 0;
            if (rawPointer < bytes.size())
                availableRawSize = std::min(static_cast<size_t>(rawSize), bytes.size() - rawPointer);

            double entropy = availableRawSize > 0 ? ComputeRegionEntropy(bytes, rawPointer, availableRawSize) : 0;
            entropySum += entropy;
            entropySquaredSum += entropy * entropy;
            minEntropy = std::min(minEntropy, entropy);
            maxEntropy = std::max(maxEntropy, entropy);

            rawSizeSum += rawSize;
            maxRawSize = std::max(maxRawSize, static_cast<double>(rawSize));
            virtualSizeSum += virtualSize;
            maxVirtualSize = std::max(maxVirtualSize, static_cast<double>(virtualSize));

            double rawVirtualRatio = virtualSize > 0 ? static_cast<double>(rawSize) / virtualSize : 0;
            rawVirtualRatioSum += rawVirtualRatio;
            maxRawVirtualRatio = std::max(maxRawVirtualRatio, rawVirtualRatio);

            long long sectionEnd = static_cast<long long>(rawPointer) + rawSize;
            lastSectionEnd = std::max(lastSectionEnd, sectionEnd);
        }

        int denominator = std::max(parsedSections, 1);
        double meanEntropy = parsedSections > 0 ? entropySum / parsedSections : 0;
        double entropyVariance = parsedSections > 0
            ? std::max(0.0, entropySquaredSum / parsedSections - meanEntropy * meanEntropy)
            : 0;
        if (minEntropy == std::numeric_limits<double>::max())
            minEntropy = 0;

        long long overlayBytes = std::max<long long>(0, static_cast<long long>(bytes.size()) - std::max<long long>(lastSectionEnd, layout.SizeOfHeaders));
        double overlayRatio = !bytes.empty() ? static_cast<double>(overlayBytes) / bytes.size() : 0;
        double entryRatio = layout.SizeOfImage > 0 ? static_cast<double>(layout.AddressOfEntryPoint) / layout.SizeOfImage : 0;

        size_t idx = 0;
        values[idx++] = static_cast<float>(sectionCount);
        values[idx++] = static_cast<float>(executableCount) / denominator;
        values[idx++] = static_cast<float>(writableCount) / denominator;
        values[idx++] = static_cast<float>(readableCount) / denominator;
        values[idx++] = static_cast<float>(codeCount) / denominator;
        values[idx++] = static_cast<float>(initializedDataCount) / denominator;
        values[idx++] = static_cast<float>(uninitializedDataCount) / denominator;
        values[idx++] = static_cast<float>(suspiciousRwxCount);
        values[idx++] = static_cast<float>(zeroRawCount) / denominator;
        values[idx++] = entrySectionIndex >= 0 && sectionCount > 0
            ? static_cast<float>(entrySectionIndex + 1) / sectionCount
            : 0;
        values[idx++] = static_cast<float>(meanEntropy);
        values[idx++] = static_cast<float>(minEntropy);
        values[idx++] = static_cast<float>(maxEntropy);
        values[idx++] = static_cast<float>(entropyVariance);
        values[idx++] = static_cast<float>(std::log(rawSizeSum / denominator + 1));
        values[idx++] = static_cast<float>(std::log(maxRawSize + 1));
        values[idx++] = static_cast<float>(std::log(virtualSizeSum / denominator + 1));
        values[idx++] = static_cast<float>(std::log(maxVirtualSize + 1));
        values[idx++] = static_cast<float>(rawVirtualRatioSum / denominator);
        values[idx++] = static_cast<float>(maxRawVirtualRatio);
        values[idx++] = static_cast<float>(std::log(static_cast<double>(layout.SizeOfImage) + 1));
        values[idx++] = static_cast<float>(std::log(static_cast<double>(layout.SizeOfCode) + 1));
        values[idx++] = static_cast<float>(std::log(static_cast<double>(layout.SizeOfInitializedData) + 1));
        values[idx++] = static_cast<float>(std::log(static_cast<double>(layout.SizeOfUninitializedData) + 1));
        values[idx++] = static_cast<float>(layout.Subsystem);
        values[idx++] = static_cast<float>(layout.DllCharacteristics);
        values[idx++] = static_cast<float>(layout.Characteristics);
        values[idx++] = !bytes.empty() ? static_cast<float>(static_cast<double>(layout.PeOffset) / bytes.size()) : 0;
        values[idx++] = !bytes.empty() ? static_cast<float>(static_cast<double>(layout.SizeOfHeaders) / bytes.size()) : 0;
        values[idx++] = static_cast<float>(entryRatio);
        values[idx++] = overlayBytes > 0 ? 1.0f : 0.0f;
        values[idx++] = static_cast<float>(overlayRatio);

        features.insert(features.end(), values.begin(), values.end());
    }

    bool ExtractFeaturesForMode(int mode, const std::vector<std::uint8_t>& bytes, std::vector<float>& features)
    {
        if (!IsPeFile(bytes))
            return false;

        features.clear();
        switch (mode)
        {
        case XdowsModelNativeModeFlash:
            AppendFlashFeatures(bytes, features);
            return features.size() == kFlashFeatureCount;
        case XdowsModelNativeModePro:
            AppendStandardFeatures(bytes, features);
            AppendFlashFeatures(bytes, features);
            AppendProRawFeatures(bytes, features);
            AppendProStructuralFeatures(bytes, features);
            return features.size() == kProHybridFeatureCount;
        default:
            AppendStandardFeatures(bytes, features);
            return features.size() == kStandardFeatureCount;
        }
    }

    int FeatureCountForMode(int mode)
    {
        switch (mode)
        {
        case XdowsModelNativeModeFlash:
            return kFlashFeatureCount;
        case XdowsModelNativeModePro:
            return kProHybridFeatureCount;
        default:
            return kStandardFeatureCount;
        }
    }

    float ThresholdForMode(int mode)
    {
        switch (mode)
        {
        case XdowsModelNativeModeFlash:
            return 96.0f;
        case XdowsModelNativeModePro:
            return 94.0f;
        default:
            return 92.0f;
        }
    }

    const wchar_t* ModelNameForMode(int mode)
    {
        switch (mode)
        {
        case XdowsModelNativeModeFlash:
            return L"Xdows-Model-Flash.onnx";
        case XdowsModelNativeModePro:
            return L"Xdows-Model-Pro.onnx";
        default:
            return L"Xdows-Model.onnx";
        }
    }

    std::wstring ModeName(int mode)
    {
        switch (mode)
        {
        case XdowsModelNativeModeFlash:
            return L"Flash";
        case XdowsModelNativeModePro:
            return L"Pro";
        default:
            return L"Standard";
        }
    }

    std::filesystem::path GetModuleDirectory()
    {
        HMODULE module = nullptr;
        wchar_t path[MAX_PATH]{};
        if (GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&XdowsModelNativeInitialize),
            &module) &&
            GetModuleFileNameW(module, path, MAX_PATH) > 0)
        {
            return std::filesystem::path(path).parent_path();
        }

        return std::filesystem::current_path();
    }

    std::filesystem::path ResolveModelPath(const wchar_t* modelDirectory, int mode)
    {
        const wchar_t* modelName = ModelNameForMode(mode);

        if (modelDirectory != nullptr && modelDirectory[0] != L'\0')
        {
            std::filesystem::path explicitPath = std::filesystem::path(modelDirectory) / modelName;
            if (std::filesystem::exists(explicitPath))
                return explicitPath;
        }

        std::filesystem::path modulePath = GetModuleDirectory() / modelName;
        if (std::filesystem::exists(modulePath))
            return modulePath;

        std::filesystem::path cwdPath = std::filesystem::current_path() / modelName;
        if (std::filesystem::exists(cwdPath))
            return cwdPath;

        return {};
    }

    wchar_t* DuplicateString(const std::wstring& value)
    {
        size_t bytes = (value.size() + 1) * sizeof(wchar_t);
        auto* buffer = static_cast<wchar_t*>(CoTaskMemAlloc(bytes));
        if (buffer == nullptr)
            return nullptr;

        memcpy(buffer, value.c_str(), bytes);
        return buffer;
    }

    void ResetResult(XDOWS_MODEL_NATIVE_SCAN_RESULT* result)
    {
        if (result == nullptr)
            return;

        result->Size = sizeof(XDOWS_MODEL_NATIVE_SCAN_RESULT);
        result->Status = XdowsModelNativeStatusOk;
        result->IsThreat = 0;
        result->Probability = 0.0f;
        result->DetectionName = nullptr;
        result->ErrorMessage = nullptr;
    }

    void SetError(XDOWS_MODEL_NATIVE_SCAN_RESULT* result, int status, const std::wstring& message)
    {
        if (result == nullptr)
            return;

        result->Status = status;
        result->IsThreat = 0;
        result->Probability = 0.0f;
        result->DetectionName = nullptr;
        result->ErrorMessage = DuplicateString(message);
    }

    bool RunOnnx(NativeSession* session, const std::vector<float>& features, float& probability, std::wstring& error)
    {
        probability = 0;
        error.clear();

        if (session == nullptr || session->Session == nullptr)
        {
            error = L"session-not-ready";
            return false;
        }

        if (static_cast<int>(features.size()) != session->FeatureCount)
        {
            error = L"feature-count-mismatch";
            return false;
        }

        try
        {
            Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::array<int64_t, 2> featuresShape{ 1, static_cast<int64_t>(features.size()) };
            std::array<int64_t, 2> labelShape{ 1, 1 };
            bool label = false;

            std::vector<Ort::Value> inputs;
            inputs.emplace_back(Ort::Value::CreateTensor<float>(
                memoryInfo,
                const_cast<float*>(features.data()),
                features.size(),
                featuresShape.data(),
                featuresShape.size()));
            inputs.emplace_back(Ort::Value::CreateTensor<bool>(
                memoryInfo,
                &label,
                1,
                labelShape.data(),
                labelShape.size()));

            const char* inputNames[] = { "Features", "Label" };
            const char* outputNames[] = { "Probability.output" };
            Ort::RunOptions runOptions;
            auto outputs = session->Session->Run(
                runOptions,
                inputNames,
                inputs.data(),
                inputs.size(),
                outputNames,
                1);

            if (outputs.empty())
            {
                error = L"missing-probability-output";
                return false;
            }

            float* output = outputs[0].GetTensorMutableData<float>();
            probability = std::clamp(output[0] * 100.0f, 0.0f, 100.0f);
            return true;
        }
        catch (const Ort::Exception& ex)
        {
            std::string message = ex.what();
            error.assign(message.begin(), message.end());
            return false;
        }
        catch (const std::exception& ex)
        {
            std::string message = ex.what();
            error.assign(message.begin(), message.end());
            return false;
        }
    }
}

extern "C" XDOWS_MODEL_NATIVE_API int __stdcall XdowsModelNativeInitialize(
    const wchar_t* modelDirectory,
    int mode,
    void** session)
{
    if (session == nullptr)
        return XdowsModelNativeStatusInvalidArgument;

    *session = nullptr;

    if (mode < XdowsModelNativeModeStandard || mode > XdowsModelNativeModePro)
        return XdowsModelNativeStatusInvalidArgument;

    std::filesystem::path modelPath = ResolveModelPath(modelDirectory, mode);
    if (modelPath.empty())
        return XdowsModelNativeStatusModelNotFound;

    try
    {
        auto nativeSession = std::make_unique<NativeSession>(mode, FeatureCountForMode(mode), modelPath);
        *session = nativeSession.release();
        return XdowsModelNativeStatusOk;
    }
    catch (...)
    {
        return XdowsModelNativeStatusInternalError;
    }
}

extern "C" XDOWS_MODEL_NATIVE_API int __stdcall XdowsModelNativeScanFile(
    void* session,
    const wchar_t* filePath,
    XDOWS_MODEL_NATIVE_SCAN_RESULT* result)
{
    ResetResult(result);

    if (session == nullptr || filePath == nullptr || result == nullptr)
        return XdowsModelNativeStatusInvalidArgument;

    auto* nativeSession = static_cast<NativeSession*>(session);
    std::filesystem::path path(filePath);
    if (!std::filesystem::exists(path))
    {
        SetError(result, XdowsModelNativeStatusFileNotFound, L"file-not-found");
        return XdowsModelNativeStatusFileNotFound;
    }

    std::vector<std::uint8_t> bytes;
    if (!ReadAllBytes(path, bytes))
    {
        SetError(result, XdowsModelNativeStatusInternalError, L"read-failed");
        return XdowsModelNativeStatusInternalError;
    }

    if (ContainsAscii(bytes, "EICAR-STANDARD-ANTIVIRUS-TEST-FILE"))
    {
        result->Status = XdowsModelNativeStatusOk;
        result->IsThreat = 1;
        result->Probability = 100.0f;
        result->DetectionName = DuplicateString(L"Xdows.Model.Native.EICAR");
        return XdowsModelNativeStatusOk;
    }

    if (!IsPeFile(bytes))
    {
        result->Status = XdowsModelNativeStatusOk;
        result->IsThreat = 0;
        result->Probability = 0.0f;
        return XdowsModelNativeStatusOk;
    }

    std::vector<float> features;
    if (!ExtractFeaturesForMode(nativeSession->Mode, bytes, features))
    {
        SetError(result, XdowsModelNativeStatusUnsupportedFile, L"feature-extraction-failed");
        return XdowsModelNativeStatusUnsupportedFile;
    }

    float probability = 0;
    std::wstring error;
    if (!RunOnnx(nativeSession, features, probability, error))
    {
        SetError(result, XdowsModelNativeStatusInternalError, error.empty() ? L"onnx-run-failed" : error);
        return XdowsModelNativeStatusInternalError;
    }

    result->Status = XdowsModelNativeStatusOk;
    result->Probability = probability;
    if (probability >= ThresholdForMode(nativeSession->Mode))
    {
        result->IsThreat = 1;
        result->DetectionName = DuplicateString(
            L"Xdows.Model.Native." + ModeName(nativeSession->Mode) + L".Probability" +
            std::to_wstring(static_cast<int>(probability)));
    }
    else
    {
        result->IsThreat = 0;
    }

    return XdowsModelNativeStatusOk;
}

extern "C" XDOWS_MODEL_NATIVE_API void __stdcall XdowsModelNativeShutdown(
    void* session)
{
    auto* nativeSession = static_cast<NativeSession*>(session);
    delete nativeSession;
}

extern "C" XDOWS_MODEL_NATIVE_API void __stdcall XdowsModelNativeFreeString(
    wchar_t* value)
{
    if (value != nullptr)
        CoTaskMemFree(value);
}
