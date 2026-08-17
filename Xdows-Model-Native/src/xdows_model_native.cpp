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
#include <future>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    constexpr int kStandardFeatureCount = 299;
    constexpr int kFlashFeatureCount = 68;
    constexpr int kProRawStatFeaturesPerSection = 40;
    constexpr int kProRawStatSectionCount = 3;
    constexpr int kProRawStatFeatureCount = kProRawStatFeaturesPerSection * kProRawStatSectionCount;
    constexpr int kProRawStatSectionSize = 512;
    constexpr int kProStructuralFeatureCount = 32;
    constexpr int kProFixedFeatureCount = kStandardFeatureCount + kFlashFeatureCount + kProStructuralFeatureCount;
    constexpr int kProHybridFeatureCount = kProFixedFeatureCount + kProRawStatFeatureCount;
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
        std::array<std::unique_ptr<Ort::Session>, 4> ProBranchSessions;
        std::unique_ptr<NativeSession> AdaptiveFlash;
        std::unique_ptr<NativeSession> AdaptiveStandard;
        std::unique_ptr<NativeSession> AdaptivePro;
        std::array<int, 4> ProBranchFeatureCounts{
            kStandardFeatureCount,
            kFlashFeatureCount,
            kProRawStatFeatureCount,
            kProStructuralFeatureCount };

        NativeSession(int mode, int featureCount, const std::filesystem::path& modelPath)
            : Mode(mode),
              FeatureCount(featureCount),
              ModelPath(modelPath),
              Env(ORT_LOGGING_LEVEL_WARNING, "XdowsModelNative")
        {
Options.SetIntraOpNumThreads(1);
            Options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            if (Mode == XdowsModelNativeModeAdaptive)
            {
                auto directory = ModelPath.parent_path();
                AdaptiveFlash = std::make_unique<NativeSession>(
                    XdowsModelNativeModeFlash,
                    kFlashFeatureCount,
                    directory / L"Xdows-Model-Flash.onnx");
                AdaptiveStandard = std::make_unique<NativeSession>(
                    XdowsModelNativeModeStandard,
                    kStandardFeatureCount,
                    directory / L"Xdows-Model.onnx");
                AdaptivePro = std::make_unique<NativeSession>(
                    XdowsModelNativeModePro,
                    kProHybridFeatureCount,
                    directory / L"Xdows-Model-Pro.onnx");
                return;
            }

            Session = std::make_unique<Ort::Session>(Env, ModelPath.c_str(), Options);

            FeatureCount = ReadFeatureCount(*Session, FeatureCount);

            if (Mode == XdowsModelNativeModePro && FeatureCount == 4)
            {
                const std::array<std::wstring, 4> suffixes{
                    L"-Standard", L"-Flash", L"-RawStat", L"-Structural" };
                for (size_t i = 0; i < suffixes.size(); i++)
                {
                    std::filesystem::path branchPath = AddSuffix(ModelPath, suffixes[i]);
                    if (!std::filesystem::exists(branchPath))
                        throw std::runtime_error("missing-pro-stacking-branch");
                    ProBranchSessions[i] = std::make_unique<Ort::Session>(Env, branchPath.c_str(), Options);
                    int actual = ReadFeatureCount(*ProBranchSessions[i], ProBranchFeatureCounts[i]);
                    if (actual != ProBranchFeatureCounts[i])
                        throw std::runtime_error("pro-stacking-branch-dimension-mismatch");
                }
            }
        }

        static int ReadFeatureCount(Ort::Session& session, int fallback)
        {
            Ort::AllocatorWithDefaultOptions allocator;
            size_t inputCount = session.GetInputCount();
            for (size_t i = 0; i < inputCount; i++)
            {
                auto inputName = session.GetInputNameAllocated(i, allocator);
                if (std::strcmp(inputName.get(), "Features") != 0)
                    continue;
                auto shape = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
                if (shape.size() == 2 && shape[1] > 0)
                    return static_cast<int>(shape[1]);
                if (shape.size() == 1 && shape[0] > 0)
                    return static_cast<int>(shape[0]);
            }
            return fallback;
        }

        static std::filesystem::path AddSuffix(const std::filesystem::path& path, const std::wstring& suffix)
        {
            return path.parent_path() / (path.stem().wstring() + suffix + path.extension().wstring());
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

    // 字节分类查表（与 Managed ByteAnalysisHelper.ByteClass 位布局完全一致）：
    // bit0=highByte(>=0x80) bit1=whitespace bit2=printable bit3=letter bit4=digit
    constexpr std::array<std::uint8_t, 256> kByteClass = [] {
        std::array<std::uint8_t, 256> table{};
        for (int i = 0; i < 256; i++)
        {
            std::uint8_t b = static_cast<std::uint8_t>(i);
            std::uint8_t c = 0;
            if (b >= 0x80) c |= 0x01;
            if (b == 9 || b == 10 || b == 13 || b == 32) c |= 0x02;
            if (b >= 32 && b <= 126)
            {
                c |= 0x04;
                if ((b >= 65 && b <= 90) || (b >= 97 && b <= 122)) c |= 0x08;
                else if (b >= 48 && b <= 57) c |= 0x10;
            }
            table[i] = c;
        }
        return table;
    }();

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

    // T4：分区域读取。只读 head（前 min(size, kFlashRegionSize)）+ tail（最后 min(size, kFlashRegionSize)）。
    // 小文件（size <= kFlashRegionSize）tail 留空，特征层复用 head（数学等价：tail 区域 == head 区域）。
    // 与 Managed FlashFeatureExtractor.ExtractFeatures(filePath) 语义一致（delta=0.00 已验证）。
    bool ReadFileRegions(const std::filesystem::path& path,
                         std::vector<std::uint8_t>& head,
                         std::vector<std::uint8_t>& tail,
                         size_t& totalSize)
    {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        if (!stream)
            return false;

        std::streamsize size = stream.tellg();
        if (size <= 0)
            return false;

        totalSize = static_cast<size_t>(size);
        size_t headLength = std::min(totalSize, kFlashRegionSize);

        stream.seekg(0, std::ios::beg);
        head.resize(headLength);
        if (!stream.read(reinterpret_cast<char*>(head.data()), static_cast<std::streamsize>(headLength)).good())
            return false;

        if (totalSize > kFlashRegionSize)
        {
            size_t tailPos = totalSize - kFlashRegionSize;
            stream.seekg(static_cast<std::streamoff>(tailPos), std::ios::beg);
            tail.resize(kFlashRegionSize);
            if (!stream.read(reinterpret_cast<char*>(tail.data()), static_cast<std::streamsize>(kFlashRegionSize)).good())
                return false;
        }

        return true;
    }

    inline void AccumulateStats(CommonStats& stats, std::uint8_t b, std::uint8_t cls,
                                int& currentZeroRun, int& currentNonZeroRun)
    {
        stats.Counts[b]++;
        stats.HighByteCount += cls & 1;
        stats.WhitespaceCount += (cls >> 1) & 1;
        int printable = (cls >> 2) & 1;
        stats.PrintableCount += printable;
        stats.ControlCount += printable ^ 1;
        stats.LetterCount += (cls >> 3) & printable;
        stats.DigitCount += (cls >> 4) & printable;

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

    inline void FinalizeRuns(CommonStats& stats, int& currentZeroRun, int& currentNonZeroRun)
    {
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
    }

    CommonStats ComputeCommonStats(const std::uint8_t* data, size_t length)
    {
        CommonStats stats;
        int currentZeroRun = 0;
        int currentNonZeroRun = 0;

        for (size_t i = 0; i < length; i++)
        {
            std::uint8_t b = data[i];
            AccumulateStats(stats, b, kByteClass[b], currentZeroRun, currentNonZeroRun);
        }

        FinalizeRuns(stats, currentZeroRun, currentNonZeroRun);
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

    // 区域 4096 块熵（等价旧 ComputeBlockEntropyStats；熵只依赖频数，无需逐块 CommonStats）
    void ComputeRegionBlockEntropy(const std::uint8_t* data, size_t length,
                                   size_t blockSize, size_t maxRegionSize,
                                   double& minEntropy, double& maxEntropy,
                                   double& meanEntropy, double& variance)
    {
        minEntropy = 0;
        maxEntropy = 0;
        meanEntropy = 0;
        variance = 0;

        size_t analysisLength = std::min(length, maxRegionSize);
        if (analysisLength == 0 || blockSize == 0)
            return;

        size_t blockCount = (analysisLength + blockSize - 1) / blockSize;
        std::array<long long, 256> counts{};
        size_t blockStart = 0;
        double totalEntropy = 0;
        double totalEntropySq = 0;
        minEntropy = std::numeric_limits<double>::max();
        maxEntropy = std::numeric_limits<double>::lowest();

        for (size_t i = 0; i < analysisLength; i++)
        {
            counts[data[i]]++;
            if ((i + 1) % blockSize == 0 || i == analysisLength - 1)
            {
                size_t currentBlockSize = (i + 1) - blockStart;
                double blockEntropy = ComputeEntropy(counts, currentBlockSize);

                minEntropy = std::min(minEntropy, blockEntropy);
                maxEntropy = std::max(maxEntropy, blockEntropy);
                totalEntropy += blockEntropy;
                totalEntropySq += blockEntropy * blockEntropy;

                counts.fill(0);
                blockStart = i + 1;
            }
        }

        meanEntropy = totalEntropy / blockCount;
        double var = (totalEntropySq / blockCount) - (meanEntropy * meanEntropy);
        variance = var < 0 ? 0 : var;
    }

    struct UnifiedScanResult
    {
        CommonStats Full;
        CommonStats Head;
        double MinBlockEntropy = 0;
        double MaxBlockEntropy = 0;
        double MeanBlockEntropy = 0;
        double BlockEntropyVariance = 0;
        double MinPosition = 0;
        double MaxPosition = 0;
        double FirstEntropy = 0;
        double LastEntropy = 0;
        double HeadMin = 0;
        double HeadMax = 0;
        double HeadMean = 0;
        double HeadVar = 0;
        double TailMin = 0;
        double TailMax = 0;
        double TailMean = 0;
        double TailVar = 0;
    };

    UnifiedScanResult ComputeUnifiedScan(const std::uint8_t* data, size_t length,
                                         bool needFull, bool needFlash)
    {
        UnifiedScanResult result;
        if (length == 0)
            return result;

        size_t headLength = std::min(length, kFlashRegionSize);
        size_t tailStart = length > kFlashRegionSize ? length - kFlashRegionSize : 0;
        size_t tailLength = length - tailStart;
        size_t headAnalysisLen = std::min(length, kBlockEntropyRegionSize);
        size_t tailAnalysisLen = std::min(tailLength, kBlockEntropyRegionSize);
        size_t tailAnalysisEnd = tailStart + tailAnalysisLen;

        int fullZeroRun = 0;
        int fullNonZeroRun = 0;
        int headZeroRun = 0;
        int headNonZeroRun = 0;

        std::array<long long, 256> block256Counts{};
        size_t block256Index = 0;
        size_t block256Count = (length + 255) / 256;
        double minEntropy = std::numeric_limits<double>::max();
        double maxEntropy = std::numeric_limits<double>::lowest();
        double totalEntropy = 0;
        double totalEntropySq = 0;
        size_t minIndex = 0;
        size_t maxIndex = 0;
        double firstEntropy = 0;
        double lastEntropy = 0;

        std::array<long long, 256> headCounts{};
        size_t headBlockIndex = 0;
        size_t headBlockCount = (headAnalysisLen + 4095) / 4096;
        double headMin = std::numeric_limits<double>::max();
        double headMax = std::numeric_limits<double>::lowest();
        double headTotal = 0;
        double headTotalSq = 0;

        std::array<long long, 256> tailCounts{};
        size_t tailBlockIndex = 0;
        size_t tailBlockCount = (tailAnalysisLen + 4095) / 4096;
        double tailMin = std::numeric_limits<double>::max();
        double tailMax = std::numeric_limits<double>::lowest();
        double tailTotal = 0;
        double tailTotalSq = 0;

        for (size_t i = 0; i < length; i++)
        {
            std::uint8_t b = data[i];
            std::uint8_t cls = kByteClass[b];

            if (needFull)
            {
                AccumulateStats(result.Full, b, cls, fullZeroRun, fullNonZeroRun);

                block256Counts[b]++;
                if ((i + 1) % 256 == 0 || i == length - 1)
                {
                    size_t blockSize = (i + 1) - block256Index * 256;
                    double blockEntropy = ComputeEntropy(block256Counts, blockSize);
                    if (block256Index == 0)
                        firstEntropy = blockEntropy;
                    if (block256Index == block256Count - 1)
                        lastEntropy = blockEntropy;
                    if (blockEntropy < minEntropy)
                    {
                        minEntropy = blockEntropy;
                        minIndex = block256Index;
                    }
                    if (blockEntropy > maxEntropy)
                    {
                        maxEntropy = blockEntropy;
                        maxIndex = block256Index;
                    }
                    totalEntropy += blockEntropy;
                    totalEntropySq += blockEntropy * blockEntropy;
                    block256Counts.fill(0);
                    block256Index++;
                }
            }

            if (needFlash && i < headLength)
                AccumulateStats(result.Head, b, cls, headZeroRun, headNonZeroRun);

            if (i < headAnalysisLen)
            {
                headCounts[b]++;
                if ((i + 1) % 4096 == 0 || i == headAnalysisLen - 1)
                {
                    size_t blockSize = (i + 1) - headBlockIndex * 4096;
                    double blockEntropy = ComputeEntropy(headCounts, blockSize);
                    if (blockEntropy < headMin)
                        headMin = blockEntropy;
                    if (blockEntropy > headMax)
                        headMax = blockEntropy;
                    headTotal += blockEntropy;
                    headTotalSq += blockEntropy * blockEntropy;
                    headCounts.fill(0);
                    headBlockIndex++;
                }
            }

            if (needFlash && i >= tailStart && i < tailAnalysisEnd)
            {
                size_t local = i - tailStart;
                tailCounts[b]++;
                if ((local + 1) % 4096 == 0 || local == tailAnalysisLen - 1)
                {
                    size_t blockSize = (local + 1) - tailBlockIndex * 4096;
                    double blockEntropy = ComputeEntropy(tailCounts, blockSize);
                    if (blockEntropy < tailMin)
                        tailMin = blockEntropy;
                    if (blockEntropy > tailMax)
                        tailMax = blockEntropy;
                    tailTotal += blockEntropy;
                    tailTotalSq += blockEntropy * blockEntropy;
                    tailCounts.fill(0);
                    tailBlockIndex++;
                }
            }
        }

        if (needFull)
            FinalizeRuns(result.Full, fullZeroRun, fullNonZeroRun);
        if (needFlash)
            FinalizeRuns(result.Head, headZeroRun, headNonZeroRun);

        if (needFull)
        {
            result.MinBlockEntropy = minEntropy;
            result.MaxBlockEntropy = maxEntropy;
            result.MeanBlockEntropy = totalEntropy / block256Count;
            double var = (totalEntropySq / block256Count) - (result.MeanBlockEntropy * result.MeanBlockEntropy);
            result.BlockEntropyVariance = var < 0 ? 0 : var;
            result.MinPosition = block256Count > 1 ? static_cast<double>(minIndex) / (block256Count - 1) : 0;
            result.MaxPosition = block256Count > 1 ? static_cast<double>(maxIndex) / (block256Count - 1) : 0;
            result.FirstEntropy = firstEntropy;
            result.LastEntropy = lastEntropy;
        }

        result.HeadMin = headMin;
        result.HeadMax = headMax;
        result.HeadMean = headTotal / headBlockCount;
        double headVar = (headTotalSq / headBlockCount) - (result.HeadMean * result.HeadMean);
        result.HeadVar = headVar < 0 ? 0 : headVar;

        if (needFlash)
        {
            result.TailMin = tailMin;
            result.TailMax = tailMax;
            result.TailMean = tailTotal / tailBlockCount;
            double tailVar = (tailTotalSq / tailBlockCount) - (result.TailMean * result.TailMean);
            result.TailVar = tailVar < 0 ? 0 : tailVar;
        }

        return result;
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

    void AppendStandardFeaturesFromUnified(const std::vector<std::uint8_t>& bytes, const UnifiedScanResult& unified, std::vector<float>& features)
    {
        features.reserve(features.size() + kStandardFeatureCount);

        const CommonStats& stats = unified.Full;
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
        double minBlockEntropy = unified.MinBlockEntropy;
        double maxBlockEntropy = unified.MaxBlockEntropy;
        double meanBlockEntropy = unified.MeanBlockEntropy;
        double blockEntropyVariance = unified.BlockEntropyVariance;
        double minEntropyBlockPosition = unified.MinPosition;
        double maxEntropyBlockPosition = unified.MaxPosition;
        double firstBlockEntropy = unified.FirstEntropy;
        double lastBlockEntropy = unified.LastEntropy;

        double meanByteValue = 0;
        double byteValueVariance = 0;
        double skewness = 0;
        double kurtosis = 0;
        ComputeByteMoments(stats.Counts, bytes.size(), meanByteValue, byteValueVariance, skewness, kurtosis);

        double lowByteRatio = 0;
        double printableAsciiRatio = 0;
        double extendedAsciiRatio = 0;
        ComputeByteRangeRatios(stats.Counts, bytes.size(), lowByteRatio, printableAsciiRatio, extendedAsciiRatio);

        double headBlockEntropyMin = unified.HeadMin;
        double headBlockEntropyMax = unified.HeadMax;
        double headBlockEntropyMean = unified.HeadMean;
        double headBlockEntropyVar = unified.HeadVar;

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

    void AppendStandardFeatures(const std::vector<std::uint8_t>& bytes, std::vector<float>& features)
    {
        UnifiedScanResult unified = ComputeUnifiedScan(bytes.data(), bytes.size(), true, false);
        AppendStandardFeaturesFromUnified(bytes, unified, features);
    }

    void AppendFlashFeatureValues(const CommonStats& stats, size_t headLength, size_t totalSize,
                                  double hMin, double hMax, double hMean, double hVar,
                                  double tMin, double tMax, double tMean, double tVar,
                                  float* peValues, std::vector<float>& features)
    {
        features.reserve(features.size() + kFlashFeatureCount);

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

        features.push_back(static_cast<float>(std::log(static_cast<double>(totalSize) + 1.0)));
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

        for (int i = 0; i < 5; i++)
            features.push_back(peValues[i]);
    }

    void AppendFlashFeaturesFromUnified(const std::vector<std::uint8_t>& bytes, const UnifiedScanResult& unified, std::vector<float>& features)
    {
        size_t headLength = std::min(bytes.size(), kFlashRegionSize);

        float peValues[5]{};
        ParsePeHeader(bytes, peValues);

        AppendFlashFeatureValues(
            unified.Head,
            headLength,
            bytes.size(),
            unified.HeadMin,
            unified.HeadMax,
            unified.HeadMean,
            unified.HeadVar,
            unified.TailMin,
            unified.TailMax,
            unified.TailMean,
            unified.TailVar,
            peValues,
            features);
    }

    // T4：从分区域读取的 head/tail 计算 Flash 特征（数学等价于全量统一扫描：
    // stats=head 区域统计；h*=head 4096 块熵；t*=tail 4096 块熵；小文件 tail 复用 head）。
    void AppendFlashFeaturesFromRegions(const std::vector<std::uint8_t>& head,
                                        const std::vector<std::uint8_t>& tail,
                                        size_t totalSize,
                                        std::vector<float>& features)
    {
        size_t headLength = head.size();

        CommonStats stats = ComputeCommonStats(head.data(), headLength);

        double hMin = 0;
        double hMax = 0;
        double hMean = 0;
        double hVar = 0;
        ComputeRegionBlockEntropy(head.data(), headLength, 4096, kBlockEntropyRegionSize, hMin, hMax, hMean, hVar);

        double tMin = hMin;
        double tMax = hMax;
        double tMean = hMean;
        double tVar = hVar;
        if (!tail.empty())
            ComputeRegionBlockEntropy(tail.data(), tail.size(), 4096, kBlockEntropyRegionSize, tMin, tMax, tMean, tVar);

        float peValues[5]{};
        ParsePeHeader(head, peValues);

        AppendFlashFeatureValues(
            stats,
            headLength,
            totalSize,
            hMin,
            hMax,
            hMean,
            hVar,
            tMin,
            tMax,
            tMean,
            tVar,
            peValues,
            features);
    }

    void AppendFlashFeatures(const std::vector<std::uint8_t>& bytes, std::vector<float>& features)
    {
        UnifiedScanResult unified = ComputeUnifiedScan(bytes.data(), bytes.size(), false, true);
        AppendFlashFeaturesFromUnified(bytes, unified, features);
    }

    bool IsProFeatureCount(int featureCount)
    {
        return featureCount == kProHybridFeatureCount;
    }

    void AppendProSectionStats(const std::vector<std::uint8_t>& bytes, size_t start, size_t length,
                                size_t fileSize, std::vector<float>& features, size_t offset)
    {
        if (length == 0 || start >= bytes.size())
            return;

        size_t actualLength = std::min(length, bytes.size() - start);
        if (actualLength == 0)
            return;

        std::array<long long, 256> counts{};
        int printableCount = 0;
        int letterCount = 0;
        int digitCount = 0;
        int highByteCount = 0;
        int zeroCount = 0;
        int maxZeroRun = 0;
        int currentZeroRun = 0;

        for (size_t i = 0; i < actualLength; i++)
        {
            std::uint8_t b = bytes[start + i];
            counts[b]++;

            if (b == 0)
            {
                zeroCount++;
                currentZeroRun++;
                if (currentZeroRun > maxZeroRun)
                    maxZeroRun = currentZeroRun;
            }
            else
            {
                currentZeroRun = 0;
            }

            if (b >= 0x80)
                highByteCount++;

            if (b >= 32 && b <= 126)
            {
                printableCount++;
                if ((b >= 65 && b <= 90) || (b >= 97 && b <= 122))
                    letterCount++;
                else if (b >= 48 && b <= 57)
                    digitCount++;
            }
        }

        double entropy = ComputeEntropy(counts, actualLength);
        features[offset + 0] = static_cast<float>(entropy);

        for (int bin = 0; bin < 32; bin++)
        {
            long long sum = 0;
            for (int j = 0; j < 8; j++)
                sum += counts[bin * 8 + j];
            features[offset + 1 + bin] = actualLength > 0
                ? static_cast<float>(sum) / static_cast<float>(actualLength)
                : 0.0f;
        }

        features[offset + 33] = actualLength > 0 ? static_cast<float>(printableCount) / static_cast<float>(actualLength) : 0.0f;
        features[offset + 34] = actualLength > 0 ? static_cast<float>(zeroCount) / static_cast<float>(actualLength) : 0.0f;
        features[offset + 35] = actualLength > 0 ? static_cast<float>(highByteCount) / static_cast<float>(actualLength) : 0.0f;
        features[offset + 36] = actualLength > 0 ? static_cast<float>(letterCount) / static_cast<float>(actualLength) : 0.0f;
        features[offset + 37] = actualLength > 0 ? static_cast<float>(digitCount) / static_cast<float>(actualLength) : 0.0f;
        features[offset + 38] = actualLength > 0 ? static_cast<float>(maxZeroRun) / static_cast<float>(actualLength) : 0.0f;
        features[offset + 39] = fileSize > 0 ? static_cast<float>(actualLength) / static_cast<float>(fileSize) : 0.0f;
    }

    void AppendProRawStatFeatures(const std::vector<std::uint8_t>& bytes, std::vector<float>& features)
    {
        size_t base = features.size();
        features.resize(base + kProRawStatFeatureCount, 0.0f);

        size_t fileSize = bytes.size();
        size_t sectionSize = kProRawStatSectionSize;

        size_t headLength = std::min(fileSize, sectionSize);
        AppendProSectionStats(bytes, 0, headLength, fileSize, features, base);

        size_t midStart = fileSize / 2 > sectionSize / 2 ? fileSize / 2 - sectionSize / 2 : 0;
        size_t midLength = std::min(midStart + sectionSize, fileSize) - midStart;
        AppendProSectionStats(bytes, midStart, midLength, fileSize, features, base + kProRawStatFeaturesPerSection);

        size_t tailStart = fileSize > sectionSize ? fileSize - sectionSize : 0;
        size_t tailLength = fileSize - tailStart;
        AppendProSectionStats(bytes, tailStart, tailLength, fileSize, features, base + kProRawStatFeaturesPerSection * 2);
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

    bool ExtractFeaturesForMode(int mode, int featureCount, const std::vector<std::uint8_t>& bytes, std::vector<float>& features)
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
        {
            if (featureCount != kProHybridFeatureCount && featureCount != 4)
                return false;

            UnifiedScanResult unified = ComputeUnifiedScan(bytes.data(), bytes.size(), true, true);
            AppendStandardFeaturesFromUnified(bytes, unified, features);
            AppendFlashFeaturesFromUnified(bytes, unified, features);
            AppendProRawStatFeatures(bytes, features);
            AppendProStructuralFeatures(bytes, features);
            return features.size() == static_cast<size_t>(kProHybridFeatureCount);
        }
        default:
            AppendStandardFeatures(bytes, features);
            return features.size() == kStandardFeatureCount;
        }
    }

    int DefaultFeatureCountForMode(int mode)
    {
        switch (mode)
        {
        case XdowsModelNativeModeFlash:
            return kFlashFeatureCount;
        case XdowsModelNativeModePro:
            return kProHybridFeatureCount;
        case XdowsModelNativeModeAdaptive:
            return 0;
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
        case XdowsModelNativeModeAdaptive:
            return L"Xdows-Model.onnx";
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
        case XdowsModelNativeModeAdaptive:
            return L"Adaptive";
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

    bool RunOnnxSession(Ort::Session* session, int featureCount, const std::vector<float>& features,
                        float& probability, std::wstring& error)
    {
        probability = 0;
        error.clear();

        if (session == nullptr)
        {
            error = L"session-not-ready";
            return false;
        }

        if (static_cast<int>(features.size()) != featureCount)
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
            auto outputs = session->Run(
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

    bool RunOnnx(NativeSession* session, const std::vector<float>& features, float& probability, std::wstring& error)
    {
        if (session == nullptr || session->Session == nullptr)
        {
            error = L"session-not-ready";
            return false;
        }

        if (session->Mode != XdowsModelNativeModePro || session->FeatureCount != 4)
            return RunOnnxSession(session->Session.get(), session->FeatureCount, features, probability, error);

        if (features.size() != static_cast<size_t>(kProHybridFeatureCount))
        {
            error = L"pro-hybrid-feature-count-mismatch";
            return false;
        }

        const std::array<size_t, 4> offsets{
            0,
            kStandardFeatureCount,
            kStandardFeatureCount + kFlashFeatureCount,
            kStandardFeatureCount + kFlashFeatureCount + kProRawStatFeatureCount };

        // T5：与 Managed ProEnsembleSession.Parallel.For 对齐，4 个分支 session 并行推理。
        // 每分支独立 Ort::Session（可并发 Run），error 线程局部，避免数据竞争。
        std::vector<float> fusionFeatures(4, 0.0f);
        std::vector<std::future<bool>> futures;
        std::vector<std::wstring> branchErrors(session->ProBranchSessions.size());
        std::vector<float> branchProbabilities(session->ProBranchSessions.size(), 0.0f);
        futures.reserve(session->ProBranchSessions.size());

        for (size_t i = 0; i < session->ProBranchSessions.size(); i++)
        {
            int count = session->ProBranchFeatureCounts[i];
            futures.push_back(std::async(std::launch::async, [&, i, count]() -> bool
            {
                std::vector<float> branchFeatures(
                    features.begin() + static_cast<std::ptrdiff_t>(offsets[i]),
                    features.begin() + static_cast<std::ptrdiff_t>(offsets[i] + count));
                return RunOnnxSession(session->ProBranchSessions[i].get(), count, branchFeatures,
                                      branchProbabilities[i], branchErrors[i]);
            }));
        }

        for (size_t i = 0; i < futures.size(); i++)
        {
            bool ok = futures[i].get();
            if (!ok && error.empty() && !branchErrors[i].empty())
                error = branchErrors[i];
            if (!ok)
                return false;
            fusionFeatures[i] = branchProbabilities[i] / 100.0f;
        }

        return RunOnnxSession(session->Session.get(), 4, fusionFeatures, probability, error);
    }

// T4：Adaptive 闪存阶段用分区域读取（只读 head+tail），escalate 才全量读。
    // 与 Managed AdaptiveModelSession.ScanFile 语义一致（flash 阶段 FlashFeatureExtractor.ExtractFeatures(filePath)）。
    bool RunAdaptive(NativeSession* session,
                     const std::filesystem::path& path,
                     const std::vector<std::uint8_t>& head,
                     const std::vector<std::uint8_t>& tail,
                     size_t totalSize,
                     float& probability, int& finalMode, std::wstring& error)
    {
        if (session == nullptr || session->AdaptiveFlash == nullptr ||
            session->AdaptiveStandard == nullptr || session->AdaptivePro == nullptr)
        {
            error = L"adaptive-session-not-ready";
            return false;
        }

        std::vector<float> flashFeatures;
        AppendFlashFeaturesFromRegions(head, tail, totalSize, flashFeatures);
        if (!RunOnnx(session->AdaptiveFlash.get(), flashFeatures, probability, error))
            return false;
        if (probability <= 100.0f - ThresholdForMode(XdowsModelNativeModeFlash))
        {
            finalMode = XdowsModelNativeModeFlash;
            return true;
        }

        std::vector<std::uint8_t> bytes;
        if (!ReadAllBytes(path, bytes))
        {
            error = L"read-failed";
            return false;
        }

        std::vector<float> standardFeatures;
        AppendStandardFeatures(bytes, standardFeatures);
        if (!RunOnnx(session->AdaptiveStandard.get(), standardFeatures, probability, error))
            return false;
        if (probability <= 100.0f - ThresholdForMode(XdowsModelNativeModeStandard))
        {
            finalMode = XdowsModelNativeModeStandard;
            return true;
        }

        std::vector<float> features;
        features.reserve(kProHybridFeatureCount);
        features.insert(features.end(), standardFeatures.begin(), standardFeatures.end());
        features.insert(features.end(), flashFeatures.begin(), flashFeatures.end());
        AppendProRawStatFeatures(bytes, features);
        AppendProStructuralFeatures(bytes, features);
        if (!RunOnnx(session->AdaptivePro.get(), features, probability, error))
            return false;
        finalMode = XdowsModelNativeModePro;
        return true;
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

    if (mode < XdowsModelNativeModeStandard || mode > XdowsModelNativeModeAdaptive)
        return XdowsModelNativeStatusInvalidArgument;

    std::filesystem::path modelPath = ResolveModelPath(modelDirectory, mode);
    if (modelPath.empty())
        return XdowsModelNativeStatusModelNotFound;

    try
    {
        auto nativeSession = std::make_unique<NativeSession>(mode, DefaultFeatureCountForMode(mode), modelPath);
        *session = nativeSession.release();
        return XdowsModelNativeStatusOk;
    }
    catch (...)
    {
        return XdowsModelNativeStatusInternalError;
    }
}

// T4：Flash/Adaptive 走分区域读取（只读 head+tail），Standard/Pro 保持全量读取。
    bool ScanViaRegions(NativeSession* nativeSession,
                        const std::filesystem::path& path,
                        XDOWS_MODEL_NATIVE_SCAN_RESULT* result)
    {
        std::vector<std::uint8_t> head;
        std::vector<std::uint8_t> tail;
        size_t totalSize = 0;
        if (!ReadFileRegions(path, head, tail, totalSize))
        {
            SetError(result, XdowsModelNativeStatusInternalError, L"read-failed");
            return false;
        }

        if (ContainsAscii(head, "EICAR-STANDARD-ANTIVIRUS-TEST-FILE") ||
            ContainsAscii(tail, "EICAR-STANDARD-ANTIVIRUS-TEST-FILE"))
        {
            result->Status = XdowsModelNativeStatusOk;
            result->IsThreat = 1;
            result->Probability = 100.0f;
            result->DetectionName = DuplicateString(L"Xdows.Model.EICAR");
            return true;
        }

        if (!IsPeFile(head))
        {
            result->Status = XdowsModelNativeStatusOk;
            result->IsThreat = 0;
            result->Probability = 0.0f;
            return true;
        }

        float probability = 0;
        int decisionMode = nativeSession->Mode;
        std::wstring error;
        if (nativeSession->Mode == XdowsModelNativeModeAdaptive)
        {
            if (!RunAdaptive(nativeSession, path, head, tail, totalSize, probability, decisionMode, error))
            {
                SetError(result, XdowsModelNativeStatusInternalError, error.empty() ? L"adaptive-run-failed" : error);
                return false;
            }
        }
        else
        {
            std::vector<float> features;
            AppendFlashFeaturesFromRegions(head, tail, totalSize, features);
            if (!RunOnnx(nativeSession, features, probability, error))
            {
                SetError(result, XdowsModelNativeStatusInternalError, error.empty() ? L"onnx-run-failed" : error);
                return false;
            }
        }

        result->Status = XdowsModelNativeStatusOk;
        result->Probability = probability;
        if (probability >= ThresholdForMode(decisionMode))
        {
            result->IsThreat = 1;
            result->DetectionName = DuplicateString(
                L"Xdows.Model." + ModeName(decisionMode) + L".Probability" +
                std::to_wstring(static_cast<int>(probability)));
        }
        else
        {
            result->IsThreat = 0;
        }

        return true;
    }

    bool ScanViaFullRead(NativeSession* nativeSession,
                         const std::filesystem::path& path,
                         XDOWS_MODEL_NATIVE_SCAN_RESULT* result)
    {
        std::vector<std::uint8_t> bytes;
        if (!ReadAllBytes(path, bytes))
        {
            SetError(result, XdowsModelNativeStatusInternalError, L"read-failed");
            return false;
        }

        if (ContainsAscii(bytes, "EICAR-STANDARD-ANTIVIRUS-TEST-FILE"))
        {
            result->Status = XdowsModelNativeStatusOk;
            result->IsThreat = 1;
            result->Probability = 100.0f;
            result->DetectionName = DuplicateString(L"Xdows.Model.EICAR");
            return true;
        }

        if (!IsPeFile(bytes))
        {
            result->Status = XdowsModelNativeStatusOk;
            result->IsThreat = 0;
            result->Probability = 0.0f;
            return true;
        }

        float probability = 0;
        int decisionMode = nativeSession->Mode;
        std::wstring error;
        if (nativeSession->Mode == XdowsModelNativeModeAdaptive)
        {
            std::vector<std::uint8_t> head;
            std::vector<std::uint8_t> tail;
            size_t totalSize = bytes.size();
            size_t headLength = std::min(totalSize, kFlashRegionSize);
            head.assign(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(headLength));
            if (totalSize > kFlashRegionSize)
                tail.assign(bytes.end() - static_cast<std::ptrdiff_t>(kFlashRegionSize), bytes.end());

            if (!RunAdaptive(nativeSession, path, head, tail, totalSize, probability, decisionMode, error))
            {
                SetError(result, XdowsModelNativeStatusInternalError, error.empty() ? L"adaptive-run-failed" : error);
                return false;
            }
        }
        else
        {
            std::vector<float> features;
            if (!ExtractFeaturesForMode(nativeSession->Mode, nativeSession->FeatureCount, bytes, features))
            {
                SetError(result, XdowsModelNativeStatusUnsupportedFile, L"feature-extraction-failed");
                return false;
            }
            if (!RunOnnx(nativeSession, features, probability, error))
            {
                SetError(result, XdowsModelNativeStatusInternalError, error.empty() ? L"onnx-run-failed" : error);
                return false;
            }
        }

        result->Status = XdowsModelNativeStatusOk;
        result->Probability = probability;
        if (probability >= ThresholdForMode(decisionMode))
        {
            result->IsThreat = 1;
            result->DetectionName = DuplicateString(
                L"Xdows.Model." + ModeName(decisionMode) + L".Probability" +
                std::to_wstring(static_cast<int>(probability)));
        }
        else
        {
            result->IsThreat = 0;
        }

        return true;
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

    bool ok;
    if (nativeSession->Mode == XdowsModelNativeModeStandard ||
        nativeSession->Mode == XdowsModelNativeModePro)
    {
        ok = ScanViaFullRead(nativeSession, path, result);
    }
    else
    {
        ok = ScanViaRegions(nativeSession, path, result);
    }

    return ok ? XdowsModelNativeStatusOk : result->Status;
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
