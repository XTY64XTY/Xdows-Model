using System.Text;
using Xdows_Model_Config;

namespace Xdows_Model_Invoker;

/// <summary>
/// 导入表解析状态。导入特征允许"缺失"（例如 PE 没有导入目录），
/// 但必须显式表达原因，调用方与统计特征都能区分"解析成功"与"没有数据"。
/// </summary>
public enum ImportParseStatus
{
    /// <summary>导入表完整解析。</summary>
    Ok = 0,

    /// <summary>PE 没有导入目录（如纯资源文件）。哈希桶为全零，属正常缺失。</summary>
    NoImportTable = 1,

    /// <summary>解析中途越界，已解析的部分保留，其余置零。</summary>
    Truncated = 2,

    /// <summary>导入目录存在但完全无法解析（RVA 无法映射或首描述符即损坏）。</summary>
    Invalid = 3
}

/// <summary>
/// ImportBehavior 特征组：DLL 哈希桶（<see cref="ImportHashConfig.DllHashDimensions"/> 维）+
/// API 哈希桶（<see cref="ImportHashConfig.ApiHashDimensions"/> 维）+ 固定统计特征
/// （<see cref="FeatureSchema.ProImportStatsCount"/> 维），总长度固定、顺序稳定。
/// </summary>
public sealed class ImportBehaviorFeatures
{
    public const int FeatureCount = FeatureSchema.ProImportBehaviorCount;

    private readonly float[] _features = new float[FeatureCount];

    public ImportParseStatus Status { get; internal set; }

    public int DllCount { get; internal set; }
    public int ApiCount { get; internal set; }
    public int DelayDllCount { get; internal set; }
    public int DelayApiCount { get; internal set; }

    internal Span<float> DllBuckets => _features.AsSpan(0, ImportHashConfig.DllHashDimensions);
    internal Span<float> ApiBuckets => _features.AsSpan(ImportHashConfig.DllHashDimensions, ImportHashConfig.ApiHashDimensions);
    internal Span<float> Stats => _features.AsSpan(ImportHashConfig.DllHashDimensions + ImportHashConfig.ApiHashDimensions, FeatureSchema.ProImportStatsCount);

    public float[] ToFloatArray()
    {
        var result = new float[FeatureCount];
        Array.Copy(_features, result, FeatureCount);
        return result;
    }

    public void WriteTo(Span<float> destination)
    {
        if (destination.Length != FeatureCount)
            throw new ArgumentException($"ImportBehavior 目标缓冲区长度必须为 {FeatureCount}。", nameof(destination));
        _features.AsSpan().CopyTo(destination);
    }
}

/// <summary>
/// 从 PE 导入表与延迟导入表提取 ImportBehavior 特征。
/// 训练端与推理端使用同一实现，保证特征顺序、维度与哈希规则完全一致。
/// 哈希算法、种子与维度见 <see cref="ImportHashConfig"/>，为编译期显式常量。
/// </summary>
public static class ImportFeatureExtractor
{
    private const int MaxImportDescriptors = 4096;
    private const int MaxApisPerDll = 65535;
    private const int MaxTotalApis = 1_000_000;
    private const int MaxNameBytes = 512;
    private const uint FnvPrime = 16777619u;

    public static ImportBehaviorFeatures ExtractFeatures(string filePath)
    {
        return ExtractFromBytes(File.ReadAllBytes(filePath));
    }

    /// <summary>
    /// 从流中提取。PE 解析需要随机访问，流内容会先完整读入内存；
    /// 流的 I/O 错误不向调用方隐瞒，直接抛出。
    /// </summary>
    public static ImportBehaviorFeatures ExtractFromStream(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        using var memory = new MemoryStream();
        stream.CopyTo(memory);
        return ExtractFromBytes(memory.ToArray());
    }

    public static ImportBehaviorFeatures ExtractFromBytes(byte[] bytes)
    {
        ArgumentNullException.ThrowIfNull(bytes);
        if (bytes.Length < 64 || !ByteAnalysisHelper.IsPeFile(bytes))
            throw new NotSupportedException("不支持该文件类型");

        var result = new ImportBehaviorFeatures();
        var context = PeImportContext.Create(bytes);
        if (context is null)
        {
            // 头结构损坏到无法读取可选头：与 ProHybridFeatureExtractor 的准入检查不一致，属异常输入。
            result.Status = ImportParseStatus.Invalid;
            WriteStats(result, duplicateDllCount: 0, ordinalApiCount: 0, delayOrdinalApiCount: 0, importDirectorySize: 0, namedApiCount: 0);
            return result;
        }

        var dllNames = new List<string>();
        var seenDllNames = new HashSet<string>(StringComparer.Ordinal);
        int duplicateDllCount = 0;
        int ordinalApiCount = 0;
        int namedApiCount = 0;

        bool hasImportDirectory = context.ImportDirectoryRva != 0;
        ImportParseStatus status = hasImportDirectory ? ImportParseStatus.Ok : ImportParseStatus.NoImportTable;

        if (hasImportDirectory)
        {
            status = ParseImportDirectory(
                context,
                context.ImportDirectoryRva,
                descriptorSize: 20,
                isDelay: false,
                result,
                dllNames,
                seenDllNames,
                ref duplicateDllCount,
                ref ordinalApiCount,
                ref namedApiCount);
        }

        int delayOrdinalApiCount = 0;
        if (context.DelayImportDirectoryRva != 0)
        {
            ImportParseStatus delayStatus = ParseImportDirectory(
                context,
                context.DelayImportDirectoryRva,
                descriptorSize: 32,
                isDelay: true,
                result,
                dllNames,
                seenDllNames,
                ref duplicateDllCount,
                ref delayOrdinalApiCount,
                ref namedApiCount);

            if (status == ImportParseStatus.Ok && delayStatus != ImportParseStatus.Ok)
                status = delayStatus == ImportParseStatus.Invalid && result.DllCount > 0
                    ? ImportParseStatus.Truncated
                    : delayStatus;
        }

        result.Status = result.DllCount == 0 && hasImportDirectory && status == ImportParseStatus.Ok
            ? ImportParseStatus.NoImportTable
            : status;

        WriteStats(result, duplicateDllCount, ordinalApiCount, delayOrdinalApiCount, context.ImportDirectorySize, namedApiCount);
        return result;
    }

    private static ImportParseStatus ParseImportDirectory(
        PeImportContext context,
        uint directoryRva,
        int descriptorSize,
        bool isDelay,
        ImportBehaviorFeatures result,
        List<string> dllNames,
        HashSet<string> seenDllNames,
        ref int duplicateDllCount,
        ref int ordinalApiCount,
        ref int namedApiCount)
    {
        byte[] bytes = context.Bytes;
        if (!context.TryMapRva(directoryRva, out int descriptorOffset))
            return ImportParseStatus.Invalid;

        ImportParseStatus status = ImportParseStatus.Ok;
        int parsedDescriptors = 0;

        for (int index = 0; index < MaxImportDescriptors; index++)
        {
            long offset = (long)descriptorOffset + (long)index * descriptorSize;
            if (offset + descriptorSize > bytes.Length)
            {
                status = parsedDescriptors == 0 ? ImportParseStatus.Invalid : ImportParseStatus.Truncated;
                break;
            }

            uint field0 = ReadUInt32(bytes, (int)offset);
            uint nameRva = ReadUInt32(bytes, (int)offset + (isDelay ? 4 : 12));
            uint firstThunkRva = ReadUInt32(bytes, (int)offset + (isDelay ? 12 : 16));
            uint lookupThunkRva = isDelay ? ReadUInt32(bytes, (int)offset + 16) : field0;

            // 延迟导入描述符（grAttrs bit0 = 1 时字段为 RVA）。旧格式（bit0 = 0）使用虚拟地址，
            // 需要 ImageBase 换算，这里统一按 RVA 解析，旧格式会因 RVA 映射失败而记为截断。
            if (isDelay && (field0 & 1) == 0)
            {
                if (field0 == 0 && nameRva == 0 && firstThunkRva == 0)
                    break;
                return parsedDescriptors == 0 && result.DllCount == 0 ? ImportParseStatus.Invalid : ImportParseStatus.Truncated;
            }

            if (field0 == 0 && nameRva == 0 && firstThunkRva == 0 && lookupThunkRva == 0)
                break;

            parsedDescriptors++;
            if (lookupThunkRva == 0)
                lookupThunkRva = firstThunkRva;

            string dllName = context.ReadAsciiString(nameRva, MaxNameBytes) ?? string.Empty;
            if (dllName.Length == 0)
            {
                status = ImportParseStatus.Truncated;
                continue;
            }

            string normalizedDll = ImportHashConfig.NormalizeDllName(dllName);
            if (!seenDllNames.Add(normalizedDll))
                duplicateDllCount++;

            int bucket = HashToBucket(normalizedDll, ImportHashConfig.DllHashDimensions);
            result.DllBuckets[bucket] = 1.0f;
            dllNames.Add(normalizedDll);

            if (isDelay)
                result.DelayDllCount++;
            else
                result.DllCount++;

            int apiCount = ParseThunkTable(
                context,
                lookupThunkRva,
                normalizedDll,
                result,
                ref ordinalApiCount,
                ref namedApiCount);

            if (apiCount < 0)
            {
                status = ImportParseStatus.Truncated;
                continue;
            }

            if (isDelay)
                result.DelayApiCount += apiCount;
            else
                result.ApiCount += apiCount;

            if (result.DllCount + result.DelayDllCount > MaxImportDescriptors ||
                result.ApiCount + result.DelayApiCount > MaxTotalApis)
            {
                return ImportParseStatus.Truncated;
            }
        }

        if (parsedDescriptors == 0 && status == ImportParseStatus.Ok)
            return ImportParseStatus.Invalid;

        return status;
    }

    /// <summary>解析一个 DLL 的 thunk 表，返回按名/按序数导入的 API 数；表本身无法映射时返回 -1。</summary>
    private static int ParseThunkTable(
        PeImportContext context,
        uint thunkTableRva,
        string normalizedDll,
        ImportBehaviorFeatures result,
        ref int ordinalApiCount,
        ref int namedApiCount)
    {
        byte[] bytes = context.Bytes;
        if (!context.TryMapRva(thunkTableRva, out int thunkOffset))
            return -1;

        int thunkSize = context.IsPe32Plus ? 8 : 4;
        ulong ordinalFlag = context.IsPe32Plus ? 0x8000000000000000UL : 0x80000000UL;
        int apiCount = 0;

        for (int index = 0; index < MaxApisPerDll; index++)
        {
            long offset = (long)thunkOffset + (long)index * thunkSize;
            if (offset + thunkSize > bytes.Length)
                break;

            ulong thunkValue = context.IsPe32Plus
                ? ReadUInt64(bytes, (int)offset)
                : ReadUInt32(bytes, (int)offset);

            if (thunkValue == 0)
                break;

            string apiKey;
            if ((thunkValue & ordinalFlag) != 0)
            {
                apiKey = normalizedDll + "!#" + (thunkValue & 0xFFFF).ToString(System.Globalization.CultureInfo.InvariantCulture);
                ordinalApiCount++;
            }
            else
            {
                string? apiName = context.ReadAsciiString((uint)(thunkValue & 0x7FFFFFFF) , MaxNameBytes, skipHint: true);
                if (string.IsNullOrEmpty(apiName))
                    continue;
                apiKey = ImportHashConfig.NormalizeApiName(apiName);
                namedApiCount++;
            }

            int bucket = HashToBucket(apiKey, ImportHashConfig.ApiHashDimensions);
            result.ApiBuckets[bucket] = 1.0f;
            apiCount++;
        }

        return apiCount;
    }

    /// <summary>统计特征布局（16 维，顺序固定，新增维度必须同步提升 Schema 版本）。</summary>
    private static void WriteStats(
        ImportBehaviorFeatures result,
        int duplicateDllCount,
        int ordinalApiCount,
        int delayOrdinalApiCount,
        uint importDirectorySize,
        int namedApiCount)
    {
        Span<float> stats = result.Stats;
        int totalDlls = result.DllCount + result.DelayDllCount;
        int totalApis = result.ApiCount + result.DelayApiCount;
        int dllBuckets = ImportHashConfig.DllHashDimensions;
        int apiBuckets = ImportHashConfig.ApiHashDimensions;

        int occupiedDll = 0;
        foreach (float v in result.DllBuckets)
            if (v != 0) occupiedDll++;
        int occupiedApi = 0;
        foreach (float v in result.ApiBuckets)
            if (v != 0) occupiedApi++;

        int idx = 0;
        stats[idx++] = (float)result.Status;
        stats[idx++] = result.Status == ImportParseStatus.NoImportTable ? 0f : 1f;
        stats[idx++] = (float)Math.Log(result.DllCount + 1);
        stats[idx++] = (float)Math.Log(totalApis + 1);
        stats[idx++] = result.DllCount > 0 ? (float)result.ApiCount / result.DllCount : 0f;
        stats[idx++] = (float)Math.Log(result.DelayDllCount + 1);
        stats[idx++] = (float)Math.Log(result.DelayApiCount + 1);
        stats[idx++] = totalApis > 0 ? (float)ordinalApiCount / totalApis : 0f;
        stats[idx++] = (float)occupiedDll / dllBuckets;
        stats[idx++] = (float)occupiedApi / apiBuckets;
        stats[idx++] = result.DllCount + result.DelayDllCount > 0
            ? (float)duplicateDllCount / (result.DllCount + result.DelayDllCount)
            : 0f;
        stats[idx++] = importDirectorySize > 0 ? (float)Math.Log(importDirectorySize + 1) : 0f;
        stats[idx++] = result.DelayDllCount > 0 ? 1f : 0f;
        stats[idx++] = totalApis > 0 ? (float)namedApiCount / totalApis : 0f;
        stats[idx++] = result.DelayApiCount > 0 ? (float)delayOrdinalApiCount / result.DelayApiCount : 0f;
        stats[idx++] = (float)Math.Log(totalDlls + 1); // 总 DLL 数（含延迟导入），供树模型拆分
    }

    /// <summary>32 位 FNV-1a：种子（偏移基值）与算法都是 <see cref="ImportHashConfig"/> 的显式常量。</summary>
    internal static int HashToBucket(string value, int dimensions)
    {
        uint hash = ImportHashConfig.Seed;
        foreach (byte b in Encoding.UTF8.GetBytes(value))
        {
            hash ^= b;
            hash *= FnvPrime;
        }
        return (int)(hash % (uint)dimensions);
    }

    private static ushort ReadUInt16(byte[] bytes, int offset)
    {
        return offset >= 0 && offset + 2 <= bytes.Length ? BitConverter.ToUInt16(bytes, offset) : (ushort)0;
    }

    private static uint ReadUInt32(byte[] bytes, int offset)
    {
        return offset >= 0 && offset + 4 <= bytes.Length ? BitConverter.ToUInt32(bytes, offset) : 0u;
    }

    private static ulong ReadUInt64(byte[] bytes, int offset)
    {
        return offset >= 0 && offset + 8 <= bytes.Length ? BitConverter.ToUInt64(bytes, offset) : 0UL;
    }

    /// <summary>
    /// 轻量 PE 导入上下文：节表驱动的 RVA→文件偏移映射与带边界的字符串读取。
    /// 与 ProHybridFeatureExtractor 的结构解析相互独立，避免牵动既有 32 维结构特征。
    /// </summary>
    private sealed class PeImportContext
    {
        private readonly (uint VirtualAddress, uint VirtualSize, uint RawSize, uint RawOffset)[] _sections;
        private readonly uint _sizeOfHeaders;

        private PeImportContext(
            byte[] bytes,
            bool isPe32Plus,
            uint importDirectoryRva,
            uint importDirectorySize,
            uint delayImportDirectoryRva,
            uint sizeOfHeaders,
            (uint, uint, uint, uint)[] sections)
        {
            Bytes = bytes;
            IsPe32Plus = isPe32Plus;
            ImportDirectoryRva = importDirectoryRva;
            ImportDirectorySize = importDirectorySize;
            DelayImportDirectoryRva = delayImportDirectoryRva;
            _sizeOfHeaders = sizeOfHeaders;
            _sections = sections;
        }

        public byte[] Bytes { get; }
        public bool IsPe32Plus { get; }
        public uint ImportDirectoryRva { get; }
        public uint ImportDirectorySize { get; }
        public uint DelayImportDirectoryRva { get; }

        public static PeImportContext? Create(byte[] bytes)
        {
            int peOffset = BitConverter.ToInt32(bytes, 60);
            if (peOffset < 0 || peOffset + 24 > bytes.Length || bytes[peOffset] != 'P' || bytes[peOffset + 1] != 'E')
                return null;

            ushort sectionCount = ReadUInt16(bytes, peOffset + 6);
            ushort optionalHeaderSize = ReadUInt16(bytes, peOffset + 20);
            int optionalHeaderOffset = peOffset + 24;
            if (optionalHeaderOffset + optionalHeaderSize > bytes.Length || optionalHeaderSize < 2)
                return null;

            ushort magic = ReadUInt16(bytes, optionalHeaderOffset);
            bool pe32 = magic == 0x10b;
            bool pe32Plus = magic == 0x20b;
            if (!pe32 && !pe32Plus)
                return null;

            int dataDirectoryOffset = optionalHeaderOffset + (pe32 ? 96 : 112);
            uint numberOfRvaAndSizes = ReadUInt32(bytes, optionalHeaderOffset + (pe32 ? 92 : 108));
            uint sizeOfHeaders = ReadUInt32(bytes, optionalHeaderOffset + 60);

            uint importRva = 0, importSize = 0, delayImportRva = 0;
            if (numberOfRvaAndSizes > 1 && dataDirectoryOffset + 16 <= bytes.Length)
            {
                importRva = ReadUInt32(bytes, dataDirectoryOffset + 8);
                importSize = ReadUInt32(bytes, dataDirectoryOffset + 12);
            }
            if (numberOfRvaAndSizes > 13 && dataDirectoryOffset + 8 * 14 <= bytes.Length)
            {
                delayImportRva = ReadUInt32(bytes, dataDirectoryOffset + 8 * 13);
            }

            int sectionTableOffset = optionalHeaderOffset + optionalHeaderSize;
            var sections = new (uint, uint, uint, uint)[Math.Min((int)sectionCount, 96)];
            for (int i = 0; i < sections.Length; i++)
            {
                int sectionOffset = sectionTableOffset + i * 40;
                if (sectionOffset + 40 > bytes.Length)
                {
                    Array.Resize(ref sections, i);
                    break;
                }
                sections[i] = (
                    ReadUInt32(bytes, sectionOffset + 12),
                    ReadUInt32(bytes, sectionOffset + 8),
                    ReadUInt32(bytes, sectionOffset + 16),
                    ReadUInt32(bytes, sectionOffset + 20));
            }

            return new PeImportContext(bytes, pe32Plus, importRva, importSize, delayImportRva, sizeOfHeaders, sections);
        }

        public bool TryMapRva(uint rva, out int fileOffset)
        {
            fileOffset = 0;
            if (rva < _sizeOfHeaders && rva < Bytes.Length)
            {
                fileOffset = (int)rva;
                return true;
            }

            foreach (var (virtualAddress, virtualSize, rawSize, rawOffset) in _sections)
            {
                uint span = Math.Max(virtualSize, rawSize);
                if (span == 0 || rva < virtualAddress || rva - virtualAddress >= span)
                    continue;

                long offset = (long)rawOffset + (rva - virtualAddress);
                if (offset < 0 || offset >= Bytes.Length)
                    return false;

                fileOffset = (int)offset;
                return true;
            }

            return false;
        }

        /// <summary>读取以 NUL 结尾的 ASCII 字符串；越界或空串返回 null，超长截断。</summary>
        public string? ReadAsciiString(uint rva, int maxBytes, bool skipHint = false)
        {
            if (!TryMapRva(rva, out int offset))
                return null;
            if (skipHint)
                offset += 2;
            if (offset >= Bytes.Length)
                return null;

            int limit = Math.Min(Bytes.Length, offset + maxBytes);
            int end = offset;
            while (end < limit && Bytes[end] != 0)
                end++;
            if (end == offset)
                return null;

            return Encoding.ASCII.GetString(Bytes, offset, end - offset);
        }
    }
}
