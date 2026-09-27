using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Xdows_Model_Config;
using Xdows_Model_Invoker;
using Xdows_Model_Maker;

namespace Xdows_Model_ArchitectureTests;

/// <summary>
/// ImportBehavior 五分支改造的架构测试：哈希稳定性、特征维度与偏移、Manifest 严格校验、
/// 旧四分支兼容、新五分支端到端训练/导出/加载/推理。
/// 全部断言都在没有真实样本目录的情况下可运行；只有 PE 样本取自系统目录的 notepad.exe。
/// </summary>
internal static class ProImportBehaviorArchitectureTests
{
    public static void Run(string peSamplePath)
    {
        AssertImportHashBucketsAreStable();
        AssertImportFeaturesAreDeterministic(peSamplePath);
        AssertFeatureLayoutIsSelfConsistent();
        AssertProBranchLayout();
        AssertProManifestRoundTrip();
        AssertProManifestRejectsInvalidContent();
        AssertProBranchResolution();
        AssertProDatasetSplit();
        AssertProFiveBranchEndToEnd();
    }

    /// <summary>DLL / API 哈希必须对同一输入稳定，且大小写规范化在哈希前生效。</summary>
    private static void AssertImportHashBucketsAreStable()
    {
        const string dll = "kernel32.dll";
        const string api = "CreateFileW";

        int dllBucket = ImportFeatureExtractor.HashToBucket(dll, ImportHashConfig.DllHashDimensions);
        int apiBucket = ImportFeatureExtractor.HashToBucket(api, ImportHashConfig.ApiHashDimensions);

        for (int attempt = 0; attempt < 8; attempt++)
        {
            if (ImportFeatureExtractor.HashToBucket(dll, ImportHashConfig.DllHashDimensions) != dllBucket)
                throw new InvalidOperationException("DLL 哈希桶在多次调用之间发生变化。");
            if (ImportFeatureExtractor.HashToBucket(api, ImportHashConfig.ApiHashDimensions) != apiBucket)
                throw new InvalidOperationException("API 哈希桶在多次调用之间发生变化。");
        }

        if (dllBucket < 0 || dllBucket >= ImportHashConfig.DllHashDimensions)
            throw new InvalidOperationException($"DLL 哈希桶 {dllBucket} 越界。");
        if (apiBucket < 0 || apiBucket >= ImportHashConfig.ApiHashDimensions)
            throw new InvalidOperationException($"API 哈希桶 {apiBucket} 越界。");

        if (ImportFeatureExtractor.HashToBucket(ImportHashConfig.NormalizeDllName("KERNEL32.DLL"), ImportHashConfig.DllHashDimensions) != dllBucket)
            throw new InvalidOperationException("DLL 名规范化后没有映射到同一哈希桶。");

        if (ImportHashConfig.Algorithm != "Fnv1a32" ||
            ImportHashConfig.Seed != 2166136261u ||
            ImportHashConfig.DllHashDimensions != 512 ||
            ImportHashConfig.ApiHashDimensions != 4096)
        {
            throw new InvalidOperationException(
                "ImportHashConfig 的算法/种子/维度已经变化；必须同步提升 FeatureSchema.Version 并重新训练模型。");
        }

        Console.WriteLine(
            $"PASS: ImportBehavior 哈希稳定（算法 {ImportHashConfig.Algorithm}，种子 {ImportHashConfig.Seed}，" +
            $"DLL 桶 {dllBucket}/{ImportHashConfig.DllHashDimensions}，API 桶 {apiBucket}/{ImportHashConfig.ApiHashDimensions}）。");
    }

    /// <summary>同一输入在 bytes / stream / file 三个入口下必须得到完全一致的特征，维度固定。</summary>
    private static void AssertImportFeaturesAreDeterministic(string peSamplePath)
    {
        byte[] bytes = File.ReadAllBytes(peSamplePath);

        ImportBehaviorFeatures first = ImportFeatureExtractor.ExtractFromBytes(bytes);
        ImportBehaviorFeatures second = ImportFeatureExtractor.ExtractFromBytes(bytes);
        ImportBehaviorFeatures fromStream = ImportFeatureExtractor.ExtractFromStream(new MemoryStream(bytes));
        ImportBehaviorFeatures fromFile = ImportFeatureExtractor.ExtractFeatures(peSamplePath);

        float[] reference = first.ToFloatArray();
        if (reference.Length != FeatureSchema.ProImportBehaviorCount || reference.Length != 4624)
            throw new InvalidOperationException($"ImportBehavior 特征维度为 {reference.Length}，期望 {FeatureSchema.ProImportBehaviorCount}。");

        if (!reference.SequenceEqual(second.ToFloatArray()) ||
            !reference.SequenceEqual(fromStream.ToFloatArray()) ||
            !reference.SequenceEqual(fromFile.ToFloatArray()))
        {
            throw new InvalidOperationException("同一输入在 bytes / stream / file 三个入口下产生了不同的 ImportBehavior 特征。");
        }

        if (first.Status != ImportParseStatus.Ok || first.DllCount == 0 || first.ApiCount == 0)
            throw new InvalidOperationException(
                $"{Path.GetFileName(peSamplePath)} 的导入表应完整解析，实际状态 {first.Status}、DLL {first.DllCount}、API {first.ApiCount}。");

        int occupiedDllBuckets = 0;
        foreach (float value in first.DllBuckets)
        {
            if (value != 0)
                occupiedDllBuckets++;
        }
        int occupiedApiBuckets = 0;
        foreach (float value in first.ApiBuckets)
        {
            if (value != 0)
                occupiedApiBuckets++;
        }
        if (occupiedDllBuckets == 0 || occupiedApiBuckets == 0)
            throw new InvalidOperationException("导入表已解析但哈希桶全为零，特征提取失效。");

        // 目标缓冲区长度不符必须报错，而不是截断或补零。
        bool rejected = false;
        try
        {
            first.WriteTo(new float[FeatureSchema.ProImportBehaviorCount - 1]);
        }
        catch (ArgumentException)
        {
            rejected = true;
        }
        if (!rejected)
            throw new InvalidOperationException("ImportBehavior 写入长度不符的缓冲区时必须抛异常。");

        AssertImportFeaturesExpressMissingImportTable(bytes);
        AssertImportFeaturesRejectNonPeInput();

        Console.WriteLine(
            $"PASS: ImportBehavior 特征可复现（DLL {first.DllCount} 个 / API {first.ApiCount} 个，" +
            $"占用 DLL 桶 {occupiedDllBuckets}、API 桶 {occupiedApiBuckets}）。");
    }

    /// <summary>没有导入目录的 PE 必须显式表达"特征缺失"，而不是抛异常也不是静默给随机值。</summary>
    private static void AssertImportFeaturesExpressMissingImportTable(byte[] peBytes)
    {
        byte[] stripped = (byte[])peBytes.Clone();
        int peOffset = BitConverter.ToInt32(stripped, 60);
        int optionalHeaderOffset = peOffset + 24;
        ushort magic = BitConverter.ToUInt16(stripped, optionalHeaderOffset);
        bool pe32Plus = magic == 0x20b;
        if (magic != 0x10b && !pe32Plus)
            throw new InvalidOperationException("测试样本不是有效的 PE32/PE32+ 文件。");

        int dataDirectoryOffset = optionalHeaderOffset + (pe32Plus ? 112 : 96);
        // 第 1 个数据目录 = 导入表，第 13 个 = 延迟导入表，各 8 字节（RVA + Size）。
        for (int i = 0; i < 8; i++)
        {
            stripped[dataDirectoryOffset + 8 + i] = 0;
            stripped[dataDirectoryOffset + 8 * 13 + i] = 0;
        }

        ImportBehaviorFeatures missing = ImportFeatureExtractor.ExtractFromBytes(stripped);
        if (missing.Status != ImportParseStatus.NoImportTable)
            throw new InvalidOperationException($"清空导入目录后状态应为 {ImportParseStatus.NoImportTable}，实际 {missing.Status}。");
        if (missing.DllCount != 0 || missing.ApiCount != 0 || missing.DelayDllCount != 0 || missing.DelayApiCount != 0)
            throw new InvalidOperationException("清空导入目录后仍解析出导入项。");

        foreach (float value in missing.DllBuckets)
        {
            if (value != 0)
                throw new InvalidOperationException("导入特征缺失时 DLL 哈希桶必须全零。");
        }
        foreach (float value in missing.ApiBuckets)
        {
            if (value != 0)
                throw new InvalidOperationException("导入特征缺失时 API 哈希桶必须全零。");
        }
    }

    private static void AssertImportFeaturesRejectNonPeInput()
    {
        bool notPeRejected = false;
        try
        {
            ImportFeatureExtractor.ExtractFromBytes(new byte[512]);
        }
        catch (NotSupportedException)
        {
            notPeRejected = true;
        }
        if (!notPeRejected)
            throw new InvalidOperationException("非 PE 输入必须被明确拒绝，不允许返回全零特征。");

        bool tooShortRejected = false;
        try
        {
            ImportFeatureExtractor.ExtractFromBytes(new byte[16]);
        }
        catch (NotSupportedException)
        {
            tooShortRejected = true;
        }
        if (!tooShortRejected)
            throw new InvalidOperationException("过短输入必须被明确拒绝。");
    }

    /// <summary>特征布局必须是"追加式"的：旧偏移不变，新段正好落在尾部。</summary>
    private static void AssertFeatureLayoutIsSelfConsistent()
    {
        if (FeatureSchema.ProStandardOffset != 0)
            throw new InvalidOperationException("Standard 分支偏移必须为 0。");
        if (FeatureSchema.ProFlashOffset != FeatureSchema.StandardFeatureCount)
            throw new InvalidOperationException("Flash 分支偏移错误。");
        if (FeatureSchema.ProRawStatOffset != FeatureSchema.ProFlashOffset + FeatureSchema.FlashFeatureCount)
            throw new InvalidOperationException("RawStat 分支偏移错误。");
        if (FeatureSchema.ProStructuralOffset != FeatureSchema.ProRawStatOffset + FeatureSchema.ProRawStatCount)
            throw new InvalidOperationException("Structural 分支偏移错误。");
        if (FeatureSchema.ProImportBehaviorOffset != FeatureSchema.ProStructuralOffset + FeatureSchema.ProStructuralCount)
            throw new InvalidOperationException("ImportBehavior 分支偏移错误。");
        if (FeatureSchema.ProImportBehaviorOffset + FeatureSchema.ProImportBehaviorCount != FeatureSchema.ProHybridFeatureCount)
            throw new InvalidOperationException("ImportBehavior 没有正好落在混合特征向量尾部。");
        if (FeatureSchema.ProImportBehaviorCount !=
            ImportHashConfig.DllHashDimensions + ImportHashConfig.ApiHashDimensions + FeatureSchema.ProImportStatsCount)
        {
            throw new InvalidOperationException("ImportBehavior 维度不等于 DLL 哈希 + API 哈希 + 统计特征。");
        }
        if (FeatureSchema.ProHybridFeatureCount != 5143 || FeatureSchema.ProImportBehaviorCount != 4624)
            throw new InvalidOperationException("混合特征维度或 ImportBehavior 维度与设计不一致。");
        // 旧模型偏移在新布局下必须原样保留，否则旧产物无法加载。
        if (FeatureSchema.ProLegacyHybridFeatureCount != FeatureSchema.ProImportBehaviorOffset)
            throw new InvalidOperationException("旧版混合特征维度必须等于 ImportBehavior 段的偏移。");
        if (FeatureSchema.ProFusionFeatureCount != 5 || FeatureSchema.ProLegacyFusionFeatureCount != 4)
            throw new InvalidOperationException("融合维度必须为新版 5、旧版 4。");

        Console.WriteLine("PASS: Pro 特征 Schema 偏移自洽，新旧布局兼容（旧 519 维 = 新版前 519 维）。");
    }

    private static void AssertProBranchLayout()
    {
        ProBranch[] expectedOrder =
        {
            ProBranch.Standard, ProBranch.Flash, ProBranch.RawStat, ProBranch.Structural, ProBranch.ImportBehavior
        };

        if (!ProBranches.All.SequenceEqual(expectedOrder))
            throw new InvalidOperationException($"Pro 分支顺序应为 {string.Join("/", expectedOrder)}，实际 {string.Join("/", ProBranches.All)}。");
        if (!ProBranches.Legacy.SequenceEqual(expectedOrder.Take(4)))
            throw new InvalidOperationException("旧版四分支集合与新版前四个分支不一致。");

        int expectedOffset = 0;
        foreach (ProBranch branch in ProBranches.All)
        {
            if (ProBranches.Offset(branch) != expectedOffset)
                throw new InvalidOperationException($"{branch} 分支偏移应为 {expectedOffset}，实际 {ProBranches.Offset(branch)}。");
            expectedOffset += ProBranches.FeatureCount(branch);
        }
        if (expectedOffset != FeatureSchema.ProHybridFeatureCount)
            throw new InvalidOperationException("分支维度之和与混合特征总维度不一致。");

        if (ProBranches.ModelFileSuffix(ProBranch.ImportBehavior) != "-ImportBehavior")
            throw new InvalidOperationException("ImportBehavior 分支模型后缀错误。");
        if (ProBranches.FileNameFor(ProBranch.ImportBehavior, @"C:\models\Xdows-Model-Pro.onnx") != "Xdows-Model-Pro-ImportBehavior.onnx")
            throw new InvalidOperationException("ImportBehavior 分支模型文件名推导错误。");

        string hash = ProBranches.ComputeFeatureLayoutHash();
        if (string.IsNullOrEmpty(hash) || hash != ProBranches.ComputeFeatureLayoutHash())
            throw new InvalidOperationException("特征布局指纹不稳定。");

        AssertModelLayoutContract();

        Console.WriteLine($"PASS: Pro 分支布局为五路（布局指纹 {hash}）。");
    }

    /// <summary>
    /// 模型部署清单只有一个来源：<see cref="ModelLayout.RequiredFileNames"/>。
    /// 训练端复制、宿主自检、原生 staging 都从它取；任何一处漏文件都会让新版模型在推理端加载失败。
    /// 同时锁住"先 Models\ 再扁平"的查找顺序。
    /// </summary>
    private static void AssertModelLayoutContract()
    {
        // 必需清单必须覆盖融合模型、每个分支与 Pro 模型清单。
        string[] required =
        [
            ModelLayout.StandardFileName,
            ModelLayout.FlashFileName,
            ModelLayout.ProFusionFileName,
            ProModelManifest.FileNameFor(ModelLayout.ProFusionFileName)
        ];
        foreach (ProBranch branch in ProBranches.All)
        {
            string branchFileName = ProBranches.FileNameFor(branch, ModelLayout.ProFusionFileName);
            if (!ModelLayout.RequiredFileNames.Contains(branchFileName, StringComparer.Ordinal))
                throw new InvalidOperationException($"必需模型清单缺少 {branch} 分支：{branchFileName}。");
        }
        foreach (string name in required)
        {
            if (!ModelLayout.RequiredFileNames.Contains(name, StringComparer.Ordinal))
                throw new InvalidOperationException($"必需模型清单缺少 {name}。");
        }
        if (ModelLayout.RequiredFileNames.Distinct(StringComparer.Ordinal).Count() != ModelLayout.RequiredFileNames.Count)
            throw new InvalidOperationException("必需模型清单包含重复项。");

        // 训练端复制清单必须与推理端必需清单完全一致。
        string[] copied = TrainingOutputCopier.EnumerateOnnxFileNames().ToArray();
        if (!copied.OrderBy(name => name, StringComparer.Ordinal)
                .SequenceEqual(ModelLayout.RequiredFileNames.OrderBy(name => name, StringComparer.Ordinal), StringComparer.Ordinal))
        {
            throw new InvalidOperationException(
                $"训练端复制清单与推理端必需清单不一致：复制 {copied.Length} 个，必需 {ModelLayout.RequiredFileNames.Count} 个。");
        }

        if (ModelLayout.RelativePath(ModelLayout.ProFusionFileName) !=
            Path.Combine(ModelLayout.ModelDirectoryName, ModelLayout.ProFusionFileName))
        {
            throw new InvalidOperationException("模型相对路径推导错误。");
        }

        AssertModelPathResolutionOrder();
    }

    /// <summary>查找顺序必须是「Models 子目录优先，旧的扁平位置兜底，都没有返回 null」。</summary>
    private static void AssertModelPathResolutionOrder()
    {
        string directory = CreateTempDirectory("xdows-model-path");
        try
        {
            string fileName = ModelLayout.ProFusionFileName;

            if (ModelLayout.ResolveExistingModelPath(directory, fileName) is not null)
                throw new InvalidOperationException("模型不存在时 ResolveExistingModelPath 必须返回 null。");

            string flatPath = Path.Combine(directory, fileName);
            File.WriteAllText(flatPath, "flat");
            if (ModelLayout.ResolveExistingModelPath(directory, fileName) != flatPath)
                throw new InvalidOperationException("旧部署的扁平位置没有被兼容命中。");

            string modelsDirectory = ModelLayout.ResolveModelDirectory(directory);
            Directory.CreateDirectory(modelsDirectory);
            string preferredPath = Path.Combine(modelsDirectory, fileName);
            File.WriteAllText(preferredPath, "models");
            if (ModelLayout.ResolveExistingModelPath(directory, fileName) != preferredPath)
                throw new InvalidOperationException("Models 子目录没有被优先命中。");
        }
        finally
        {
            DeleteTempDirectory(directory);
        }
    }

    private static void AssertProManifestRoundTrip()
    {
        string directory = CreateTempDirectory("xdows-pro-manifest-ok");
        try
        {
            string fusionPath = Path.Combine(directory, "Xdows-Model-Pro.onnx");
            File.WriteAllText(fusionPath, "placeholder");
            WritePlaceholderBranchFiles(directory, fusionPath);

            ProModelManifest manifest = CreateValidProManifest("Xdows-Model-Pro.onnx");
            manifest.Save(fusionPath);

            if (!ProModelManifest.Exists(fusionPath))
                throw new InvalidOperationException("Manifest 没有被写入预期位置。");

            ProModelManifest loaded = ProModelManifest.Load(fusionPath);
            loaded.Validate(fusionPath);

            if (loaded.FeatureSchemaVersion != FeatureSchema.Version ||
                loaded.FeatureCount != FeatureSchema.ProHybridFeatureCount ||
                loaded.FusionInputCount != ProBranches.All.Count ||
                loaded.Branches.Count != ProBranches.All.Count ||
                loaded.HashAlgorithm != ImportHashConfig.Algorithm ||
                loaded.HashSeed != ImportHashConfig.Seed)
            {
                throw new InvalidOperationException("Manifest 往返后字段不再一致。");
            }

            Console.WriteLine("PASS: Pro 模型 Manifest 往返一致，五分支清单可通过校验。");
        }
        finally
        {
            DeleteTempDirectory(directory);
        }
    }

    private static void AssertProManifestRejectsInvalidContent()
    {
        int rejectedCount = 0;

        ExpectManifestRejected("清单版本不匹配", manifest => manifest.SchemaVersion++);
        ExpectManifestRejected("特征版本不匹配", manifest => manifest.FeatureSchemaVersion++);
        ExpectManifestRejected("特征布局指纹不匹配", manifest => manifest.FeatureHash = "deadbeef");
        ExpectManifestRejected("哈希算法不匹配", manifest => manifest.HashAlgorithm = "Crc32");
        ExpectManifestRejected("哈希种子不匹配", manifest => manifest.HashSeed = 0u);
        ExpectManifestRejected("DLL 哈希维度不匹配", manifest => manifest.DllHashDimensions = 256);
        ExpectManifestRejected("API 哈希维度不匹配", manifest => manifest.ApiHashDimensions = 2048);
        ExpectManifestRejected("融合模型文件名不匹配", manifest => manifest.FusionModelFileName = "Xdows-Model-Pro-Other.onnx");
        ExpectManifestRejected("融合输入维度不匹配", manifest => manifest.FusionInputCount = FeatureSchema.ProLegacyFusionFeatureCount);
        ExpectManifestRejected("混合特征维度不匹配", manifest => manifest.FeatureCount = FeatureSchema.ProLegacyHybridFeatureCount);
        ExpectManifestRejected("分支输入维度不匹配", manifest => manifest.Branches[4].InputDimension--);
        ExpectManifestRejected("分支特征数不匹配", manifest => manifest.Branches[4].FeatureCount--);
        ExpectManifestRejected("分支偏移不匹配", manifest => manifest.Branches[4].FeatureOffset++);
        ExpectManifestRejected("分支缺失", manifest => manifest.Branches.RemoveAt(4));
        ExpectManifestRejected("分支重名", manifest => manifest.Branches.Add(manifest.Branches[0]));
        ExpectManifestRejected("未知分支名", manifest => manifest.Branches[4].Name = "NotABranch");
        ExpectManifestRejected("固定阈值越界", manifest => manifest.Threshold = 150);
        ExpectManifestRejected("推荐阈值越界", manifest => manifest.RecommendedThreshold = -1);
        ExpectManifestRejected(
            "分支模型文件缺失",
            _ => { },
            directory => File.Delete(Path.Combine(directory, "Xdows-Model-Pro-ImportBehavior.onnx")));
        rejectedCount += 19;

        // 清单文件本身缺失时必须抛出"缺少清单"，而不是当作旧模型静默通过。
        string missingDirectory = CreateTempDirectory("xdows-pro-manifest-missing");
        try
        {
            string fusionPath = Path.Combine(missingDirectory, "Xdows-Model-Pro.onnx");
            File.WriteAllText(fusionPath, "placeholder");
            bool rejected = false;
            try
            {
                ProModelManifest.Load(fusionPath);
            }
            catch (FileNotFoundException)
            {
                rejected = true;
            }
            if (!rejected)
                throw new InvalidOperationException("Manifest 文件缺失时必须抛出 FileNotFoundException。");
        }
        finally
        {
            DeleteTempDirectory(missingDirectory);
        }

        Console.WriteLine($"PASS: Pro 模型 Manifest 拒绝了全部 {rejectedCount} 种非法内容，并区分「清单缺失」与「内容非法」。");
    }

    private static void AssertProBranchResolution()
    {
        string directory = CreateTempDirectory("xdows-pro-resolve");
        try
        {
            string fusionPath = Path.Combine(directory, "Xdows-Model-Pro.onnx");
            File.WriteAllText(fusionPath, "placeholder");

            // 旧版四分支：没有 Manifest，融合输入 4 维 → 按后缀解析出四条旧分支路径。
            string[] legacyPaths = ProEnsembleSession.ResolveBranchPaths(
                fusionPath,
                FeatureSchema.ProLegacyFusionFeatureCount,
                out ProBranch[] legacyKinds);
            if (!legacyKinds.SequenceEqual(ProBranches.Legacy) || legacyPaths.Length != 4)
                throw new InvalidOperationException("旧版四分支模型的分支解析结果不正确。");
            foreach (ProBranch branch in ProBranches.Legacy)
            {
                string expected = "Xdows-Model-Pro" + ProBranches.ModelFileSuffix(branch) + ".onnx";
                if (Path.GetFileName(legacyPaths[(int)branch]) != expected)
                    throw new InvalidOperationException($"旧版 {branch} 分支路径推导错误：{legacyPaths[(int)branch]}。");
            }

            // 新版五分支但没有 Manifest：必须明确报错，不允许猜。
            bool missingManifestRejected = false;
            try
            {
                ProEnsembleSession.ResolveBranchPaths(fusionPath, FeatureSchema.ProFusionFeatureCount, out _);
            }
            catch (InvalidOperationException)
            {
                missingManifestRejected = true;
            }
            if (!missingManifestRejected)
                throw new InvalidOperationException("五维融合模型缺少 Manifest 时必须抛出明确异常。");

            // 既不是 4 也不是 5 的融合维度：必须明确报错。
            bool unknownDimensionRejected = false;
            try
            {
                ProEnsembleSession.ResolveBranchPaths(fusionPath, 7, out _);
            }
            catch (InvalidOperationException)
            {
                unknownDimensionRejected = true;
            }
            if (!unknownDimensionRejected)
                throw new InvalidOperationException("未知融合维度必须抛出明确异常。");

            // 新版五分支：带 Manifest → 按清单解析。
            WritePlaceholderBranchFiles(directory, fusionPath);
            CreateValidProManifest("Xdows-Model-Pro.onnx").Save(fusionPath);
            string[] branchPaths = ProEnsembleSession.ResolveBranchPaths(
                fusionPath,
                FeatureSchema.ProFusionFeatureCount,
                out ProBranch[] branchKinds);
            if (!branchKinds.SequenceEqual(ProBranches.All) || branchPaths.Length != 5)
                throw new InvalidOperationException("新版五分支模型的分支解析结果不正确。");

            // 清单融合维度与实际融合模型不一致：必须报错。
            bool mismatchRejected = false;
            try
            {
                ProEnsembleSession.ResolveBranchPaths(fusionPath, FeatureSchema.ProLegacyFusionFeatureCount, out _);
            }
            catch (InvalidOperationException)
            {
                mismatchRejected = true;
            }
            if (!mismatchRejected)
                throw new InvalidOperationException("清单融合维度与实际模型不一致时必须抛出明确异常。");

            Console.WriteLine("PASS: 旧四分支模型可继续解析，新五分支必须带 Manifest，维度冲突明确报错。");
        }
        finally
        {
            DeleteTempDirectory(directory);
        }
    }

    private static void AssertProDatasetSplit()
    {
        var features = new float[FeatureSchema.ProHybridFeatureCount];
        var config = new TrainingConfig();

        // 内容哈希去重必须大小写不敏感，且不丢弃不同内容。
        var duplicated = new List<ProStackingSample>
        {
            new(features, true) { ContentHash = "AA" },
            new(features, true) { ContentHash = "aa" },
            new(features, true) { ContentHash = "BB" },
            new(features, false) { ContentHash = "CC" },
            new(features, false) { ContentHash = "DD" },
            new(features, false) { ContentHash = "EE" }
        };
        ProDatasetSplitResult deduplicated = ProDatasetSplitter.Split(duplicated, config);
        if (deduplicated.Diagnostics.DuplicateSamplesRemoved != 1 || deduplicated.Samples.Count != 5)
            throw new InvalidOperationException(
                $"内容哈希去重结果错误：剔除 {deduplicated.Diagnostics.DuplicateSamplesRemoved} 个，剩余 {deduplicated.Samples.Count} 个。");
        if (deduplicated.Samples.Count(sample => sample.Label) != 2 || deduplicated.Samples.Count(sample => !sample.Label) != 3)
            throw new InvalidOperationException("内容哈希去重改变了类别分布以外的样本。");

        // 分组隔离：同 GroupKey 的样本必须整体落在同一侧。
        var grouped = new List<ProStackingSample>();
        for (int group = 0; group < 10; group++)
        {
            bool label = group % 2 == 0;
            for (int member = 0; member < 4; member++)
            {
                grouped.Add(new ProStackingSample(features, label)
                {
                    GroupKey = $"family-{group}",
                    ContentHash = $"H{group}-{member}"
                });
            }
        }

        ProDatasetSplitResult groupSplit = ProDatasetSplitter.Split(grouped, config);
        if (!groupSplit.Diagnostics.GroupIsolationApplied || groupSplit.Diagnostics.DistinctGroupCount != 10)
            throw new InvalidOperationException("分组隔离没有被识别为启用状态。");
        if (groupSplit.TrainIndices.Count + groupSplit.TestIndices.Count != groupSplit.Samples.Count)
            throw new InvalidOperationException("分组切分后训练集与测试集样本数之和不等于去重后的样本总数。");
        if (groupSplit.TrainIndices.Distinct().Count() != groupSplit.TrainIndices.Count ||
            groupSplit.TestIndices.Distinct().Count() != groupSplit.TestIndices.Count ||
            groupSplit.TrainIndices.Intersect(groupSplit.TestIndices).Any())
        {
            throw new InvalidOperationException("分组切分产生了重复或交叉的样本索引。");
        }

        var groupSides = new Dictionary<string, HashSet<bool>>(StringComparer.Ordinal);
        var testSet = groupSplit.TestIndices.ToHashSet();
        for (int index = 0; index < groupSplit.Samples.Count; index++)
        {
            string key = groupSplit.Samples[index].GroupKey!;
            if (!groupSides.TryGetValue(key, out HashSet<bool>? sides))
            {
                sides = new HashSet<bool>();
                groupSides.Add(key, sides);
            }
            sides.Add(testSet.Contains(index));
        }
        foreach ((string key, HashSet<bool> sides) in groupSides)
        {
            if (sides.Count != 1)
                throw new InvalidOperationException($"分组 {key} 的样本被切到了训练集与测试集两侧。");
        }
        if (!groupSplit.Diagnostics.GroupIsolationApplied || groupSplit.Diagnostics.TestGroupCount == 0)
            throw new InvalidOperationException("分组切分没有把任何分组放进测试集。");

        foreach ((string name, IReadOnlyList<int> indices) in new[]
                 {
                     ("训练集", groupSplit.TrainIndices), ("测试集", groupSplit.TestIndices)
                 })
        {
            if (!indices.Any(i => groupSplit.Samples[i].Label) || !indices.Any(i => !groupSplit.Samples[i].Label))
                throw new InvalidOperationException($"{name}缺少某一类别的样本。");
        }

        // 时间切分：测试集取时间键最大的一段。
        var timed = new List<ProStackingSample>();
        for (int index = 0; index < 20; index++)
            timed.Add(new ProStackingSample(features, index % 2 == 0) { TimeKey = 1000 + index });

        var timeConfig = new TrainingConfig { ProSplitMode = ProDatasetSplitMode.TimeOrdered, ProTestFraction = 0.25 };
        ProDatasetSplitResult timeSplit = ProDatasetSplitter.Split(timed, timeConfig);
        if (timeSplit.TestIndices.Count != 5)
            throw new InvalidOperationException($"时间切分测试集样本数为 {timeSplit.TestIndices.Count}，期望 5。");
        foreach (int index in timeSplit.TestIndices)
        {
            if (timed[index].TimeKey < 1015)
                throw new InvalidOperationException("时间切分把较早的样本放进了测试集。");
        }
        foreach (int index in timeSplit.TrainIndices)
        {
            if (timed[index].TimeKey >= 1015)
                throw new InvalidOperationException("时间切分把较晚的样本放进了训练集。");
        }

        // 时间切分但样本缺少时间键：必须明确报错。
        var missingTimeKey = new List<ProStackingSample>
        {
            new(features, true), new(features, false), new(features, true), new(features, false)
        };
        bool timeKeyRejected = false;
        try
        {
            ProDatasetSplitter.Split(missingTimeKey, timeConfig);
        }
        catch (InvalidOperationException)
        {
            timeKeyRejected = true;
        }
        if (!timeKeyRejected)
            throw new InvalidOperationException("时间切分在样本缺少时间键时必须明确报错。");

        // 空数据必须给出可诊断错误。
        bool emptyRejected = false;
        try
        {
            ProDatasetSplitter.Split(Array.Empty<ProStackingSample>(), config);
        }
        catch (InvalidOperationException)
        {
            emptyRejected = true;
        }
        if (!emptyRejected)
            throw new InvalidOperationException("空训练数据必须抛出可诊断异常。");

        Console.WriteLine("PASS: Pro 数据集切分完成内容哈希去重、分组隔离与可选时间切分，并区分缺少时间键/空数据。");
    }

    /// <summary>
    /// 端到端：训练一个小型五分支 Pro 模型 → 导出五分支 + 融合 ONNX + Manifest →
    /// 用推理端会话加载并预测。覆盖"训练端与推理端特征一致"之外的模型产物契约。
    /// </summary>
    private static void AssertProFiveBranchEndToEnd()
    {
        string directory = CreateTempDirectory("xdows-pro-e2e");
        try
        {
            var random = new Random(20260925);
            const int sampleCount = 200;
            var samples = new List<ProStackingSample>(sampleCount);
            for (int index = 0; index < sampleCount; index++)
            {
                bool label = (index & 1) == 0;
                var features = new float[FeatureSchema.ProHybridFeatureCount];
                for (int featureIndex = 0; featureIndex < features.Length; featureIndex++)
                    features[featureIndex] = (float)random.NextDouble() + (label ? 0.1f : 0f);
                samples.Add(new ProStackingSample(features, label));
            }

            var config = new TrainingConfig
            {
                ProNumberOfIterations = 4,
                ProNumberOfLeaves = 5,
                ProMinimumExampleCountPerLeaf = 5,
                ProMaxParallelBranches = 2
            };
            var mlContext = new MLContext(seed: config.RandomSeed);
            var trainer = new ProStackingTrainer(mlContext, config, new ProGbdtLearner());
            ProStackingTrainingResult result = trainer.Train(samples);

            if (result.BranchModels.Count != ProBranches.All.Count)
                throw new InvalidOperationException($"训练产生的分支数为 {result.BranchModels.Count}，期望 {ProBranches.All.Count}。");
            if (result.Evaluation.FeatureCount != FeatureSchema.ProFusionFeatureCount)
                throw new InvalidOperationException($"评估记录的融合维度为 {result.Evaluation.FeatureCount}，期望 {FeatureSchema.ProFusionFeatureCount}。");

            string modelPath = Path.Combine(directory, "Xdows-Model-Pro.zip");
            string fusionOnnxPath = Path.Combine(directory, "Xdows-Model-Pro.onnx");
            result.SaveArtifacts(mlContext, modelPath, fusionOnnxPath, config.ProThreshold);

            if (!File.Exists(modelPath) || !File.Exists(fusionOnnxPath))
                throw new InvalidOperationException("Pro 融合模型（zip / onnx）没有生成。");
            foreach (ProBranch branch in ProBranches.All)
            {
                string branchPath = Path.Combine(directory, ProBranches.FileNameFor(branch, fusionOnnxPath));
                if (!File.Exists(branchPath))
                    throw new InvalidOperationException($"{branch} 分支 ONNX 没有导出：{branchPath}。");
            }
            if (!ProModelManifest.Exists(fusionOnnxPath))
                throw new InvalidOperationException("Pro 模型 Manifest 没有随模型导出。");

            ProModelManifest manifest = ProModelManifest.Load(fusionOnnxPath);
            manifest.Validate(fusionOnnxPath);
            if (manifest.Branches.Count != ProBranches.All.Count ||
                manifest.FusionInputCount != FeatureSchema.ProFusionFeatureCount)
            {
                throw new InvalidOperationException("导出的 Manifest 分支数量或融合维度不正确。");
            }
            if (Math.Abs(manifest.Threshold - config.ProThreshold) > 0.000001)
                throw new InvalidOperationException("Manifest 没有记录配置的固定判毒阈值。");

            using var fusionSession = new InferenceSession(fusionOnnxPath, ModelInvoker.CreateSessionOptions());
            int fusionDimension = ProEnsembleSession.ReadFeatureDimension(fusionSession);
            if (fusionDimension != FeatureSchema.ProFusionFeatureCount)
                throw new InvalidOperationException(
                    $"融合 ONNX 输入维度为 {fusionDimension}，期望 {FeatureSchema.ProFusionFeatureCount}。");
            if (fusionDimension != manifest.FusionInputCount)
                throw new InvalidOperationException("融合 ONNX 输入维度与 Manifest 声明不一致。");

            // 每个分支 ONNX 的输入维度必须能被推理端读到，否则加载期校验形同虚设。
            foreach (ProBranch branch in ProBranches.All)
            {
                string branchPath = Path.Combine(directory, ProBranches.FileNameFor(branch, fusionOnnxPath));
                using var branchSession = new InferenceSession(branchPath, ModelInvoker.CreateSessionOptions());
                int branchDimension = ProEnsembleSession.ReadFeatureDimension(branchSession);
                if (branchDimension != ProBranches.FeatureCount(branch))
                    throw new InvalidOperationException(
                        $"{branch} 分支 ONNX 输入维度为 {branchDimension}，期望 {ProBranches.FeatureCount(branch)}。");
            }

            using var ensemble = new ProEnsembleSession(fusionOnnxPath, fusionDimension);
            if (ensemble.FusionInputCount != ProBranches.All.Count)
                throw new InvalidOperationException($"分支会话的融合输入维度为 {ensemble.FusionInputCount}，期望 {ProBranches.All.Count}。");

            float probability = ensemble.Predict(fusionSession, samples[0].Features);
            if (!float.IsFinite(probability) || probability < 0 || probability > 100)
                throw new InvalidOperationException($"五分支 Pro 推理返回了非法概率 {probability}。");

            AssertLegacyFourBranchModelStillLoads(mlContext, directory, fusionOnnxPath, samples[0]);

            Console.WriteLine(
                $"PASS: 五分支 Pro 模型可训练、导出、按 Manifest 加载并推理（测试样本概率 {probability:F2}%）。");
        }
        finally
        {
            DeleteTempDirectory(directory);
        }
    }

    /// <summary>
    /// 旧版四分支兼容：融合模型只有 4 路输入、不带 Manifest，分支文件沿用旧后缀。
    /// 推理端必须继续按后缀解析旧分支并给出概率，而不是要求重新训练。
    /// </summary>
    private static void AssertLegacyFourBranchModelStillLoads(
        MLContext mlContext,
        string directory,
        string fusionOnnxPath,
        ProStackingSample sample)
    {
        string legacyFusionPath = Path.Combine(directory, "Xdows-Model-Pro-Legacy.onnx");
        var legacyRows = Enumerable.Range(0, 64)
            .Select(index => new LegacyProFusionTrainingData
            {
                Features = new float[FeatureSchema.ProLegacyFusionFeatureCount],
                Label = (index & 1) == 0
            })
            .ToList();
        IDataView legacyData = mlContext.Data.LoadFromEnumerable(legacyRows);
        ITransformer legacyFusion = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
            labelColumnName: nameof(LegacyProFusionTrainingData.Label),
            featureColumnName: nameof(LegacyProFusionTrainingData.Features)).Fit(legacyData);
        using (FileStream stream = File.Create(legacyFusionPath))
            mlContext.Model.ConvertToOnnx(legacyFusion, legacyData, stream);

        foreach (ProBranch branch in ProBranches.Legacy)
        {
            File.Copy(
                Path.Combine(directory, ProBranches.FileNameFor(branch, fusionOnnxPath)),
                Path.Combine(directory, ProBranches.FileNameFor(branch, legacyFusionPath)));
        }

        if (ProModelManifest.Exists(legacyFusionPath))
            throw new InvalidOperationException("旧版兼容测试的前提是模型旁没有 Manifest。");

        using var legacySession = new InferenceSession(legacyFusionPath, ModelInvoker.CreateSessionOptions());
        int legacyDimension = ProEnsembleSession.ReadFeatureDimension(legacySession);
        if (legacyDimension != FeatureSchema.ProLegacyFusionFeatureCount)
            throw new InvalidOperationException($"旧版融合 ONNX 输入维度为 {legacyDimension}，期望 {FeatureSchema.ProLegacyFusionFeatureCount}。");

        using var legacyEnsemble = new ProEnsembleSession(legacyFusionPath, legacyDimension);
        if (legacyEnsemble.FusionInputCount != FeatureSchema.ProLegacyFusionFeatureCount)
            throw new InvalidOperationException($"旧版分支会话的融合输入维度为 {legacyEnsemble.FusionInputCount}，期望 {FeatureSchema.ProLegacyFusionFeatureCount}。");

        float legacyProbability = legacyEnsemble.Predict(legacySession, sample.Features);
        if (!float.IsFinite(legacyProbability) || legacyProbability < 0 || legacyProbability > 100)
            throw new InvalidOperationException($"旧版四分支 Pro 推理返回了非法概率 {legacyProbability}。");

        Console.WriteLine($"PASS: 无 Manifest 的旧四分支 Pro 模型仍可加载并推理（测试样本概率 {legacyProbability:F2}%）。");
    }

    private static ProModelManifest CreateValidProManifest(string fusionFileName)
    {
        return new ProModelManifest
        {
            SchemaVersion = ProModelManifest.CurrentSchemaVersion,
            ModelVersion = "architecture-test",
            FeatureSchemaVersion = FeatureSchema.Version,
            FeatureCount = FeatureSchema.ProHybridFeatureCount,
            FeatureHash = ProBranches.ComputeFeatureLayoutHash(),
            FusionModelFileName = fusionFileName,
            FusionInputCount = ProBranches.All.Count,
            Branches = ProBranches.All
                .Select(branch => new ProBranchManifestEntry
                {
                    Name = branch.ToString(),
                    FileName = ProBranches.FileNameFor(branch, fusionFileName),
                    InputDimension = ProBranches.FeatureCount(branch),
                    FeatureOffset = ProBranches.Offset(branch),
                    FeatureCount = ProBranches.FeatureCount(branch)
                })
                .ToList(),
            HashAlgorithm = ImportHashConfig.Algorithm,
            HashSeed = ImportHashConfig.Seed,
            DllHashDimensions = ImportHashConfig.DllHashDimensions,
            ApiHashDimensions = ImportHashConfig.ApiHashDimensions,
            Threshold = 94.0,
            RecommendedThreshold = 56.45,
            GeneratedAt = "2026-09-25 00:00:00"
        };
    }

    private static void WritePlaceholderBranchFiles(string directory, string fusionPath)
    {
        foreach (ProBranch branch in ProBranches.All)
            File.WriteAllText(Path.Combine(directory, ProBranches.FileNameFor(branch, fusionPath)), "placeholder");
    }

    private static void ExpectManifestRejected(
        string scenario,
        Action<ProModelManifest> mutate,
        Action<string>? afterWrite = null)
    {
        string directory = CreateTempDirectory("xdows-pro-manifest-bad");
        try
        {
            string fusionPath = Path.Combine(directory, "Xdows-Model-Pro.onnx");
            File.WriteAllText(fusionPath, "placeholder");
            WritePlaceholderBranchFiles(directory, fusionPath);

            ProModelManifest manifest = CreateValidProManifest("Xdows-Model-Pro.onnx");
            mutate(manifest);
            manifest.Save(fusionPath);
            afterWrite?.Invoke(directory);

            bool rejected = false;
            string reason = string.Empty;
            try
            {
                ProModelManifest.Load(fusionPath).Validate(fusionPath);
            }
            catch (Exception ex) when (ex is InvalidOperationException or FileNotFoundException)
            {
                rejected = true;
                reason = ex.Message;
            }

            if (!rejected)
                throw new InvalidOperationException($"Manifest 校验没有拒绝非法清单：{scenario}。");

            Console.WriteLine($"  INFO 清单已按预期拒绝 [{scenario}]: {reason}");
        }
        finally
        {
            DeleteTempDirectory(directory);
        }
    }

    private static string CreateTempDirectory(string prefix)
    {
        string directory = Path.Combine(Path.GetTempPath(), $"{prefix}-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static void DeleteTempDirectory(string directory)
    {
        try
        {
            Directory.Delete(directory, recursive: true);
        }
        catch (IOException ex)
        {
            // ONNX Runtime 可能仍持有映射文件；清理失败不影响断言结论，但要报出来。
            Console.WriteLine($"  INFO 临时目录未能删除 {directory}: {ex.Message}");
        }
    }
}

/// <summary>旧版四分支 Pro 模型的融合输入行（4 维），仅用于兼容性测试。</summary>
internal sealed class LegacyProFusionTrainingData
{
    [VectorType(FeatureSchema.ProLegacyFusionFeatureCount)]
    public float[] Features { get; set; } = Array.Empty<float>();

    public bool Label { get; set; }
}
