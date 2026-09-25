using Xdows_Model_Config;

namespace Xdows_Model_Maker;

/// <summary>
/// Pro 数据集切分诊断信息。切分过程中的每一次"被丢弃/未生效"都必须在这里留下计数，
/// 由 <see cref="ProDatasetReporter"/> 打出来，不允许静默处理。
/// </summary>
internal sealed class ProDatasetDiagnostics
{
    public required ProDatasetSplitMode Mode { get; init; }
    public required int InputSampleCount { get; init; }

    /// <summary>因内容哈希相同被剔除的样本数。</summary>
    public required int DuplicateSamplesRemoved { get; init; }

    /// <summary>没有内容哈希、无法参与去重的样本数（去重关闭时等于输入样本数）。</summary>
    public required int SamplesWithoutContentHash { get; init; }

    /// <summary>没有 GroupKey、只能作为独立分组处理的样本数。</summary>
    public required int SamplesWithoutGroupKey { get; init; }

    public required int DistinctGroupCount { get; init; }

    /// <summary>同一分组内出现两种标签的分组数。整组按多数标签归边，隔离性仍然成立。</summary>
    public required int MixedLabelGroupCount { get; init; }

    /// <summary>落在测试集一侧的分组数。</summary>
    public required int TestGroupCount { get; init; }

    /// <summary>是否真正按 GroupKey 做了同源隔离（样本全部没有 GroupKey 时为 false）。</summary>
    public required bool GroupIsolationApplied { get; init; }
}

/// <summary>Pro 数据集切分结果。索引都指向 <see cref="Samples"/>（已去重）。</summary>
internal sealed class ProDatasetSplitResult
{
    public required IReadOnlyList<ProStackingSample> Samples { get; init; }
    public required IReadOnlyList<int> TrainIndices { get; init; }
    public required IReadOnlyList<int> TestIndices { get; init; }
    public required ProDatasetDiagnostics Diagnostics { get; init; }
}

/// <summary>
/// Pro 训练数据的外层切分：先按内容哈希去重，再按分组隔离 + 类别分层（或时间顺序）划分训练集与测试集。
/// 分组键与时间键都是 <see cref="ProStackingSample"/> 上的可选字段；数据源没有这些信息时不会被伪造，
/// 切分报告会明确说明哪一项没有生效。
/// </summary>
internal static class ProDatasetSplitter
{
    public static ProDatasetSplitResult Split(IReadOnlyList<ProStackingSample> samples, TrainingConfig config)
    {
        ArgumentNullException.ThrowIfNull(samples);
        ArgumentNullException.ThrowIfNull(config);

        if (samples.Count == 0)
            throw new InvalidOperationException("Pro 训练数据为空，无法切分数据集。");

        double testFraction = config.ResolveProTestFraction();

        (List<ProStackingSample> deduplicated, int removedDuplicates, int missingHash) =
            DeduplicateByContentHash(samples, config.ProDeduplicateByContentHash);

        if (deduplicated.Count < 4)
            throw new InvalidOperationException(
                $"内容哈希去重后只剩 {deduplicated.Count} 个样本（原始 {samples.Count} 个），不足以切分训练集与测试集。");

        ProDatasetSplitMode mode = config.ProSplitMode;
        List<int> trainIndices;
        List<int> testIndices;
        int samplesWithoutGroupKey = 0;
        int distinctGroupCount = deduplicated.Count;
        int mixedLabelGroupCount = 0;
        int testGroupCount = 0;
        bool groupIsolationApplied = false;

        if (mode == ProDatasetSplitMode.TimeOrdered)
        {
            (trainIndices, testIndices) = SplitByTimeKey(deduplicated, testFraction);
        }
        else
        {
            List<SampleGroup> groups = BuildGroups(deduplicated);
            bool hasGroupKey = deduplicated.Any(sample => !string.IsNullOrEmpty(sample.GroupKey));
            groupIsolationApplied = hasGroupKey && config.ProEnforceGroupIsolation;

            samplesWithoutGroupKey = deduplicated.Count(sample => string.IsNullOrEmpty(sample.GroupKey));
            distinctGroupCount = groups.Count;
            mixedLabelGroupCount = groups.Count(group => group.IsMixedLabel);

            (trainIndices, testIndices, testGroupCount) = SplitByGroups(deduplicated, groups, testFraction);
        }

        var diagnostics = new ProDatasetDiagnostics
        {
            Mode = mode,
            InputSampleCount = samples.Count,
            DuplicateSamplesRemoved = removedDuplicates,
            SamplesWithoutContentHash = missingHash,
            SamplesWithoutGroupKey = samplesWithoutGroupKey,
            DistinctGroupCount = distinctGroupCount,
            MixedLabelGroupCount = mixedLabelGroupCount,
            TestGroupCount = testGroupCount,
            GroupIsolationApplied = groupIsolationApplied
        };

        return new ProDatasetSplitResult
        {
            Samples = deduplicated,
            TrainIndices = trainIndices,
            TestIndices = testIndices,
            Diagnostics = diagnostics
        };
    }

    /// <summary>按内容哈希去除完全重复的样本，保留首次出现的那个。</summary>
    private static (List<ProStackingSample> Samples, int Removed, int MissingHash) DeduplicateByContentHash(
        IReadOnlyList<ProStackingSample> samples,
        bool enabled)
    {
        if (!enabled)
            return (samples.ToList(), 0, samples.Count);

        var result = new List<ProStackingSample>(samples.Count);
        var seenHashes = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        int removed = 0;
        int missingHash = 0;

        foreach (ProStackingSample sample in samples)
        {
            if (string.IsNullOrEmpty(sample.ContentHash))
            {
                missingHash++;
                result.Add(sample);
                continue;
            }

            if (seenHashes.Add(sample.ContentHash))
                result.Add(sample);
            else
                removed++;
        }

        return (result, removed, missingHash);
    }

    /// <summary>
    /// 把样本按 GroupKey 归纳成组。没有 GroupKey 的样本各自成组，
    /// 这样分组切分在"完全没有家族标签"时退化为逐样本分层，语义不变。
    /// </summary>
    private static List<SampleGroup> BuildGroups(IReadOnlyList<ProStackingSample> samples)
    {
        var byKey = new Dictionary<string, SampleGroup>(StringComparer.Ordinal);
        var groups = new List<SampleGroup>();

        for (int index = 0; index < samples.Count; index++)
        {
            string key = samples[index].GroupKey ?? string.Empty;
            if (key.Length == 0)
            {
                var singleton = new SampleGroup(string.Empty);
                singleton.Add(index, samples[index].Label);
                groups.Add(singleton);
                continue;
            }

            if (!byKey.TryGetValue(key, out SampleGroup? group))
            {
                group = new SampleGroup(key);
                byKey.Add(key, group);
                groups.Add(group);
            }

            group.Add(index, samples[index].Label);
        }

        return groups;
    }

    /// <summary>
    /// 按类别分层、以"组"为最小单位分配训练/测试集：一个组整体落在同一侧。
    /// 两侧都必须拿到该类别至少一个组，否则明确报错而不是产出不可用的数据集。
    /// </summary>
    private static (List<int> Train, List<int> Test, int TestGroupCount) SplitByGroups(
        IReadOnlyList<ProStackingSample> samples,
        IReadOnlyList<SampleGroup> groups,
        double testFraction)
    {
        var train = new List<int>();
        var test = new List<int>();
        int testGroupCount = 0;

        foreach (bool label in new[] { false, true })
        {
            string labelName = label ? "黑" : "白";
            int labelSampleCount = samples.Count(sample => sample.Label == label);
            if (labelSampleCount < 2)
                throw new InvalidOperationException($"Pro 训练数据的{labelName}样本只有 {labelSampleCount} 个，每类至少需要 2 个样本。");

            List<SampleGroup> ordered = groups
                .Where(group => group.DominantLabel == label)
                .OrderBy(group => group.StableOrderKey)
                .ToList();
            if (ordered.Count == 0)
                throw new InvalidOperationException($"Pro 训练数据的{labelName}样本无法切分：没有任何分组的主标签属于该类别。");

            Shuffle(ordered, new Random(unchecked(43846 + (label ? 1 : 0) + ordered.Count)));

            int target = Math.Clamp((int)Math.Round(labelSampleCount * testFraction), 1, labelSampleCount - 1);
            int assigned = 0;
            int testGroupsForLabel = 0;

            for (int i = 0; i < ordered.Count; i++)
            {
                // 至少给训练集留下一个分组，否则该类别在训练侧为空。
                if (assigned >= target || ordered.Count - i == 1)
                    break;

                test.AddRange(ordered[i].Indices);
                assigned += ordered[i].Indices.Count;
                testGroupsForLabel++;
            }

            if (assigned == 0)
                throw new InvalidOperationException(
                    $"Pro 训练数据的{labelName}样本只有一个分组，无法在保持分组隔离的前提下同时填充训练集与测试集。");

            for (int i = testGroupsForLabel; i < ordered.Count; i++)
                train.AddRange(ordered[i].Indices);

            testGroupCount += testGroupsForLabel;
        }

        return (train, test, testGroupCount);
    }

    /// <summary>按 TimeKey 升序切分：较早的样本进训练集，最晚的一段进测试集。</summary>
    private static (List<int> Train, List<int> Test) SplitByTimeKey(
        IReadOnlyList<ProStackingSample> samples,
        double testFraction)
    {
        for (int index = 0; index < samples.Count; index++)
        {
            if (samples[index].TimeKey is null)
                throw new InvalidOperationException(
                    $"已配置时间切分（{nameof(TrainingConfig.ProSplitMode)} = {ProDatasetSplitMode.TimeOrdered}），" +
                    $"但第 {index} 个样本没有 TimeKey。请为全部样本提供时间键，或改用分层随机切分。");
        }

        int[] ordered = Enumerable.Range(0, samples.Count)
            .OrderBy(index => samples[index].TimeKey!.Value)
            .ThenBy(index => index)
            .ToArray();

        int testCount = Math.Clamp((int)Math.Round(samples.Count * testFraction), 1, samples.Count - 1);
        var test = ordered.Skip(samples.Count - testCount).ToList();
        var train = ordered.Take(samples.Count - testCount).ToList();

        // 时间切分不做分层，两侧都必须同时含黑白样本，否则指标无从解读。
        foreach ((string name, List<int> indices) in new[] { ("训练集", train), ("测试集", test) })
        {
            if (!indices.Any(index => samples[index].Label) || !indices.Any(index => !samples[index].Label))
                throw new InvalidOperationException(
                    $"时间切分后{name}只有单一标签，无法训练或评估。请调整测试集比例或改用分层随机切分。");
        }

        return (train, test);
    }

    private static void Shuffle<T>(IList<T> items, Random random)
    {
        for (int i = items.Count - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (items[i], items[j]) = (items[j], items[i]);
        }
    }

    /// <summary>一个分组（同一 GroupKey 的样本集合）及其多数标签。</summary>
    private sealed class SampleGroup
    {
        private int _blackCount;

        public SampleGroup(string key)
        {
            Key = key;
            StableOrderKey = StableOrder(key);
        }

        public string Key { get; }
        public List<int> Indices { get; } = new();
        public bool DominantLabel { get; private set; }
        public bool IsMixedLabel { get; private set; }

        /// <summary>与样本枚举顺序无关的稳定排序键，保证同一次训练的结果可复现。</summary>
        public int StableOrderKey { get; }

        public void Add(int index, bool label)
        {
            Indices.Add(index);
            if (label)
                _blackCount++;

            DominantLabel = _blackCount * 2 >= Indices.Count;
            IsMixedLabel = _blackCount != 0 && _blackCount != Indices.Count;
        }

        private static int StableOrder(string key)
        {
            unchecked
            {
                int hash = 17;
                foreach (char c in key)
                    hash = hash * 31 + c;
                return hash;
            }
        }
    }
}

/// <summary>把切分结果与类别分布打成中文报告，训练日志里能直接看到每次实验的数据构成。</summary>
internal static class ProDatasetReporter
{
    public static void Print(ProDatasetSplitResult split, int foldCount)
    {
        ProDatasetDiagnostics diagnostics = split.Diagnostics;
        IReadOnlyList<ProStackingSample> samples = split.Samples;

        Console.WriteLine();
        Console.WriteLine("  === Pro 训练数据集切分 ===");
        Console.WriteLine($"  输入样本: {diagnostics.InputSampleCount}");

        if (diagnostics.DuplicateSamplesRemoved > 0)
            Console.WriteLine($"  内容哈希去重: 剔除 {diagnostics.DuplicateSamplesRemoved} 个重复样本");
        else
            Console.WriteLine("  内容哈希去重: 未发现重复样本");

        if (diagnostics.SamplesWithoutContentHash > 0)
            Console.WriteLine(
                $"  提示: {diagnostics.SamplesWithoutContentHash} 个样本没有内容哈希，未参与去重" +
                "（Standard/Flash 模式加载的数据不计算哈希，改用 Pro 模式加载可补齐）。");

        if (diagnostics.GroupIsolationApplied)
            Console.WriteLine(
                $"  分组隔离: 已启用，{diagnostics.DistinctGroupCount} 个分组，测试集占用 {diagnostics.TestGroupCount} 个分组" +
                $"（混合标签分组 {diagnostics.MixedLabelGroupCount} 个，按多数标签归边）");
        else if (diagnostics.SamplesWithoutGroupKey > 0)
            Console.WriteLine(
                $"  分组隔离: 未启用 — 全部 {diagnostics.SamplesWithoutGroupKey} 个样本都没有 GroupKey（恶意家族标签），" +
                "同源样本隔离无法生效。");

        Console.WriteLine($"  切分方式: {(diagnostics.Mode == ProDatasetSplitMode.TimeOrdered ? "时间切分" : "分层随机切分")}");

        PrintDistribution("  训练集", split.TrainIndices, samples);
        Console.WriteLine($"  验证集: OOF {foldCount} 折（复用训练集样本，每折留出一部分做验证）");
        PrintDistribution("  测试集", split.TestIndices, samples);
        Console.WriteLine("  =========================");
        Console.WriteLine();
    }

    private static void PrintDistribution(string label, IReadOnlyList<int> indices, IReadOnlyList<ProStackingSample> samples)
    {
        int black = indices.Count(index => samples[index].Label);
        int white = indices.Count - black;
        Console.WriteLine($"{label}: 共 {indices.Count}（黑 {black} / 白 {white}）");
    }
}
