using System.Diagnostics;
using System.Globalization;
using Xdows_Model_Config;
using Xdows_Model_Invoker;

namespace Xdows_Model_Evaluator;

internal static class Program
{
    private static int Main(string[] args)
    {
        var options = EvaluationOptions.Parse(args);
        if (options.ShowHelp)
        {
            PrintUsage();
            return 0;
        }

        var config = new TrainingConfig();
        string blackFolder = options.BlackFolder ?? config.BlackFolder;
        string whiteFolder = options.WhiteFolder ?? config.WhiteFolder;

        if (!Directory.Exists(blackFolder))
        {
            Console.WriteLine($"黑样本目录不存在: {blackFolder}");
            return 1;
        }
        if (!Directory.Exists(whiteFolder))
        {
            Console.WriteLine($"白样本目录不存在: {whiteFolder}");
            return 1;
        }

        var blackFiles = PickFiles(blackFolder, options.LimitPerClass, options.Seed);
        var whiteFiles = PickFiles(whiteFolder, options.LimitPerClass, options.Seed + 1);
        var samples = blackFiles.Select(path => new EvaluationSample(path, true))
            .Concat(whiteFiles.Select(path => new EvaluationSample(path, false)))
            .OrderBy(_ => StableRandom.Shared.Next())
            .ToList();

        Console.WriteLine($"黑样本: {blackFiles.Count}");
        Console.WriteLine($"白样本: {whiteFiles.Count}");
        Console.WriteLine($"总样本: {samples.Count}");
        Console.WriteLine();

        ModelInvoker.ConfigureThresholds(config);

        using var csv = !string.IsNullOrWhiteSpace(options.CsvPath)
            ? new StreamWriter(options.CsvPath)
            : null;
        csv?.WriteLine("mode,path,label,prediction,probability,error");

        foreach (var mode in options.Modes)
        {
            try
            {
                var result = EvaluateMode(mode, samples, options, csv);
                PrintResult(result);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"=== {mode} ===");
                Console.WriteLine($"评估失败: {ex.Message}");
            }
            Console.WriteLine();
        }

        ModelInvoker.UnloadModel();
        return 0;
    }

    private static EvaluationResult EvaluateMode(string mode, List<EvaluationSample> samples, EvaluationOptions options, StreamWriter? csv)
    {
        ModelInvoker.UnloadModel();
        InitializeMode(mode, options);

        long truePositive = 0;
        long falseNegative = 0;
        long falsePositive = 0;
        long trueNegative = 0;
        long failed = 0;

        var stopwatch = Stopwatch.StartNew();

        foreach (var sample in samples)
        {
            try
            {
                var (isVirus, probability) = ModelInvoker.ScanFile(sample.Path);
                if (sample.IsMalicious && isVirus)
                    truePositive++;
                else if (sample.IsMalicious)
                    falseNegative++;
                else if (isVirus)
                    falsePositive++;
                else
                    trueNegative++;

                if ((sample.IsMalicious && !isVirus) || (!sample.IsMalicious && isVirus))
                    WriteCsv(csv, mode, sample.Path, sample.IsMalicious, isVirus, probability, "");
            }
            catch (Exception ex)
            {
                failed++;
                WriteCsv(csv, mode, sample.Path, sample.IsMalicious, null, null, ex.Message);
            }
        }

        stopwatch.Stop();

        return EvaluationResult.Create(
            mode,
            truePositive,
            falseNegative,
            falsePositive,
            trueNegative,
            failed,
            stopwatch.Elapsed);
    }

    private static void InitializeMode(string mode, EvaluationOptions options)
    {
        switch (mode)
        {
            case "standard":
                ModelInvoker.Initialize(options.StandardModelPath);
                break;
            case "flash":
                ModelInvoker.InitializeFlash(options.FlashModelPath);
                break;
            case "pro":
                ModelInvoker.InitializePro(options.ProModelPath);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(mode), $"未知模式: {mode}");
        }
    }

    private static List<string> PickFiles(string folder, int? limit, int seed)
    {
        var files = Directory.EnumerateFiles(folder)
            .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
            .ToList();

        if (limit is null || limit <= 0 || limit >= files.Count)
            return files;

        var random = new Random(seed);
        return files
            .OrderBy(_ => random.Next())
            .Take(limit.Value)
            .ToList();
    }

    private static void PrintResult(EvaluationResult result)
    {
        Console.WriteLine($"=== {result.Mode} ===");
        Console.WriteLine($"N: {result.Total}, failed: {result.Failed}");
        Console.WriteLine($"TP: {result.TruePositive}, FN: {result.FalseNegative}");
        Console.WriteLine($"FP: {result.FalsePositive}, TN: {result.TrueNegative}");
        Console.WriteLine($"Accuracy: {result.Accuracy:P4}");
        Console.WriteLine($"TPR: {result.TruePositiveRate:P4}");
        Console.WriteLine($"FPR: {result.FalsePositiveRate:P4}");
        Console.WriteLine($"Raw score: {result.RawScore:F2}");
        Console.WriteLine($"BrewTotal proxy score: {result.BrewTotalProxyScore:F2}");
        Console.WriteLine($"Avg scan time: {result.AverageScanTime.TotalMilliseconds:F3} ms");
    }

    private static void WriteCsv(StreamWriter? writer, string mode, string path, bool label, bool? prediction, float? probability, string error)
    {
        if (writer == null)
            return;

        writer.WriteLine(string.Join(',',
            Escape(mode),
            Escape(path),
            label ? "malicious" : "clean",
            prediction.HasValue ? (prediction.Value ? "malicious" : "clean") : "",
            probability.HasValue ? probability.Value.ToString("F4", CultureInfo.InvariantCulture) : "",
            Escape(error)));
    }

    private static string Escape(string value)
    {
        if (!value.Contains(',') && !value.Contains('"') && !value.Contains('\n') && !value.Contains('\r'))
            return value;

        return $"\"{value.Replace("\"", "\"\"")}\"";
    }

    private static void PrintUsage()
    {
        Console.WriteLine("用法:");
        Console.WriteLine("  Xdows-Model-Evaluator --mode all --limit 700 --csv hard-cases.csv");
        Console.WriteLine();
        Console.WriteLine("选项:");
        Console.WriteLine("  --mode <all|standard|flash|pro>    默认 all");
        Console.WriteLine("  --black <folder>                  黑样本目录，默认 TrainingConfig.BlackFolder");
        Console.WriteLine("  --white <folder>                  白样本目录，默认 TrainingConfig.WhiteFolder");
        Console.WriteLine("  --limit <n>                       每类抽样数量，默认全量");
        Console.WriteLine("  --seed <n>                        抽样种子，默认 42");
        Console.WriteLine("  --csv <path>                      只导出 FP/FN/失败样本");
        Console.WriteLine("  --standard-model <path>           Standard ONNX 路径");
        Console.WriteLine("  --flash-model <path>              Flash ONNX 路径");
        Console.WriteLine("  --pro-model <path>                Pro ONNX 路径");
    }
}

internal sealed record EvaluationSample(string Path, bool IsMalicious);

internal sealed record EvaluationResult(
    string Mode,
    long TruePositive,
    long FalseNegative,
    long FalsePositive,
    long TrueNegative,
    long Failed,
    TimeSpan Elapsed)
{
    public long Total => TruePositive + FalseNegative + FalsePositive + TrueNegative;
    public double Accuracy => Total > 0 ? (double)(TruePositive + TrueNegative) / Total : 0;
    public double TruePositiveRate => TruePositive + FalseNegative > 0 ? (double)TruePositive / (TruePositive + FalseNegative) : 0;
    public double FalsePositiveRate => FalsePositive + TrueNegative > 0 ? (double)FalsePositive / (FalsePositive + TrueNegative) : 0;
    public double RawScore => TruePositive * 10 - FalseNegative * 7 - FalsePositive * 10;
    public double BrewTotalProxyScore => RawScore * TruePositiveRate - Math.Abs(RawScore) * FalsePositiveRate * 1.27;
    public TimeSpan AverageScanTime => Total > 0 ? TimeSpan.FromTicks(Elapsed.Ticks / Total) : TimeSpan.Zero;

    public static EvaluationResult Create(
        string mode,
        long truePositive,
        long falseNegative,
        long falsePositive,
        long trueNegative,
        long failed,
        TimeSpan elapsed)
    {
        return new EvaluationResult(mode, truePositive, falseNegative, falsePositive, trueNegative, failed, elapsed);
    }
}

internal sealed class EvaluationOptions
{
    public bool ShowHelp { get; private init; }
    public IReadOnlyList<string> Modes { get; private init; } = ["standard", "flash", "pro"];
    public string? BlackFolder { get; private init; }
    public string? WhiteFolder { get; private init; }
    public int? LimitPerClass { get; private init; }
    public int Seed { get; private init; } = 42;
    public string? CsvPath { get; private init; }
    public string? StandardModelPath { get; private init; }
    public string? FlashModelPath { get; private init; }
    public string? ProModelPath { get; private init; }

    public static EvaluationOptions Parse(string[] args)
    {
        if (args.Length == 0)
            return new EvaluationOptions();

        var options = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < args.Length; i++)
        {
            string arg = args[i];
            if (arg is "-h" or "--help" or "/?")
                return new EvaluationOptions { ShowHelp = true };

            if (!arg.StartsWith("--", StringComparison.Ordinal))
                continue;

            string key = arg[2..];
            string? value = i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal)
                ? args[++i]
                : null;
            options[key] = value;
        }

        string mode = Get(options, "mode") ?? "all";
        IReadOnlyList<string> modes = mode.Equals("all", StringComparison.OrdinalIgnoreCase)
            ? ["standard", "flash", "pro"]
            : [mode.ToLowerInvariant()];

        return new EvaluationOptions
        {
            Modes = modes,
            BlackFolder = Get(options, "black"),
            WhiteFolder = Get(options, "white"),
            LimitPerClass = int.TryParse(Get(options, "limit"), out int limit) ? limit : null,
            Seed = int.TryParse(Get(options, "seed"), out int seed) ? seed : 42,
            CsvPath = Get(options, "csv"),
            StandardModelPath = Get(options, "standard-model"),
            FlashModelPath = Get(options, "flash-model"),
            ProModelPath = Get(options, "pro-model")
        };
    }

    private static string? Get(Dictionary<string, string?> options, string key)
    {
        return options.TryGetValue(key, out string? value) && !string.IsNullOrWhiteSpace(value)
            ? value
            : null;
    }
}

internal sealed class StableRandom
{
    public static readonly Random Shared = new(44);
}
