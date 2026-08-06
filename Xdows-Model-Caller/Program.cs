using Xdows_Model_Config;
using Xdows_Model_Invoker;

namespace Xdows_Model_Caller;

internal static class Program
{
    private enum CallerMode
    {
        Standard,
        Flash,
        Pro,
        Adaptive
    }

    private sealed record CallerOptions(CallerMode Mode, string? ModelPath, bool UseFixedThreshold);

    private interface IScanEngine : IDisposable
    {
        CallerMode Mode { get; }

        void Initialize();

        ScanResult Scan(string filePath);
    }

    private readonly record struct ScanResult(bool IsThreat, float Probability);

    private static void Main(string[] args)
    {
        PrintBanner();

        if (args.Any(IsHelpArgument))
        {
            PrintUsage();
            return;
        }

        if (!TryParseOptions(args, out CallerOptions options, out string? error))
        {
            Console.WriteLine($"参数错误：{error}");
            Console.WriteLine();
            PrintUsage();
            return;
        }

        try
        {
            ModelInvoker.AutoThresholdSelection = !options.UseFixedThreshold;
            using IScanEngine engine = CreateEngine(options);
            Console.WriteLine($"正在初始化 {engine.Mode} 模型...");
            engine.Initialize();
            Console.WriteLine($"{engine.Mode} 模型初始化完成。");
            PrintActiveThreshold(engine.Mode);
            Console.WriteLine();

            RunInputLoop(engine);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"模型返回：Error(初始化模型过程中发生错误：{ToSingleLine(ex.Message)})");
        }
    }

    private static void PrintBanner()
    {
        Console.WriteLine("Xdows Model 调用器 By Shiyi");
        Console.WriteLine("输入 Help 以获取帮助");
        Console.WriteLine();
    }

    private static void RunInputLoop(IScanEngine engine)
    {
        while (true)
        {
            Console.Write("输入操作：");
            string? input = Console.ReadLine();
            if (input is null)
                return;

            string filePath = Unquote(input.Trim());
            if (IsHelpCommand(filePath))
            {
                Console.WriteLine();
                PrintUsage();
                Console.WriteLine();
                continue;
            }

            if (IsQuitCommand(filePath))
                return;
            Console.WriteLine($"扫描模式：{engine.Mode}");

            if (filePath.Length == 0)
            {
                Console.WriteLine("模型返回：Error(文件名没有被指定)\n");
                continue;
            }

            try
            {
                if (!File.Exists(filePath))
                {
                    Console.WriteLine($"模型返回：Error(找不到指定文件：{filePath})\n");
                    continue;
                }

                ScanResult result = engine.Scan(filePath);
                string verdict = result.IsThreat ? "Virus" : "Safe";
                Console.WriteLine($"模型返回：{verdict}({result.Probability:F2}%)\n");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"模型返回：Error({ToSingleLine(ex.Message)})\n");
            }
        }
    }

    private static void PrintActiveThreshold(CallerMode mode)
    {
        ModelMode modelMode = mode switch
        {
            CallerMode.Flash => ModelMode.Flash,
            CallerMode.Pro => ModelMode.Pro,
            CallerMode.Adaptive => ModelMode.Adaptive,
            _ => ModelMode.Standard
        };

        Console.WriteLine(
            $"判毒阈值：{ModelInvoker.GetThreshold(modelMode):F2}%（{ModelInvoker.GetThresholdSource(modelMode)}）");
    }

    private static IScanEngine CreateEngine(CallerOptions options) =>
        options.Mode switch
        {
            CallerMode.Standard => new SingleModelScanEngine(CallerMode.Standard, ModelInvoker.Initialize, options.ModelPath),
            CallerMode.Flash => new SingleModelScanEngine(CallerMode.Flash, ModelInvoker.InitializeFlash, options.ModelPath),
            CallerMode.Pro => new SingleModelScanEngine(CallerMode.Pro, ModelInvoker.InitializePro, options.ModelPath),
            CallerMode.Adaptive => new AdaptiveScanEngine(options.ModelPath),
            _ => throw new ArgumentOutOfRangeException(nameof(options.Mode), options.Mode, "不支持的模型模式。")
        };

    private sealed class SingleModelScanEngine : IScanEngine
    {
        private readonly Action<string?> _initialize;
        private readonly string? _modelPath;
        private bool _initialized;

        public SingleModelScanEngine(CallerMode mode, Action<string?> initialize, string? modelPath)
        {
            Mode = mode;
            _initialize = initialize;
            _modelPath = modelPath;
        }

        public CallerMode Mode { get; }

        public void Initialize()
        {
            // 先落配置默认值，再加载模型：加载过程会用模型旁的阈值清单覆盖默认值。
            // 顺序颠倒会让固定阈值把清单推荐值覆盖掉。
            ModelInvoker.ConfigureThresholds(new TrainingConfig());
            _initialize(_modelPath);
            _initialized = true;
        }

        public ScanResult Scan(string filePath)
        {
            if (!_initialized)
                throw new InvalidOperationException($"{Mode} 模型尚未初始化。");

            var (isThreat, probability) = ModelInvoker.ScanFile(filePath);
            return new ScanResult(isThreat, probability);
        }

        public void Dispose()
        {
            if (_initialized)
                ModelInvoker.UnloadModel();
        }
    }

    private sealed class AdaptiveScanEngine : IScanEngine
    {
        private readonly string? _modelDirectory;
        private AdaptiveModelSession? _session;

        public AdaptiveScanEngine(string? modelPath)
        {
            _modelDirectory = ResolveModelDirectory(modelPath);
        }

        public CallerMode Mode => CallerMode.Adaptive;

        public void Initialize()
        {
            _session = ModelInvoker.CreateAdaptiveSession(_modelDirectory, new TrainingConfig());
        }

        public ScanResult Scan(string filePath)
        {
            AdaptiveModelSession session = _session ??
                throw new InvalidOperationException("Adaptive 模型尚未初始化。");
            AdaptiveScanResult result = session.ScanFile(filePath);
            return new ScanResult(result.IsVirus, result.Probability);
        }

        public void Dispose() => _session?.Dispose();
    }

    private static string? ResolveModelDirectory(string? modelPath)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            return null;
        return Directory.Exists(modelPath) ? modelPath : Path.GetDirectoryName(modelPath);
    }

    private static bool TryParseOptions(
        string[] args,
        out CallerOptions options,
        out string? error)
    {
        CallerMode mode = CallerMode.Standard;
        bool modeSpecified = false;
        string? modelPath = null;
        bool useFixedThreshold = false;

        for (int index = 0; index < args.Length; index++)
        {
            string argument = args[index];
            if (TryParseMode(argument, out CallerMode parsedMode))
            {
                if (modeSpecified)
                {
                    options = null!;
                    error = "-s、-f、-p 和 -a 只能指定一个。";
                    return false;
                }

                mode = parsedMode;
                modeSpecified = true;
                continue;
            }

            if (string.Equals(argument, "--fixed-threshold", StringComparison.OrdinalIgnoreCase))
            {
                useFixedThreshold = true;
                continue;
            }

            if (string.Equals(argument, "--model", StringComparison.OrdinalIgnoreCase))
            {
                if (++index >= args.Length)
                {
                    options = null!;
                    error = "--model 后缺少模型路径。";
                    return false;
                }

                modelPath = Unquote(args[index]);
                continue;
            }

            options = null!;
            error = $"无法识别参数：{argument}";
            return false;
        }

        options = new CallerOptions(mode, modelPath, useFixedThreshold);
        error = null;
        return true;
    }

    private static bool TryParseMode(string argument, out CallerMode mode)
    {
        mode = argument.ToLowerInvariant() switch
        {
            "-s" => CallerMode.Standard,
            "-f" => CallerMode.Flash,
            "-p" => CallerMode.Pro,
            "-a" => CallerMode.Adaptive,
            _ => (CallerMode)(-1)
        };
        return Enum.IsDefined(mode);
    }

    private static bool IsHelpArgument(string argument) =>
        argument.Equals("-h", StringComparison.OrdinalIgnoreCase) ||
        argument.Equals("--help", StringComparison.OrdinalIgnoreCase) ||
        argument.Equals("/?", StringComparison.OrdinalIgnoreCase);

    private static bool IsHelpCommand(string input) =>
        input.Equals("Help", StringComparison.OrdinalIgnoreCase);

    private static bool IsQuitCommand(string input) =>
        input.Equals("Quit", StringComparison.OrdinalIgnoreCase);

    private static string Unquote(string value)
    {
        string trimmed = value.Trim();
        if (trimmed.Length >= 2 &&
            ((trimmed[0] == '"' && trimmed[^1] == '"') ||
             (trimmed[0] == '\'' && trimmed[^1] == '\'')))
        {
            return trimmed[1..^1];
        }

        return trimmed;
    }

    private static string ToSingleLine(string message) =>
        message.Replace('\r', ' ').Replace('\n', ' ');

    private static void PrintUsage()
    {
        Console.WriteLine("帮助菜单");
        Console.WriteLine();
        Console.WriteLine("用法：");
        Console.WriteLine("  Xdows-Model-Caller.exe [-s|-f|-p|-a] [--model <模型路径>]");
        Console.WriteLine();
        Console.WriteLine("模型模式：");
        Console.WriteLine("  -s                  Standard 模式（默认）");
        Console.WriteLine("  -f                  Flash 模式");
        Console.WriteLine("  -p                  Pro 模式");
        Console.WriteLine("  -a                  Adaptive 模式");
        Console.WriteLine();
        Console.WriteLine("其他选项：");
        Console.WriteLine("  --model <路径>      指定模型文件；Adaptive 模式可指定模型目录");
        Console.WriteLine("  --fixed-threshold   忽略模型旁的阈值清单，使用配置里的固定阈值");
        Console.WriteLine("  -h, --help, /?      显示此帮助菜单");
        Console.WriteLine();
        Console.WriteLine("阈值说明：");
        Console.WriteLine("  默认自动采用模型旁 *.threshold.json 中训练阶段校准出的推荐阈值。");
        Console.WriteLine("  清单缺失或无效时回退到配置里的固定阈值，初始化后会打印实际生效的阈值。");
        Console.WriteLine();
        Console.WriteLine("交互说明：");
        Console.WriteLine("  模型初始化后可连续输入需要扫描的文件名。");
        Console.WriteLine("  带空格的文件名可以使用单引号或双引号包裹。");
        Console.WriteLine("  输入 Help 可显示此帮助菜单。");
        Console.WriteLine("  输入 Quit 可关闭调用器。");
        Console.WriteLine();
        Console.WriteLine("示例：");
        Console.WriteLine("  Xdows-Model-Caller.exe");
        Console.WriteLine("  Xdows-Model-Caller.exe -p");
        Console.WriteLine("  Xdows-Model-Caller.exe -a --model \"D:\\Models\"");
    }
}
