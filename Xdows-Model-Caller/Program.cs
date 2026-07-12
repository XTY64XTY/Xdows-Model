using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using Xdows_Model_Config;
using Xdows_Model_Invoker;

namespace Xdows_Model_Caller;

internal class Program
{
    private enum XdowsModelNativeMode
    {
        Standard = 0,
        Flash = 1,
        Pro = 2,
        Adaptive = 3
    }

    private enum XdowsModelNativeStatus
    {
        Ok = 0,
        InvalidArgument = 1,
        FileNotFound = 2,
        UnsupportedFile = 3,
        ModelNotFound = 4,
        InternalError = 5
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct XdowsModelNativeScanResult
    {
        public int Size;
        public int Status;
        public int IsThreat;
        public float Probability;
        public IntPtr DetectionName;
        public IntPtr ErrorMessage;
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SetConsoleOutputCP(uint wCodePageID);

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SetDllDirectory(string lpPathName);

    [DllImport("Xdows-Model-Native.dll", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Unicode)]
    private static extern int XdowsModelNativeInitialize(string modelDirectory, int mode, out IntPtr session);

    [DllImport("Xdows-Model-Native.dll", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Unicode)]
    private static extern int XdowsModelNativeScanFile(IntPtr session, string filePath, out XdowsModelNativeScanResult result);

    [DllImport("Xdows-Model-Native.dll", CallingConvention = CallingConvention.StdCall)]
    private static extern void XdowsModelNativeShutdown(IntPtr session);

    [DllImport("Xdows-Model-Native.dll", CallingConvention = CallingConvention.StdCall)]
    private static extern void XdowsModelNativeFreeString(IntPtr value);

    private static void Main(string[] args)
    {
        SetConsoleOutputCP(65001);
        Console.OutputEncoding = Encoding.UTF8;

        if (args.Length > 0 && args[0] == "-benchmark")
        {
            RunBenchmark(args);
            return;
        }

        Console.WriteLine("Xdows Model 调用器 By Shiyi");
        Console.WriteLine();

        bool standardMode = false;
        bool flashMode = false;
        bool proMode = false;
        bool adaptiveMode = false;
        bool forceManaged = false;
        string filePath = string.Empty;
        string? modelPath = null;

        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "-s")
            {
                standardMode = true;
            }
            else if (args[i] == "-f")
            {
                flashMode = true;
            }
            else if (args[i] == "-p")
            {
                proMode = true;
            }
            else if (args[i] == "-a")
            {
                adaptiveMode = true;
            }
            else if (args[i] == "-managed")
            {
                forceManaged = true;
            }
            else if (string.IsNullOrEmpty(filePath))
            {
                filePath = args[i];
            }
            else if (string.IsNullOrEmpty(modelPath))
            {
                modelPath = args[i];
            }
        }

        int modeCount = (standardMode ? 1 : 0) + (flashMode ? 1 : 0) + (proMode ? 1 : 0) + (adaptiveMode ? 1 : 0);
        if (modeCount > 1)
        {
            Console.WriteLine("错误：-s、-f、-p 和 -a 参数互斥，不能同时指定。");
            return;
        }

        if (string.IsNullOrEmpty(filePath))
        {
            Console.WriteLine("用法: Xdows-Model-Caller.exe <文件路径> [模型路径] [-s] [-f] [-p] [-a]");
            Console.WriteLine();
            Console.WriteLine("选项:");
            Console.WriteLine("  -s    使用 Standard 模型");
            Console.WriteLine("  -f    使用 Flash 模型");
            Console.WriteLine("  -p    使用 Pro 模型");
            Console.WriteLine("  -a    使用 Flash → Standard → Pro 自适应级联");
            Console.WriteLine("  -managed  强制使用托管 ONNX 推理路径");
            Console.WriteLine();
            Console.WriteLine("注意: -s、-f、-p 和 -a 互斥，不能同时指定");
            Console.WriteLine("如果模型未指定，默认使用 Standard");
            return;
        }

        if (!File.Exists(filePath))
        {
            Console.WriteLine($"错误：文件不存在：{filePath}");
            return;
        }

        string modelName = adaptiveMode ? "Adaptive" : (proMode ? "Pro" : (flashMode ? "Flash" : "Standard"));
        Console.WriteLine($"开始扫描：{filePath}");
        Console.WriteLine($"扫描模型：{modelName}");
        Console.WriteLine();

        string? nativeDllPath = FindNativeDll();
        if (!forceManaged && nativeDllPath != null && TryRunNative(nativeDllPath, filePath, modelPath, proMode, flashMode, adaptiveMode, out bool isVirus, out float probability))
        {
            PrintResult(isVirus, probability);
            return;
        }

        RunManaged(filePath, modelPath, proMode, flashMode, adaptiveMode);
    }

    private static string? FindNativeDll()
    {
        string baseDir = AppContext.BaseDirectory;
        string candidate = Path.Combine(baseDir, "Xdows-Model-Native.dll");
        if (File.Exists(candidate))
            return candidate;

        for (DirectoryInfo? directory = new(baseDir); directory != null; directory = directory.Parent)
        {
            if (!File.Exists(Path.Combine(directory.FullName, "Xdows-Model.slnx")))
                continue;

            foreach (string relativePath in new[]
            {
                Path.Combine("x64", "Release", "Xdows-Model-Native.dll"),
                Path.Combine("Xdows-Model-Native", "x64", "Release", "Xdows-Model-Native.dll")
            })
            {
                candidate = Path.Combine(directory.FullName, relativePath);
                if (File.Exists(candidate))
                    return candidate;
            }
        }

        return null;
    }

    private static bool TryRunNative(string nativeDllPath, string filePath, string? modelPath, bool proMode, bool flashMode, bool adaptiveMode, out bool isVirus, out float probability)
    {
        isVirus = false;
        probability = 0f;

        string modelDirectory = !string.IsNullOrEmpty(modelPath)
            ? Path.GetDirectoryName(modelPath) ?? AppContext.BaseDirectory
            : AppContext.BaseDirectory;

        if (!Directory.Exists(modelDirectory))
            return false;

        var mode = adaptiveMode
            ? XdowsModelNativeMode.Adaptive
            : (proMode ? XdowsModelNativeMode.Pro : (flashMode ? XdowsModelNativeMode.Flash : XdowsModelNativeMode.Standard));

        try
        {
            string? nativeDirectory = Path.GetDirectoryName(nativeDllPath);
            if (string.IsNullOrEmpty(nativeDirectory) || !SetDllDirectory(nativeDirectory))
                return false;

            int initStatus = XdowsModelNativeInitialize(modelDirectory, (int)mode, out IntPtr session);
            if (initStatus != (int)XdowsModelNativeStatus.Ok || session == IntPtr.Zero)
                return false;

            try
            {
                int scanStatus = XdowsModelNativeScanFile(session, filePath, out XdowsModelNativeScanResult result);
                if (scanStatus == (int)XdowsModelNativeStatus.Ok)
                {
                    isVirus = result.IsThreat != 0;
                    probability = result.Probability;

                    if (result.DetectionName != IntPtr.Zero)
                        XdowsModelNativeFreeString(result.DetectionName);
                    if (result.ErrorMessage != IntPtr.Zero)
                        XdowsModelNativeFreeString(result.ErrorMessage);

                    return true;
                }

                if (result.ErrorMessage != IntPtr.Zero)
                    XdowsModelNativeFreeString(result.ErrorMessage);
                if (result.DetectionName != IntPtr.Zero)
                    XdowsModelNativeFreeString(result.DetectionName);

                return false;
            }
            finally
            {
                XdowsModelNativeShutdown(session);
            }
        }
        catch (DllNotFoundException)
        {
            return false;
        }
        catch (BadImageFormatException)
        {
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    private static void RunManaged(string filePath, string? modelPath, bool proMode, bool flashMode, bool adaptiveMode)
    {
        try
        {
            if (adaptiveMode)
            {
                string? modelDirectory = string.IsNullOrWhiteSpace(modelPath)
                    ? null
                    : (Directory.Exists(modelPath) ? modelPath : Path.GetDirectoryName(modelPath));
                using var adaptiveSession = ModelInvoker.CreateAdaptiveSession(modelDirectory, new TrainingConfig());
                AdaptiveScanResult result = adaptiveSession.ScanFile(filePath);
                Console.WriteLine($"最终判定层：{result.FinalMode}");
                PrintResult(result.IsVirus, result.Probability);
                return;
            }
            else if (proMode)
            {
                if (!string.IsNullOrEmpty(modelPath))
                    Xdows_Model_Invoker.ModelInvoker.InitializePro(modelPath);
                else
                    Xdows_Model_Invoker.ModelInvoker.InitializePro();
            }
            else if (flashMode)
            {
                if (!string.IsNullOrEmpty(modelPath))
                    Xdows_Model_Invoker.ModelInvoker.InitializeFlash(modelPath);
                else
                    Xdows_Model_Invoker.ModelInvoker.InitializeFlash();
            }
            else
            {
                if (!string.IsNullOrEmpty(modelPath))
                    Xdows_Model_Invoker.ModelInvoker.Initialize(modelPath);
                else
                    Xdows_Model_Invoker.ModelInvoker.Initialize();
            }

            Xdows_Model_Invoker.ModelInvoker.ConfigureThresholds(new TrainingConfig());

            var (isVirus, probability) = Xdows_Model_Invoker.ModelInvoker.ScanFile(filePath);
            PrintResult(isVirus, probability);
        }
        catch (Exception ex)
        {
            Console.WriteLine("错误：" + ex.Message);
        }
        finally
        {
            Xdows_Model_Invoker.ModelInvoker.UnloadModel();
        }
    }

    private static void PrintResult(bool isVirus, float probability)
    {
        if (!isVirus)
        {
            Console.WriteLine($"Safe({probability:F2}%)");
        }
        else
        {
            Console.WriteLine($"Virus({probability:F2}%)");
        }
    }

    private static void RunBenchmark(string[] args)
    {
        int iterations = 100;
        string filePath = args.Length > 1 ? args[1] : string.Empty;
        if (args.Length > 2 && int.TryParse(args[2], out int it))
            iterations = it;

        if (string.IsNullOrEmpty(filePath) || !File.Exists(filePath))
        {
            Console.WriteLine("用法: -benchmark <文件路径> [迭代次数]");
            return;
        }

        byte[] bytes = File.ReadAllBytes(filePath);
        Console.WriteLine($"Benchmark: {filePath}");
        Console.WriteLine($"FileSize: {bytes.Length:N0} bytes");
        Console.WriteLine($"Iterations: {iterations}");
        Console.WriteLine();

        _ = FeatureExtractor.ExtractFromBytes(bytes);
        _ = FlashFeatureExtractor.ExtractFromBytes(bytes);
        _ = ProHybridFeatureExtractor.ExtractFromBytes(bytes);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            _ = FeatureExtractor.ExtractFromBytes(bytes);
        sw.Stop();
        Console.WriteLine($"Standard: {sw.Elapsed.TotalMilliseconds / iterations:F4} ms/iter");

        sw.Restart();
        for (int i = 0; i < iterations; i++)
            _ = FlashFeatureExtractor.ExtractFromBytes(bytes);
        sw.Stop();
        Console.WriteLine($"Flash:    {sw.Elapsed.TotalMilliseconds / iterations:F4} ms/iter");

        sw.Restart();
        for (int i = 0; i < iterations; i++)
            _ = ProHybridFeatureExtractor.ExtractFromBytes(bytes);
        sw.Stop();
        Console.WriteLine($"Pro:      {sw.Elapsed.TotalMilliseconds / iterations:F4} ms/iter");
    }
}
