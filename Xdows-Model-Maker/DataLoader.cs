using System.Collections.Concurrent;
using System.Diagnostics;

namespace Xdows_Model_Maker;

public class DataLoader
{
    private static int _loadedCount;
    private static int _failedCount;
    private static readonly object _lockObject = new();
    private static int _lastReportedCount;
    private static Stopwatch? _loadingStopwatch;

    public static List<FileData> LoadData(TrainingConfig config)
    {
        return LoadData(config.BlackFolder, config.WhiteFolder);
    }

    public static List<FileData> LoadData(string blackFolder, string whiteFolder, bool enableParallelLoading = true, int maxParallelism = -1)
    {
        var data = new List<FileData>();

        Console.WriteLine($"正在加载黑文件: {blackFolder}");
        if (Directory.Exists(blackFolder))
        {
            var blackFiles = Directory.GetFiles(blackFolder);
            _loadedCount = 0;
            _failedCount = 0;
            _lastReportedCount = 0;
            var blackData = LoadFilesParallelAsync(blackFiles, true, enableParallelLoading).GetAwaiter().GetResult();
            data.AddRange(blackData);
            Console.WriteLine($"\n黑文件加载完成，成功 {_loadedCount} 个，失败 {_failedCount} 个");
        }
        else
        {
            Console.WriteLine($"黑文件目录不存在: {blackFolder}");
        }

        Console.WriteLine($"\n正在加载白文件: {whiteFolder}");
        if (Directory.Exists(whiteFolder))
        {
            var whiteFiles = Directory.GetFiles(whiteFolder);
            _loadedCount = 0;
            _failedCount = 0;
            _lastReportedCount = 0;
            var whiteData = LoadFilesParallelAsync(whiteFiles, false, enableParallelLoading).GetAwaiter().GetResult();
            data.AddRange(whiteData);
            Console.WriteLine($"\n白文件加载完成，成功 {_loadedCount} 个，失败 {_failedCount} 个");
        }
        else
        {
            Console.WriteLine($"白文件目录不存在: {whiteFolder}");
        }

        Console.WriteLine($"\n数据加载完成，总共 {data.Count} 个文件");
        return data;
    }

    private static async Task<List<FileData>> LoadFilesParallelAsync(string[] files, bool isBlack, bool enableParallelLoading)
    {
        var results = new ConcurrentBag<FileData>();
        int totalFiles = files.Length;


        Console.WriteLine($"文件总数：{files.Length}");

        _loadingStopwatch = Stopwatch.StartNew();

        var tasks = new Task[files.Length];
        for (int i = 0; i < files.Length; i++)
        {
            tasks[i] = ProcessSingleFileAsync(files[i], isBlack, results, totalFiles);
        }

        await Task.WhenAll(tasks);

        _loadingStopwatch.Stop();
        double filesPerSecond = totalFiles * 1000.0 / _loadingStopwatch.ElapsedMilliseconds;
        Console.WriteLine($"\n并行加载耗时: {_loadingStopwatch.ElapsedMilliseconds} ms ({filesPerSecond:F2} 文件/秒)");

        return [.. results];
    }

    private static async Task ProcessSingleFileAsync(string file, bool isBlack, ConcurrentBag<FileData> results, int totalFiles)
    {
        try
        {
            var features = await FeatureExtractor.ExtractFeaturesAsync(file);
            results.Add(new FileData
            {
                FilePath = file,
                Features = features,
                Label = isBlack
            });

            lock (_lockObject)
            {
                _loadedCount++;

                _lastReportedCount = _loadedCount;
                string label = isBlack ? "黑文件" : "白文件";

                Console.Write($"\r已加载{label} ({_loadedCount}/{totalFiles})");
            }
        }
        catch (Exception ex)
        {
            lock (_lockObject)
            {
                _failedCount++;
                Console.WriteLine($"\n加载失败 {Path.GetFileName(file)}: {ex.Message}");
                File.Delete(file);
            }
        }
    }
}

public class FileData
{
    public string FilePath { get; set; } = string.Empty;
    public FileFeatures Features { get; set; } = new FileFeatures();
    public bool Label { get; set; }
}

public class TrainingData
{
    public float[] Features { get; set; } = Array.Empty<float>();
    public bool Label { get; set; }
}
