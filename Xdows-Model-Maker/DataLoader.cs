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

    public static void CleanNonPEFiles(TrainingConfig config)
    {
        CleanNonPEFiles(config.BlackFolder, config.WhiteFolder);
    }

    public static void CleanNonPEFiles(string blackFolder, string whiteFolder)
    {
        int totalDeleted = 0;

        Console.WriteLine("开始清洗非PE文件...\n");

        // 清洗黑文件目录
        if (Directory.Exists(blackFolder))
        {
            Console.WriteLine($"正在清洗黑文件目录: {blackFolder}");
            int deleted = CleanDirectory(blackFolder, "黑文件");
            totalDeleted += deleted;
        }
        else
        {
            Console.WriteLine($"黑文件目录不存在: {blackFolder}");
        }

        // 清洗白文件目录
        if (Directory.Exists(whiteFolder))
        {
            Console.WriteLine($"\n正在清洗白文件目录: {whiteFolder}");
            int deleted = CleanDirectory(whiteFolder, "白文件");
            totalDeleted += deleted;
        }
        else
        {
            Console.WriteLine($"白文件目录不存在: {whiteFolder}");
        }

        Console.WriteLine($"\n=============================================");
        Console.WriteLine($"  清洗完成！共删除 {totalDeleted} 个非PE文件");
        Console.WriteLine("=============================================");
    }

    private static int CleanDirectory(string folder, string folderName)
    {
        var files = Directory.GetFiles(folder);
        int deletedCount = 0;
        int totalCount = files.Length;

        Console.WriteLine($"文件总数：{totalCount}");

        for (int i = 0; i < files.Length; i++)
        {
            var file = files[i];
            string fileName = Path.GetFileName(file);

            try
            {
                var bytes = File.ReadAllBytes(file);

                // 文件长度为64时不进行检测
                if (bytes.Length == 64)
                {
                    Console.Write($"\r检查{folderName} ({i + 1}/{totalCount}) - {fileName} [跳过64字节文件]");
                    continue;
                }

                if (!FeatureExtractor.IsPeFile(bytes))
                    throw new Exception("不是PE文件");

                Console.Write($"\r检查{folderName} ({i + 1}/{totalCount}) - {fileName}");
            }
            catch
            {
                File.Delete(file);
                deletedCount++;
                Console.Write($"\r已删除{folderName}非PE文件 ({deletedCount}/{totalCount}) - {fileName}");
            }
        }

        Console.WriteLine($"\n{folderName}目录清洗完成，删除 {deletedCount} 个非PE文件");
        return deletedCount;
    }
}

public class FileData
{
    public string FilePath { get; set; } = string.Empty;
    public FileFeatures Features { get; set; } = new FileFeatures();
    public bool Label { get; set; }
}

