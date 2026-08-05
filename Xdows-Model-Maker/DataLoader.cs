using System.Diagnostics;
using Xdows_Model_Config;
using Xdows_Model_Invoker;

namespace Xdows_Model_Maker;

public enum DataLoadMode
{
    Standard,
    FlashOnly,
    Both,
    ProOnly,
    All
}

public class DataLoader
{
    private static int _loadedCount;
    private static int _failedCount;
    private static readonly object _lockObject = new();
    private static Stopwatch? _loadingStopwatch;
    private static DataLoadMode _currentMode;

    public static List<FileData> LoadData(TrainingConfig config)
    {
        return LoadData(config.BlackFolder, config.WhiteFolder);
    }

    public static List<FileData> LoadData(TrainingConfig config, DataLoadMode mode)
    {
        return LoadData(config.BlackFolder, config.WhiteFolder, mode: mode);
    }

    public static List<FileData> LoadData(string blackFolder, string whiteFolder, bool enableParallelLoading = true, DataLoadMode mode = DataLoadMode.Both)
    {
        _currentMode = mode;
        var data = new List<FileData>();

        Console.WriteLine($"正在加载黑文件: {blackFolder}");
        if (Directory.Exists(blackFolder))
        {
            var blackFiles = Directory.GetFiles(blackFolder);
            _loadedCount = 0;
            _failedCount = 0;
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
        var results = new FileData?[files.Length];
        int totalFiles = files.Length;

        Console.WriteLine($"文件总数：{files.Length}");

        _loadingStopwatch = Stopwatch.StartNew();

        if (enableParallelLoading)
        {
            await Parallel.ForEachAsync(
                Enumerable.Range(0, files.Length),
                new ParallelOptions
                {
                    MaxDegreeOfParallelism = Math.Max(1, Environment.ProcessorCount)
                },
                async (index, _) => await ProcessSingleFileAsync(files[index], index, isBlack, results, totalFiles));
        }
        else
        {
            for (int index = 0; index < files.Length; index++)
                await ProcessSingleFileAsync(files[index], index, isBlack, results, totalFiles);
        }

        _loadingStopwatch.Stop();
        double filesPerSecond = totalFiles * 1000.0 / _loadingStopwatch.ElapsedMilliseconds;
        Console.WriteLine($"\n并行加载耗时: {_loadingStopwatch.ElapsedMilliseconds} ms ({filesPerSecond:F2} 文件/秒)");

        var loaded = new List<FileData>(results.Length);
        foreach (FileData? fileData in results)
        {
            if (fileData is not null)
                loaded.Add(fileData);
        }
        return loaded;
    }

    private static async Task ProcessSingleFileAsync(string file, int index, bool isBlack, FileData?[] results, int totalFiles)
    {
        try
        {
            var fileData = new FileData
            {
                FilePath = file,
                Label = isBlack
            };

            switch (_currentMode)
            {
                case DataLoadMode.ProOnly:
                    {
                        var bytes = await File.ReadAllBytesAsync(file);
                        fileData.ProFeaturesAttempted = true;
                        fileData.ProFeatures = ProHybridFeatureExtractor.ExtractFromBytes(bytes).ToFloatArray();
                    }
                    break;
                case DataLoadMode.FlashOnly:
                    {
                        var bytes = await File.ReadAllBytesAsync(file);
                        fileData.FlashFeatures = FlashFeatureExtractor.ExtractFromBytes(bytes);
                    }
                    break;
                case DataLoadMode.Standard:
                    {
                        var bytes = await File.ReadAllBytesAsync(file);
                        fileData.Features = FeatureExtractor.ExtractFromBytes(bytes);
                    }
                    break;
                case DataLoadMode.Both:
                case DataLoadMode.All:
                    {
                        var bytes = await File.ReadAllBytesAsync(file);
                        fileData.Features = FeatureExtractor.ExtractFromBytes(bytes);
                        bool flashFeaturesAvailable = true;
                        try
                        {
                            fileData.FlashFeatures = FlashFeatureExtractor.ExtractFromBytes(bytes);
                        }
                        catch (NotSupportedException)
                        {
                            flashFeaturesAvailable = false;
                            fileData.FlashFeatures = new FlashFileFeatures();
                        }

                        if (_currentMode == DataLoadMode.All)
                        {
                            fileData.ProFeaturesAttempted = true;
                            if (flashFeaturesAvailable)
                            {
                                try
                                {
                                    fileData.ProFeatures = AdaptiveFeatureComposer.ComposePro(
                                        bytes,
                                        fileData.Features.ToFloatArray(),
                                        fileData.FlashFeatures.ToFloatArray());
                                }
                                catch
                                {
                                    fileData.ProFeatures = null;
                                }
                            }
                        }
                    }
                    break;
            }

            results[index] = fileData;

            int loadedCount = Interlocked.Increment(ref _loadedCount);
            int progressInterval = Math.Max(100, totalFiles / 100);
            if (loadedCount % progressInterval == 0 || loadedCount == totalFiles)
            {
                lock (_lockObject)
                {
                    string label = isBlack ? "黑文件" : "白文件";
                    Console.Write($"\r已加载{label} ({loadedCount}/{totalFiles})");
                }
            }
        }
        catch (Exception ex)
        {
            Interlocked.Increment(ref _failedCount);
            lock (_lockObject)
            {
                Console.WriteLine($"\n加载失败 {Path.GetFileName(file)}: {ex.Message}");
            }
        }
    }

    public static void CleanNonPEFiles(TrainingConfig config, bool proCheck = false)
    {
        CleanNonPEFiles(config.BlackFolder, config.WhiteFolder, proCheck, ProRawStatFeatures.SectionSize);
    }

    public static void CleanNonPEFiles(string blackFolder, string whiteFolder, bool proCheck = false, int proBytesPerSection = 512)
    {
        int totalDeleted = 0;

        Console.WriteLine(proCheck ? "开始清洗非PE文件（含Pro兼容性检查）...\n" : "开始清洗非PE文件...\n");

        if (Directory.Exists(blackFolder))
        {
            Console.WriteLine($"正在清洗黑文件目录: {blackFolder}");
            int deleted = CleanDirectory(blackFolder, "黑文件", proCheck, proBytesPerSection);
            totalDeleted += deleted;
        }
        else
        {
            Console.WriteLine($"黑文件目录不存在: {blackFolder}");
        }

        if (Directory.Exists(whiteFolder))
        {
            Console.WriteLine($"\n正在清洗白文件目录: {whiteFolder}");
            int deleted = CleanDirectory(whiteFolder, "白文件", proCheck, proBytesPerSection);
            totalDeleted += deleted;
        }
        else
        {
            Console.WriteLine($"白文件目录不存在: {whiteFolder}");
        }

        Console.WriteLine($"\n=============================================");
        if (proCheck)
            Console.WriteLine($"  清洗完成！共删除 {totalDeleted} 个非PE或不兼容Pro的文件");
        else
            Console.WriteLine($"  清洗完成！共删除 {totalDeleted} 个非PE文件");
        Console.WriteLine("=============================================");
    }

    private static int CleanDirectory(string folder, string folderName, bool proCheck = false, int proBytesPerSection = 512)
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
                if (proCheck)
                {
                    var fileInfo = new FileInfo(file);
                    if (fileInfo.Length < 64)
                        throw new Exception("文件过小，不兼容Pro模型");
                    ProHybridFeatureExtractor.ExtractFeatures(file);
                }
                else
                {
                    var bytes = File.ReadAllBytes(file);

                    if (bytes.Length == 64)
                    {
                        Console.Write($"\r检查{folderName} ({i + 1}/{totalCount}) - {fileName} [跳过64字节文件]");
                        continue;
                    }

                    if (!FeatureExtractor.IsPeFile(bytes))
                        throw new Exception("不是PE文件");
                }

                Console.Write($"\r检查{folderName} ({i + 1}/{totalCount}) - {fileName}");
            }
            catch
            {
                File.Delete(file);
                deletedCount++;
                string reason = proCheck ? "不兼容文件" : "非PE文件";
                Console.Write($"\r已删除{folderName}{reason} ({deletedCount}/{totalCount}) - {fileName}");
            }
        }

        string summaryLabel = proCheck ? "不兼容文件" : "非PE文件";
        Console.WriteLine($"\n{folderName}目录清洗完成，删除 {deletedCount} 个{summaryLabel}");
        return deletedCount;
    }
}
