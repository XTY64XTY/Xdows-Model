namespace Xdows_Model_Maker;

internal class Program
{
    private static void Main(string[] args)
    {
        Console.WriteLine("================================================================");
        Console.WriteLine(@" __  __   _                      __  __           _      _ ");
        Console.WriteLine(@" \ \/ /__| | _____      _____   |  \/  | ___   __| | ___| |");
        Console.WriteLine(@"  \  // _` |/ _ \ \ /\ / / __|  | |\/| |/ _ \ / _` |/ _ \ |");
        Console.WriteLine(@"  /  \ (_| | (_) \ V  V /\__ \  | |  | | (_) | (_| |  __/ |");
        Console.WriteLine(@" /_/\_\__,_|\___/ \_/\_/ |___/  |_|  |_|\___/ \__,_|\___|_|");
        Console.WriteLine("\n                                                  —— By Shiyi");
        Console.WriteLine("================================================================");

        var config = new TrainingConfig();
        config.PrintConfig();

        Console.WriteLine("\n请选择操作模式:");
        Console.WriteLine("  1 - 训练标准模型");
        Console.WriteLine("  2 - 训练 Flash 模型");
        Console.WriteLine("  3 - 同时训练标准模型和 Flash 模型");
        Console.WriteLine("  4 - 清洗非PE文件");
        Console.WriteLine("  0 - 退出");
        Console.Write("\n请输入选项: ");

        string? choice = Console.ReadLine();

        switch (choice)
        {
            case "1":
                TrainAndEvaluate(config, DataLoadMode.Standard);
                break;
            case "2":
                TrainAndEvaluate(config, DataLoadMode.FlashOnly);
                break;
            case "3":
                TrainAndEvaluate(config, DataLoadMode.Both);
                break;
            case "4":
                CleanFiles(config);
                break;
            case "0":
                Console.WriteLine("退出程序。");
                break;
            default:
                Console.WriteLine("无效的选项，退出程序。");
                break;
        }

        Console.WriteLine("\n按任意键退出...");
        Console.ReadKey();
    }

    private static void TrainAndEvaluate(TrainingConfig config, DataLoadMode mode)
    {
        try
        {
            var data = DataLoader.LoadData(config, mode);

            if (!ValidateData(data))
                return;

            var trainer = new ModelTrainer(config);
            var testSamples = data.Take(Math.Min(5, data.Count)).ToList();

            if (mode == DataLoadMode.Standard || mode == DataLoadMode.Both)
            {
                Console.WriteLine("\n=============================================");
                Console.WriteLine("  开始训练标准模型...");
                Console.WriteLine("=============================================");
                var model = trainer.TrainModel(data);

                Console.WriteLine("\n对部分样本进行预测测试:");
                foreach (var sample in testSamples)
                    trainer.Predict(model, sample);

                Console.WriteLine("\n=============================================");
                Console.WriteLine("  标准模型训练完成！");
                Console.WriteLine($"  ML.NET 模型已保存至: {config.ModelPath}");
                Console.WriteLine($"  ONNX 模型已保存至: {config.OnnxPath}");
                Console.WriteLine("=============================================");
            }

            if (mode == DataLoadMode.FlashOnly || mode == DataLoadMode.Both)
            {
                Console.WriteLine("\n=============================================");
                Console.WriteLine("  开始训练 Flash 模型...");
                Console.WriteLine("=============================================");
                var flashModel = trainer.TrainFlashModel(data);

                Console.WriteLine("\n对部分样本进行 Flash 预测测试:");
                foreach (var sample in testSamples)
                    trainer.PredictFlash(flashModel, sample);

                Console.WriteLine("\n=============================================");
                Console.WriteLine("  Flash 模型训练完成！");
                Console.WriteLine($"  Flash ML.NET 模型已保存至: {config.FlashModelPath}");
                Console.WriteLine($"  Flash ONNX 模型已保存至: {config.FlashOnnxPath}");
                Console.WriteLine("=============================================");
            }

            if (mode == DataLoadMode.Both)
            {
                Console.WriteLine("\n*********************************************");
                Console.WriteLine("  所有模型训练完成！");
                Console.WriteLine("*********************************************");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n发生错误: {ex.Message}");
            Console.WriteLine($"堆栈跟踪: {ex.StackTrace}");
        }
    }

    private static bool ValidateData(List<FileData> data)
    {
        if (data.Count == 0)
        {
            Console.WriteLine("没有加载到任何数据，请检查文件目录是否正确。");
            return false;
        }

        var blackCount = data.Count(d => d.Label);
        var whiteCount = data.Count(d => !d.Label);
        Console.WriteLine($"\n数据统计:");
        Console.WriteLine($"黑文件数量: {blackCount}");
        Console.WriteLine($"白文件数量: {whiteCount}");

        if (blackCount == 0 || whiteCount == 0)
        {
            Console.WriteLine("需要同时有黑文件和白文件才能训练模型！");
            return false;
        }

        return true;
    }

    private static void CleanFiles(TrainingConfig config)
    {
        try
        {
            DataLoader.CleanNonPEFiles(config);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n发生错误: {ex.Message}");
            Console.WriteLine($"堆栈跟踪: {ex.StackTrace}");
        }
    }
}
