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

        try
        {
            var data = DataLoader.LoadData(config);

            if (data.Count == 0)
            {
                Console.WriteLine("没有加载到任何数据，请检查文件目录是否正确。");
                return;
            }

            var blackCount = data.Count(d => d.Label);
            var whiteCount = data.Count(d => !d.Label);
            Console.WriteLine($"\n数据统计:");
            Console.WriteLine($"黑文件数量: {blackCount}");
            Console.WriteLine($"白文件数量: {whiteCount}");

            if (blackCount == 0 || whiteCount == 0)
            {
                Console.WriteLine("需要同时有黑文件和白文件才能训练模型！");
                return;
            }

            var trainer = new ModelTrainer(config);
            var model = trainer.TrainModel(data);

            Console.WriteLine("\n对部分样本进行预测测试:");
            var testSamples = data.Take(Math.Min(5, data.Count)).ToList();
            foreach (var sample in testSamples)
            {
                trainer.Predict(model, sample);
            }

            Console.WriteLine("\n=============================================");
            Console.WriteLine("  训练完成！");
            Console.WriteLine($"  ML.NET 模型已保存至: {config.ModelPath}");
            Console.WriteLine($"  ONNX 模型已保存至: {config.OnnxPath}");
            Console.WriteLine("=============================================");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n发生错误: {ex.Message}");
            Console.WriteLine($"堆栈跟踪: {ex.StackTrace}");
        }

        Console.WriteLine("\n按任意键退出...");
        Console.ReadKey();
    }
}
