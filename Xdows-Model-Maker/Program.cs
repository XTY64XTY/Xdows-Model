using Xdows_Model_Config;

namespace Xdows_Model_Maker;

internal class Program
{
    private static CancellationTokenSource? _proCts;

    private enum MenuItemType
    {
        Header,
        Checkbox,
        Action
    }

    private class MenuItem
    {
        public string Text { get; set; } = "";
        public MenuItemType Type { get; set; }
        public bool IsChecked { get; set; }
        public Action? Action { get; set; }
        public Func<bool>? IsEnabled { get; set; }
    }

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

        Console.WriteLine("\n（↑ ↓ 选择，Enter 确定/切换选中状态）\n");

        var menuItems = new List<MenuItem>
        {
            new() { Text = "模型训练", Type = MenuItemType.Header },
            new() { Text = "Flash 模型", Type = MenuItemType.Checkbox, IsChecked = false },
            new() { Text = "Standard 模型", Type = MenuItemType.Checkbox, IsChecked = false },
            new() { Text = "Pro 模型", Type = MenuItemType.Checkbox, IsChecked = false },
            new() { Text = "", Type = MenuItemType.Header },
            new() { Text = "开始训练", Type = MenuItemType.Action },
            new() { Text = "", Type = MenuItemType.Header },
            new() { Text = "样本清洗", Type = MenuItemType.Header },
            new() { Text = "含Pro兼容性检查", Type = MenuItemType.Checkbox, IsChecked = false },
            new() { Text = "", Type = MenuItemType.Header },
            new() { Text = "开始清洗", Type = MenuItemType.Action },
            new() { Text = "", Type = MenuItemType.Header },
            new() { Text = "其它选项", Type = MenuItemType.Header },
            new() { Text = "退出程序", Type = MenuItemType.Action }
        };

        int selectedIndex = 1;
        bool exitMenu = false;

        while (!exitMenu)
        {
            DrawMenu(menuItems, selectedIndex);

            var key = Console.ReadKey(intercept: true);
            switch (key.Key)
            {
                case ConsoleKey.UpArrow:
                    selectedIndex = GetPreviousSelectableIndex(menuItems, selectedIndex);
                    break;
                case ConsoleKey.DownArrow:
                    selectedIndex = GetNextSelectableIndex(menuItems, selectedIndex);
                    break;
                case ConsoleKey.Enter:
                    var item = menuItems[selectedIndex];
                    switch (item.Type)
                    {
                        case MenuItemType.Checkbox:
                            item.IsChecked = !item.IsChecked;
                            break;
                        case MenuItemType.Action:
                            if (item.Text == "开始训练")
                            {
                                var config = new TrainingConfig();
                                ExecuteTraining(config, menuItems);
                                exitMenu = true;
                            }
                            else if (item.Text == "开始清洗")
                            {
                                var config = new TrainingConfig();
                                ExecuteCleaning(config, menuItems);
                                exitMenu = true;
                            }
                            else if (item.Text == "退出程序")
                            {
                                exitMenu = true;
                            }
                            break;
                    }
                    break;
                case ConsoleKey.Escape:
                    exitMenu = true;
                    break;
            }
        }

        Console.WriteLine("\n按任意键退出...");
        Console.ReadKey();
    }

    private static void ClearConsole()
    {
        // 清除整个屏幕缓冲区，包括滚动区域
        Console.Clear();

        // 尝试清除滚动缓冲区（Windows 特定）
        try
        {
            if (OperatingSystem.IsWindows())
            {
                // 使用 ANSI 转义序列清除屏幕和滚动缓冲区
                Console.Write("\x1b[3J");
                Console.Write("\x1b[H");
            }
        }
        catch
        {
            // 如果 ANSI 转义序列不支持，忽略错误
        }
    }

    private static void DrawMenu(List<MenuItem> items, int selectedIndex)
    {
        // 清除整个屏幕缓冲区（包括滚动区域）
        ClearConsole();

        // 重新绘制标题
        Console.WriteLine("================================================================");
        Console.WriteLine(@" __  __   _                      __  __           _      _ ");
        Console.WriteLine(@" \ \/ /__| | _____      _____   |  \/  | ___   __| | ___| |");
        Console.WriteLine(@"  \  // _` |/ _ \ \ /\ / / __|  | |\/| |/ _ \ / _` |/ _ \ |");
        Console.WriteLine(@"  /  \ (_| | (_) \ V  V /\__ \  | |  | | (_) | (_| |  __/ |");
        Console.WriteLine(@" /_/\_\__,_|\___/ \_/\_/ |___/  |_|  |_|\___/ \__,_|\___|_|");
        Console.WriteLine("\n                                                  —— By Shiyi");
        Console.WriteLine("================================================================");
        Console.WriteLine("\n（↑ ↓ 选择，Enter 确定/切换选中状态）\n");

        // 绘制菜单项
        for (int i = 0; i < items.Count; i++)
        {
            var item = items[i];
            bool isSelected = i == selectedIndex;

            if (item.Type == MenuItemType.Header)
            {
                if (!string.IsNullOrEmpty(item.Text))
                {
                    Console.WriteLine($"{item.Text}");
                }
                else
                {
                    Console.WriteLine();
                }
            }
            else if (item.Type == MenuItemType.Checkbox)
            {
                string checkbox = item.IsChecked ? "[x]" : "[ ]";
                string prefix = isSelected ? "> " : "  ";
                Console.WriteLine($"{prefix}{checkbox} {item.Text}");
            }
            else if (item.Type == MenuItemType.Action)
            {
                string prefix = isSelected ? "> " : "  ";
                Console.WriteLine($"{prefix}{item.Text}");
            }
        }
    }

    private static int GetPreviousSelectableIndex(List<MenuItem> items, int currentIndex)
    {
        int index = currentIndex;
        do
        {
            index--;
            if (index < 0) index = items.Count - 1;
        } while (items[index].Type == MenuItemType.Header && index != currentIndex);
        return index;
    }

    private static int GetNextSelectableIndex(List<MenuItem> items, int currentIndex)
    {
        int index = currentIndex;
        do
        {
            index++;
            if (index >= items.Count) index = 0;
        } while (items[index].Type == MenuItemType.Header && index != currentIndex);
        return index;
    }

    private static void ExecuteTraining(TrainingConfig config, List<MenuItem> menuItems)
    {
        bool trainFlash = menuItems[1].IsChecked;
        bool trainStandard = menuItems[2].IsChecked;
        bool trainPro = menuItems[3].IsChecked;

        if (!trainFlash && !trainStandard && !trainPro)
        {
            Console.WriteLine("\n请至少选择一个模型进行训练！");
            return;
        }

        DataLoadMode mode;
        if (trainFlash && trainStandard && trainPro)
            mode = DataLoadMode.All;
        else if (trainFlash && trainStandard)
            mode = DataLoadMode.Both;
        else if (trainFlash)
            mode = DataLoadMode.FlashOnly;
        else if (trainStandard)
            mode = DataLoadMode.Standard;
        else
            mode = DataLoadMode.ProOnly;

        try
        {
            var data = DataLoader.LoadData(config, mode);

            if (!ValidateData(data))
                return;

            var trainer = new ModelTrainer(config);
            var testSamples = data.Take(Math.Min(5, data.Count)).ToList();

            if (trainStandard)
            {
                Console.WriteLine("\n=============================================");
                Console.WriteLine("  开始训练 Standard 模型...");
                Console.WriteLine("=============================================");
                config.PrintStandardConfig();
                var model = trainer.TrainModel(data);

                Console.WriteLine("\n对部分样本进行预测测试:");
                foreach (var sample in testSamples)
                    trainer.Predict(model, sample);

                Console.WriteLine("\n=============================================");
                Console.WriteLine("  Standard 模型训练完成！");
                Console.WriteLine($"  ML.NET 模型已保存至: {config.ModelPath}");
                Console.WriteLine($"  ONNX 模型已保存至: {config.OnnxPath}");
                Console.WriteLine("=============================================");
            }

            if (trainFlash)
            {
                Console.WriteLine("\n=============================================");
                Console.WriteLine("  开始训练 Flash 模型...");
                Console.WriteLine("=============================================");
                config.PrintFlashConfig();
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

            if (trainPro)
            {
                Console.WriteLine("\n=============================================");
                Console.WriteLine("  开始训练 Pro 模型...");
                Console.WriteLine("  提示：按 S 键可随时停止并保存当前最优模型");
                Console.WriteLine("=============================================");
                config.PrintProConfig();

                _proCts = new CancellationTokenSource();
                var keyboardThread = new Thread(() => ProKeyboardListener(trainer, _proCts));
                keyboardThread.IsBackground = true;
                keyboardThread.Start();

                try
                {
                    var (proModel, optimalBytesPerSection) = trainer.TrainProModel(data);

                    if (proModel == null)
                    {
                        Console.WriteLine("\nPro 模型训练失败，跳过预测测试。");
                    }
                    else
                    {
                        Console.WriteLine("\n对部分样本进行 Pro 预测测试:");
                        foreach (var sample in testSamples)
                            trainer.PredictPro(proModel, sample, optimalBytesPerSection);
                    }

                    Console.WriteLine("\n=============================================");
                    Console.WriteLine("  Pro 模型训练完成！");
                    Console.WriteLine($"  Pro ML.NET 模型已保存至: {config.ProModelPath}");
                    Console.WriteLine($"  Pro ONNX 模型已保存至: {config.ProOnnxPath}");
                    Console.WriteLine($"  最优每段字节数: {optimalBytesPerSection}");
                    Console.WriteLine($"  最优特征维度: {ProFileFeatures.SectionCount * optimalBytesPerSection}");
                    Console.WriteLine("=============================================");
                }
                finally
                {
                    _proCts?.Cancel();
                    keyboardThread.Join(500);
                    _proCts?.Dispose();
                    _proCts = null;
                }
            }

            if ((trainFlash && trainStandard) || trainPro || (trainFlash && trainStandard && trainPro))
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

    private static void ExecuteCleaning(TrainingConfig config, List<MenuItem> menuItems)
    {
        bool proCheck = menuItems[8].IsChecked;

        try
        {
            if (proCheck)
            {
                DataLoader.CleanNonPEFiles(config, proCheck: true);
            }
            else
            {
                DataLoader.CleanNonPEFiles(config);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n发生错误: {ex.Message}");
            Console.WriteLine($"堆栈跟踪: {ex.StackTrace}");
        }
    }

    private static void ProKeyboardListener(ModelTrainer trainer, CancellationTokenSource cts)
    {
        while (!cts.Token.IsCancellationRequested)
        {
            if (Console.KeyAvailable)
            {
                var key = Console.ReadKey(intercept: true);
                if (key.Key == ConsoleKey.S)
                {
                    Console.WriteLine("\n[系统] 检测到停止请求 (S键)，将在当前步骤完成后保存最优模型并停止...");
                    trainer.CancelProTraining();
                    break;
                }
            }
            Thread.Sleep(100);
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
}
