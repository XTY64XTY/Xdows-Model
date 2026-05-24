namespace Xdows_Model_Caller;

internal class Program
{
    private static void Main(string[] args)
    {
        Console.WriteLine("Xdows Model 调用器 By Shiyi");
        Console.WriteLine();

        bool flashMode = false;
        bool proMode = false;
        string filePath = string.Empty;

        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "-f")
            {
                flashMode = true;
            }
            else if (args[i] == "-p")
            {
                proMode = true;
            }
            else if (string.IsNullOrEmpty(filePath))
            {
                filePath = args[i];
            }
        }

        if (flashMode && proMode)
        {
            Console.WriteLine("错误：-f 和 -p 参数互斥，不能同时指定。");
            return;
        }

        if (string.IsNullOrEmpty(filePath))
        {
            Console.WriteLine("用法: Xdows-Model-Caller.exe <文件路径> [-f] [-p]");
            Console.WriteLine();
            Console.WriteLine("选项:");
            Console.WriteLine("  -f    使用 Flash 模式");
            Console.WriteLine("  -p    使用 Pro 模式");
            Console.WriteLine();
            Console.WriteLine("注意: -f 和 -p 互斥，不能同时指定");
            return;
        }

        string modeName = proMode ? "Pro" : (flashMode ? "Flash" : "标准");
        Console.WriteLine($"开始扫描：{filePath}");
        Console.WriteLine($"扫描模式：{modeName}");
        Console.WriteLine();

        try
        {
            if (proMode)
            {
                Xdows_Model_Invoker.ModelInvoker.InitializePro();
            }
            else if (flashMode)
            {
                Xdows_Model_Invoker.ModelInvoker.InitializeFlash();
            }
            else
            {
                Xdows_Model_Invoker.ModelInvoker.Initialize();
            }

            var (isVirus, probability) = Xdows_Model_Invoker.ModelInvoker.ScanFile(filePath);

            if (!isVirus)
            {
                Console.WriteLine($"Safe({probability:F2}%)");
            }
            else
            {
                Console.WriteLine($"Virus({probability:F2}%)");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("错误：" + ex.Message);
        }
    }

}
