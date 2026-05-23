namespace Xdows_Model_Caller;

internal class Program
{
    private static void Main(string[] args)
    {
        Console.WriteLine("Xdows Model 调用器 By Shiyi");
        Console.WriteLine();

        bool flashMode = false;
        string filePath = string.Empty;

        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "-f")
            {
                flashMode = true;
            }
            else if (string.IsNullOrEmpty(filePath))
            {
                filePath = args[i];
            }
        }

        if (string.IsNullOrEmpty(filePath))
        {
            Console.WriteLine("用法: Xdows-Model-Caller.exe <文件路径> [-f]");
            Console.WriteLine();
            Console.WriteLine("选项:");
            Console.WriteLine("  -f    使用 Flash 模式");
            return;
        }

        Console.WriteLine($"开始扫描：{filePath}");
        Console.WriteLine($"扫描模式：{(flashMode ? "Flash" : "标准")}");
        Console.WriteLine();

        try
        {
            if (flashMode)
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
