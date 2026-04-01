namespace Xdows_Model_Caller;

internal class Program
{
    private static void Main(string[] args)
    {
        Console.WriteLine("Xdows Model 调用器 By Shiyi");
        Console.WriteLine();

        string filePath = string.Empty;
        if (args.Length > 0)
        {
            filePath = args[0];
        }
        else
        {
            Console.WriteLine("用法: Xdows-Model-Caller.exe <文件路径>");
            return;
        }

        Console.WriteLine($"开始扫描：{filePath}");
        Console.WriteLine("测试模型：开启");
        Console.WriteLine();

        if (!File.Exists(filePath))
        {
            Console.WriteLine("错误：文件不存在！");
            return;
        }

        try
        {
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
            // Print full exception including inner exceptions to reveal the real cause (TargetInvocationException wraps inner)
            Console.WriteLine("错误：发生异常，详细信息：");
            Console.WriteLine(ex.ToString());
            if (ex.InnerException != null)
            {
                Console.WriteLine("内部异常：");
                Console.WriteLine(ex.InnerException.ToString());
            }
        }
    }

}

