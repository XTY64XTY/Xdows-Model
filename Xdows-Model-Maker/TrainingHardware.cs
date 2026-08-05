using System.Runtime.InteropServices;

namespace Xdows_Model_Maker;

internal static class TrainingHardware
{
    private const int RelationProcessorCore = 0;
    private const int ErrorInsufficientBuffer = 122;
    private static readonly Lazy<int> LazyPhysicalCoreCount = new(DetectPhysicalCoreCount);

    public static int PhysicalCoreCount => LazyPhysicalCoreCount.Value;

    public static int ResolveTrainingThreadCount(int? configuredThreadCount)
    {
        int logicalCoreCount = Math.Max(1, Environment.ProcessorCount);
        if (configuredThreadCount is { } configured && configured > 0)
            return Math.Min(configured, logicalCoreCount);
        return PhysicalCoreCount;
    }

    private static int DetectPhysicalCoreCount()
    {
        int logicalCoreCount = Math.Max(1, Environment.ProcessorCount);
        if (!OperatingSystem.IsWindows())
            return logicalCoreCount;

        try
        {
            uint length = 0;
            if (GetLogicalProcessorInformationEx(RelationProcessorCore, IntPtr.Zero, ref length) ||
                Marshal.GetLastWin32Error() != ErrorInsufficientBuffer ||
                length == 0)
            {
                return logicalCoreCount;
            }

            IntPtr buffer = Marshal.AllocHGlobal((int)length);
            try
            {
                if (!GetLogicalProcessorInformationEx(RelationProcessorCore, buffer, ref length))
                    return logicalCoreCount;

                int coreCount = 0;
                int offset = 0;
                while (offset + sizeof(int) * 2 <= (int)length)
                {
                    int recordSize = Marshal.ReadInt32(buffer, offset + sizeof(int));
                    if (recordSize <= 0)
                        break;
                    coreCount++;
                    offset += recordSize;
                }

                return coreCount > 0 ? Math.Min(coreCount, logicalCoreCount) : logicalCoreCount;
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }
        }
        catch
        {
            return logicalCoreCount;
        }
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GetLogicalProcessorInformationEx(
        int relationshipType,
        IntPtr buffer,
        ref uint returnedLength);
}
