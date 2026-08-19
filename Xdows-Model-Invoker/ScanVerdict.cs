namespace Xdows_Model_Invoker;

/// <summary>
/// 模型判定的三档结果。
/// 数值与 Native 侧 <c>XDOWS_MODEL_NATIVE_VERDICT</c> 保持一致（0=Clean, 1=Suspicious, 2=Malware），
/// 供 Managed/Native 一致性测试直接映射。
/// </summary>
public enum ScanVerdict
{
    Clean = 0,
    Suspicious = 1,
    Malware = 2
}
