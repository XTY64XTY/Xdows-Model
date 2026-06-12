param(
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Debug",

    [ValidateSet("x64", "ARM64")]
    [string]$Platform = "x64",

    [string]$SamplePath,

    [double]$Tolerance = 0.25,

    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$solution = Join-Path $repoRoot "Xdows-Model.slnx"
$msbuild = "D:\Visual-Studio\MSBuild\Current\Bin\amd64\MSBuild.exe"

if (!$SkipBuild) {
    if (!(Test-Path $msbuild)) {
        throw "MSBuild was not found: $msbuild"
    }

    & $msbuild $solution /p:Configuration=$Configuration /p:Platform=$Platform /m
    if ($LASTEXITCODE -ne 0) {
        throw "Model build failed with exit code $LASTEXITCODE"
    }
}

$nativeDir = Join-Path $repoRoot (Join-Path $Platform $Configuration)
$nativeDll = Join-Path $nativeDir "Xdows-Model-Native.dll"
$modelDir = Join-Path $repoRoot "Xdows-Model-Invoker"

foreach ($required in @($nativeDll, (Join-Path $nativeDir "onnxruntime.dll"), (Join-Path $nativeDir "onnxruntime_providers_shared.dll"))) {
    if (!(Test-Path $required)) {
        throw "Native runtime asset was not found: $required"
    }
}

foreach ($required in @("Xdows-Model.onnx", "Xdows-Model-Flash.onnx", "Xdows-Model-Pro.onnx")) {
    $path = Join-Path $modelDir $required
    if (!(Test-Path $path)) {
        throw "Model file was not found: $path"
    }
}

$callerCandidates = @(
    (Join-Path $repoRoot "Xdows-Model-Caller\bin\$Platform\$Configuration\net10.0-windows10.0.26100.0\Xdows-Model-Caller.exe"),
    (Join-Path $repoRoot "Xdows-Model-Caller\bin\$Configuration\net10.0-windows10.0.26100.0\Xdows-Model-Caller.exe"),
    (Join-Path $repoRoot "Xdows-Model-Caller\bin\$Platform\$Configuration\net10.0-windows\Xdows-Model-Caller.exe"),
    (Join-Path $repoRoot "Xdows-Model-Caller\bin\$Configuration\net10.0-windows\Xdows-Model-Caller.exe")
)

$callerExe = $callerCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (!$callerExe) {
    throw "Xdows-Model-Caller.exe was not found. Build the model solution first."
}

if ([string]::IsNullOrWhiteSpace($SamplePath)) {
    $SamplePath = $callerExe
}

if (!(Test-Path $SamplePath)) {
    throw "Sample file was not found: $SamplePath"
}

$source = @"
using System;
using System.Runtime.InteropServices;

public static class XdowsModelNativeProbe
{
    [StructLayout(LayoutKind.Sequential)]
    public struct ScanResult
    {
        public int Size;
        public int Status;
        public int IsThreat;
        public float Probability;
        public IntPtr DetectionName;
        public IntPtr ErrorMessage;
    }

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    public static extern bool SetDllDirectory(string lpPathName);

    [DllImport("Xdows-Model-Native.dll", CharSet = CharSet.Unicode, CallingConvention = CallingConvention.StdCall)]
    public static extern int XdowsModelNativeInitialize(string modelDirectory, int mode, out IntPtr session);

    [DllImport("Xdows-Model-Native.dll", CharSet = CharSet.Unicode, CallingConvention = CallingConvention.StdCall)]
    public static extern int XdowsModelNativeScanFile(IntPtr session, string filePath, out ScanResult result);

    [DllImport("Xdows-Model-Native.dll", CallingConvention = CallingConvention.StdCall)]
    public static extern void XdowsModelNativeShutdown(IntPtr session);

    [DllImport("Xdows-Model-Native.dll", CallingConvention = CallingConvention.StdCall)]
    public static extern void XdowsModelNativeFreeString(IntPtr value);

    public static string ReadAndFree(IntPtr value)
    {
        if (value == IntPtr.Zero)
        {
            return string.Empty;
        }

        try
        {
            return Marshal.PtrToStringUni(value) ?? string.Empty;
        }
        finally
        {
            XdowsModelNativeFreeString(value);
        }
    }
}
"@

Add-Type -TypeDefinition $source
[XdowsModelNativeProbe]::SetDllDirectory($nativeDir) | Out-Null

function Invoke-ManagedScan {
    param(
        [string]$ModeName,
        [string]$ModeFlag
    )

    $callerDir = Split-Path -Parent $callerExe
    Push-Location $callerDir
    try {
        $arguments = @($SamplePath)
        if (![string]::IsNullOrWhiteSpace($ModeFlag)) {
            $arguments += $ModeFlag
        }

        $output = & $callerExe @arguments 2>&1 | Out-String
        if ($output -notmatch "(Safe|Virus)\(([0-9]+(?:\.[0-9]+)?)%\)") {
            throw "Managed caller did not return a parseable result for $ModeName. Output: $output"
        }

        [pscustomobject]@{
            IsThreat = $Matches[1] -eq "Virus"
            Probability = [double]::Parse($Matches[2], [Globalization.CultureInfo]::InvariantCulture)
            Raw = $output.Trim()
        }
    }
    finally {
        Pop-Location
    }
}

function Invoke-NativeScan {
    param(
        [int]$Mode
    )

    Push-Location $nativeDir
    $session = [IntPtr]::Zero
    try {
        $initStatus = [XdowsModelNativeProbe]::XdowsModelNativeInitialize($modelDir, $Mode, [ref]$session)
        if ($initStatus -ne 0 -or $session -eq [IntPtr]::Zero) {
            throw "Native initialize failed for mode $Mode with status $initStatus"
        }

        $scanResult = New-Object XdowsModelNativeProbe+ScanResult
        $scanStatus = [XdowsModelNativeProbe]::XdowsModelNativeScanFile($session, $SamplePath, [ref]$scanResult)
        $detectionName = [XdowsModelNativeProbe]::ReadAndFree($scanResult.DetectionName)
        $errorMessage = [XdowsModelNativeProbe]::ReadAndFree($scanResult.ErrorMessage)

        if ($scanStatus -ne 0 -or $scanResult.Status -ne 0) {
            throw "Native scan failed for mode $Mode with status $scanStatus/$($scanResult.Status): $errorMessage"
        }

        [pscustomobject]@{
            IsThreat = $scanResult.IsThreat -ne 0
            Probability = [double]$scanResult.Probability
            DetectionName = $detectionName
            ErrorMessage = $errorMessage
        }
    }
    finally {
        if ($session -ne [IntPtr]::Zero) {
            [XdowsModelNativeProbe]::XdowsModelNativeShutdown($session)
        }

        Pop-Location
    }
}

$modes = @(
    @{ Name = "Standard"; NativeMode = 0; Flag = "-s" },
    @{ Name = "Flash"; NativeMode = 1; Flag = "-f" },
    @{ Name = "Pro"; NativeMode = 2; Flag = "-p" }
)

$results = foreach ($mode in $modes) {
    $managed = Invoke-ManagedScan -ModeName $mode.Name -ModeFlag $mode.Flag
    $native = Invoke-NativeScan -Mode $mode.NativeMode
    $delta = [Math]::Abs($managed.Probability - $native.Probability)

    if ($managed.IsThreat -ne $native.IsThreat) {
        throw "$($mode.Name) threat decision mismatch. Managed=$($managed.IsThreat), Native=$($native.IsThreat)"
    }

    if ($delta -gt $Tolerance) {
        throw "$($mode.Name) probability delta $delta exceeds tolerance $Tolerance. Managed=$($managed.Probability), Native=$($native.Probability)"
    }

    [pscustomobject]@{
        Mode = $mode.Name
        ManagedThreat = $managed.IsThreat
        ManagedProbability = [Math]::Round($managed.Probability, 4)
        NativeThreat = $native.IsThreat
        NativeProbability = [Math]::Round($native.Probability, 4)
        Delta = [Math]::Round($delta, 4)
        DetectionName = $native.DetectionName
    }
}

$results | Format-Table -AutoSize
Write-Host "Native consistency smoke passed for sample: $SamplePath"
