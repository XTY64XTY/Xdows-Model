param(
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release",

    [ValidateSet("x64", "ARM64")]
    [string]$Platform = "x64",

    [string]$SamplePath,

    [double]$Tolerance = 0.25,

    [switch]$SkipBuild,

    [switch]$SkipNative
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$solution = Join-Path $repoRoot "Xdows-Model.slnx"
$msbuild = "D:\Visual-Studio\MSBuild\Current\Bin\amd64\MSBuild.exe"

function Invoke-DotNetBuild {
    $projects = @(
        "Xdows-Model-Invoker",
        "Xdows-Model-Config",
        "Xdows-Model-Caller",
        "Xdows-Model-Maker",
        "Xdows-Model-Evaluator"
    )

    foreach ($proj in $projects) {
        $csproj = Join-Path $repoRoot "$proj\$proj.csproj"
        dotnet build $csproj -c $Configuration --nologo -v q
        if ($LASTEXITCODE -ne 0) {
            throw "dotnet build failed for $proj"
        }
    }
}

if (!$SkipBuild) {
    if (Test-Path $msbuild) {
        & $msbuild $solution /p:Configuration=$Configuration /p:Platform=$Platform /m
        if ($LASTEXITCODE -ne 0) {
            throw "MSBuild failed with exit code $LASTEXITCODE"
        }
    }
    else {
        Write-Warning "MSBuild not found at $msbuild; building managed projects only."
        Invoke-DotNetBuild
    }
}

$nativeDir = Join-Path $repoRoot (Join-Path $Platform $Configuration)
$nativeDll = Join-Path $nativeDir "Xdows-Model-Native.dll"
$modelDir = Join-Path $repoRoot "Xdows-Model-Invoker"

$nativeAvailable = !$SkipNative -and (Test-Path $nativeDll)
if ($nativeAvailable) {
    foreach ($required in @($nativeDll, (Join-Path $nativeDir "onnxruntime.dll"), (Join-Path $nativeDir "onnxruntime_providers_shared.dll"))) {
        if (!(Test-Path $required)) {
            throw "Native runtime asset was not found: $required"
        }
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

$tempDir = Join-Path $env:TEMP "XdowsModelConsistency-$(Get-Random)"
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

try {
    $eicarPath = Join-Path $tempDir "eicar.com"
    "X5O!P%@AP[4\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*" | Set-Content -Path $eicarPath -NoNewline -Encoding ASCII

    $emptyPath = Join-Path $tempDir "empty.exe"
    [System.IO.File]::WriteAllBytes($emptyPath, [byte[]]::new(0))

    $truncatedPePath = Join-Path $tempDir "truncated.exe"
    [System.IO.File]::WriteAllBytes($truncatedPePath, [byte[]](0x4D, 0x5A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00))

    $nonPePath = Join-Path $tempDir "notpe.txt"
    "This is not a PE file." | Set-Content -Path $nonPePath -Encoding UTF8 -NoNewline

    if ([string]::IsNullOrWhiteSpace($SamplePath)) {
        $SamplePath = $callerExe
    }

    $managedSamples = @($SamplePath, $truncatedPePath, $nonPePath, $emptyPath)
    $nativeSamples = if ($nativeAvailable) { @($SamplePath) } else { @() }

    if ($nativeAvailable) {
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
    }

    function Invoke-ManagedScan {
        param(
            [string]$Sample,
            [string]$ModeName,
            [string]$ModeFlag
        )

        $callerDir = Split-Path -Parent $callerExe
        Push-Location $callerDir
        try {
            $arguments = @()
            if (![string]::IsNullOrWhiteSpace($ModeFlag)) {
                $arguments += $ModeFlag
            }
            $output = @($Sample, "QUIT") | & $callerExe @arguments 2>&1 | Out-String
            if ($output -match "(Safe|Virus)\(([0-9]+(?:\.[0-9]+)?)%\)") {
                [pscustomobject]@{
                    Success = $true
                    IsThreat = $Matches[1] -eq "Virus"
                    Probability = [double]::Parse($Matches[2], [Globalization.CultureInfo]::InvariantCulture)
                    ExpectedFailure = $false
                    FailureReason = $null
                    Raw = $output.Trim()
                }
            }
            else {
                $outputText = [string]$output
                $isDimMismatch = [regex]::IsMatch($outputText, "模型特征维度不匹配|模型维度不匹配|dimension mismatch|Feature dimension mismatch|维，期望")
                $isUnsupportedFile = [regex]::IsMatch($outputText, "文件过小|不支持该文件类型|文件不存在|找不到指定文件|NotSupported|FileNotFound|不是有效的PE|不是PE")

                $expectedFailure = $isDimMismatch -or $isUnsupportedFile
                $failureReason = if ($isDimMismatch) { "ModelDimensionMismatch" }
                    elseif ($isUnsupportedFile) { "UnsupportedFile" }
                    else { "Unknown" }

                [pscustomobject]@{
                    Success = $false
                    IsThreat = $false
                    Probability = 0.0
                    ExpectedFailure = $expectedFailure
                    FailureReason = $failureReason
                    Raw = $outputText.Trim()
                }
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

    $managedResults = foreach ($sample in $managedSamples) {
        foreach ($mode in @(
            @{ Name = "Standard"; Flag = "-s" },
            @{ Name = "Flash"; Flag = "-f" },
            @{ Name = "Pro"; Flag = "-p" },
            @{ Name = "Adaptive"; Flag = "-a" }
        )) {
            $result = Invoke-ManagedScan -Sample $sample -ModeName $mode.Name -ModeFlag $mode.Flag
            [pscustomobject]@{
                Sample = Split-Path $sample -Leaf
                Mode = $mode.Name
                Success = $result.Success
                ExpectedFailure = $result.ExpectedFailure
                FailureReason = $result.FailureReason
                IsThreat = $result.IsThreat
                Probability = [Math]::Round($result.Probability, 4)
                Raw = $result.Raw
            }
        }
    }

    $failedManaged = $managedResults | Where-Object { !$_.Success -and !$_.ExpectedFailure -and $_.Sample -eq (Split-Path $SamplePath -Leaf) }
    if ($failedManaged) {
        Write-Warning "Managed scan had unexpected failures for primary sample: $($failedManaged | ForEach-Object { $_.Raw })"
    }

    if ($nativeAvailable) {
        $modes = @(
            @{ Name = "Standard"; NativeMode = 0; Flag = "-s" },
            @{ Name = "Flash"; NativeMode = 1; Flag = "-f" },
            @{ Name = "Pro"; NativeMode = 2; Flag = "-p" },
            @{ Name = "Adaptive"; NativeMode = 3; Flag = "-a" }
        )

        $nativeResults = foreach ($mode in $modes) {
            $managed = $managedResults | Where-Object { $_.Sample -eq (Split-Path $SamplePath -Leaf) -and $_.Mode -eq $mode.Name } | Select-Object -First 1
            if (!$managed -or !$managed.Success) {
                Write-Warning "Skipping Native $($mode.Name) consistency because Managed scan did not succeed."
                continue
            }

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

        $nativeResults | Format-Table -AutoSize
    }
    else {
        Write-Warning "Native DLL not available; skipping Native consistency checks."
    }

    $managedResults | Format-Table -AutoSize

    $unexpectedFailures = $managedResults | Where-Object { !$_.Success -and !$_.ExpectedFailure }
    $expectedFailures = $managedResults | Where-Object { $_.ExpectedFailure }

    $report = [pscustomobject]@{
        Timestamp = [DateTime]::UtcNow.ToString("O")
        NativeAvailable = $nativeAvailable
        ManagedResults = $managedResults
        NativeResults = if ($nativeAvailable) { $nativeResults } else { $null }
        Summary = [pscustomobject]@{
            TotalManagedRuns = $managedResults.Count
            UnexpectedFailures = $unexpectedFailures.Count
            ExpectedFailures = $expectedFailures.Count
            ModelDimensionMismatches = ($expectedFailures | Where-Object { $_.FailureReason -eq "ModelDimensionMismatch" }).Count
        }
    }

    $reportPath = Join-Path (Join-Path $repoRoot "tests") "NativeConsistencyReport.json"
    $report | ConvertTo-Json -Depth 4 | Set-Content -Path $reportPath -Encoding UTF8
    Write-Host "Report written to $reportPath"

    if ($unexpectedFailures) {
        throw "Native consistency smoke had $($unexpectedFailures.Count) unexpected failure(s)."
    }

    Write-Host "Native consistency smoke completed for sample: $SamplePath"
}
finally {
    if (Test-Path $tempDir) {
        Remove-Item -Path $tempDir -Recurse -Force
    }
}
