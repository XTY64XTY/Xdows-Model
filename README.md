# Xdows-Model v2

此版本基于 LightGBM

## Native ONNX Runtime

`Xdows-Model-Native` exposes a stable C ABI for Xdows Security driver protection. The main app loads `Xdows-Model-Native.dll` directly through P/Invoke; it does not start `Xdows-Model-Caller.exe` in the protection path.

Native modes:

- Standard: `Xdows-Model.onnx`
- Flash: `Xdows-Model-Flash.onnx`
- Pro: `Xdows-Model-Pro.onnx`

Build:

```powershell
& 'D:\Visual-Studio\MSBuild\Current\Bin\amd64\MSBuild.exe' `
  'D:\Code\Xdows-Model\Xdows-Model.slnx' `
  /p:Configuration=Debug `
  /p:Platform=x64 `
  /m
```

Expected native output:

```text
D:\Code\Xdows-Model\x64\Debug\Xdows-Model-Native.dll
D:\Code\Xdows-Model\x64\Debug\onnxruntime.dll
D:\Code\Xdows-Model\x64\Debug\onnxruntime_providers_shared.dll
```

Consistency test:

```powershell
& 'D:\Code\Xdows-Model\tests\Invoke-NativeConsistency.ps1' -SkipBuild
```

The test scans the same safe PE sample through the managed caller and the native DLL for Standard, Flash, and Pro modes. Threat decisions must match and probability delta must stay within the configured tolerance.

Do not commit live malware samples. Safe sample guidance lives in `tests\samples\README.md`.
