# Xdows-Model v2

此版本基于 LightGBM

## Repository Layout

The solution currently contains these projects:

- `Xdows-Model-Config`: shared training paths, thresholds, and model hyperparameters.
- `Xdows-Model-Maker`: data loading, feature extraction, LightGBM training, evaluation, and ONNX export.
- `Xdows-Model-Invoker`: managed ONNX Runtime inference library used by callers and native consistency checks.
- `Xdows-Model-Caller`: command-line sample for scanning one file.
- `Xdows-Model-Evaluator`: evaluation and smoke-test helper.
- `Xdows-Model-Native`: C ABI wrapper used by Xdows Security driver protection.

## Model Modes

The repository ships three ONNX models:

- Standard: `Xdows-Model.onnx`, using the full 299-dimensional managed feature set.
- Flash: `Xdows-Model-Flash.onnx`, using a 68-dimensional fast head/tail feature set.
- Pro: `Xdows-Model-Pro.onnx`, using a hybrid feature vector made from Standard + Flash + PE structure features plus dynamically sized raw section bytes.

Pro inference reads the `Features` input dimension from the ONNX model and derives the raw bytes-per-section value from that dimension. Do not hard-code a fixed Pro feature length when updating Maker, Invoker, or Native behavior.

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
