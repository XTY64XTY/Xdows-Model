# Xdows-Model v2

此版本基于 LightGBM

## Native ONNX Runtime

`Xdows-Model-Native` exposes a stable C ABI for Xdows Security driver protection. The main app loads `Xdows-Model-Native.dll` directly through P/Invoke; it does not start `Xdows-Model-Caller.exe` in the protection path.

Native modes:

- Standard: `Xdows-Model.onnx`
- Flash: `Xdows-Model-Flash.onnx`
- Pro: `Xdows-Model-Pro.onnx`, the five branch models `Xdows-Model-Pro-{Standard,Flash,RawStat,Structural,ImportBehavior}.onnx`, and `Xdows-Model-Pro.manifest.json`

Pro is a five-branch stacking ensemble (Standard / Flash / RawStat / Structural / ImportBehavior). The mixed feature vector is 5143 dims (`299 + 68 + 120 + 32 + 4624`) and the fusion model takes 5 inputs. The manifest records the feature layout hash, each branch's file name / offset / dimension, the ImportBehavior hash algorithm (`Fnv1a32`), seed and bucket sizes, and the fixed / recommended thresholds. The invoker validates it on load and fails loudly on any mismatch. Legacy four-branch models (519-dim hybrid input, 4-input fusion, no manifest) keep loading through the compatibility path.

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

The test scans the same safe PE sample through the managed caller and the native DLL for Standard, Flash, Pro, and Adaptive modes. Verdicts (Clean / Suspicious / Malware) and threat decisions must match, and probability delta must stay within the configured tolerance. Three-tier verdicts map `probability >= fixed threshold` to Malware, `fixed threshold > probability >= recommended threshold` to Suspicious (recommended thresholds come from `<model>.threshold.json` manifests), and lower probabilities to Clean.

Do not commit live malware samples. Safe sample guidance lives in `tests\samples\README.md`.
