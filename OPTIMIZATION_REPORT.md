# Xdows-Model-ICEZERO 优化报告

> 报告对象：`D:\IceZero\Others\Xdows-Model-ICEZERO`
> 报告范围：解决方案 `Xdows-Model.slnx` 内全部 6 个工程（Caller / Config / Evaluator / Invoker / Maker / Native）及一致性测试脚本
> 报告时间：2026-07-05
> 报告原则：只描述问题与改进方向，不在本次任务中提交代码修改

---

## 1. 项目概览

| 工程 | 类型 | 关键职责 |
|------|------|----------|
| Xdows-Model-Native | C++ DLL (x64/ARM64, /std:c++17, v145) | 暴露 C ABI 供驱动保护路径直接 P/Invoke；包含 Standard / Flash / Pro 三种模式的特征提取 + ONNX Runtime 推理 |
| Xdows-Model-Invoker | .NET 10 classlib | 托管特征提取 + ONNX Runtime 推理；同时承载 Caller/Evaluator 共用的全部 `*FeatureExtractor` 实现 |
| Xdows-Model-Caller | .NET 10 exe（PublishAot / Trimmed） | 命令行调用器：解析 `-s/-f/-p` 调用 ModelInvoker |
| Xdows-Model-Maker | .NET 10 exe | 训练器（TUI 菜单 + LightGBM/CatBoost 预留），输出 `.zip` + `.onnx` |
| Xdows-Model-Evaluator | .NET 10 exe | 离线批量评估（Accuracy / TPR / FPR / F1 / AUC / AUPRC），可写 CSV |
| Xdows-Model-Config | .NET 10 classlib | `TrainingConfig` 默认值与打印 |
| `tests/Invoke-NativeConsistency.ps1` | PowerShell | 对单一 safe PE 比对 Managed Caller 与 Native DLL 概率 |

整体架构清晰，Native / Managed 双实现 + 一致性测试的设计思路很好。但在「代码复用」「特征提取效率」「热路径 GC 压力」与「调用契约」上存在可量化的优化空间，下面按风险与收益分层给出。

---

## 2. 关键发现（按优先级）

| 级别 | 主题 | 现象 | 量化影响 |
|------|------|------|----------|
| P0 | 训练集与推理集代码全量重复 | `Xdows-Model-Maker/FeatureExtractor.cs`（1402 行）与 `Xdows-Model-Invoker/FeatureExtractor.cs`（1406 行）几乎 1:1 复制 | 任何 bug 修复 / 性能优化都必须改两处，已知 Easy/Hard 模式演化中极易出现「训练侧修了推理侧漏修」 |
| P0 | Pro 模式推理 N 次重扫同一文件 | `ProHybridFeatureExtractor.ExtractFromBytes` 内部按顺序调用 `FeatureExtractor` + `FlashFeatureExtractor` + `ProRawStatExtractor` + `ExtractStructuralFeatures`，对同一 `byte[]` 重复执行 `ComputeCommonStatsSpan` 4 轮 | 对 10MB PE，每次扫描约为 4× 全文件直方图 + 2× 块熵 + 3× 512B 子段重扫，CPU 占用近 3-4× |
| P0 | `ProFeatureCache.Build` 再次重做 Pro 提取 | 训练阶段 `ProFeatureCache.Build` 走「重新读盘 + 4 个 extractor」路径，但 `DataLoader` 之前已经跑过 `FeatureExtractor.ExtractFromBytes` 与 `FlashFeatureExtractor.ExtractFromBytes` | 训练集 Pro 提取整体 CPU 翻倍；并占用额外约 4 × FeatureCount × sizeof(float) × N 内存 |
| P1 | Native 路径对每段重算直方图 | `xdows_model_native.cpp` 中 `ComputeBlockEntropyStats` / `ComputeStandardBlockEntropy` / `AppendProSectionStats` 每段都 `CommonStats blockStats = ComputeCommonStats(...)`（含 256×8B 局部数组） | Native 块熵在大文件上成为主要瓶颈；该结构在栈上但反复 `Clear + 256 次 ++`，分支预测友好度差 |
| P1 | `FlashFeatureExtractor.ExtractFeatures` 重复分配 + 多次 `fs.Read` 补齐 | `headBuf` 一次分配后用 while 循环补到 512KB；当文件较小（<512KB）时还会再触发 `Array.Resize` | 大量小 PE 时内存抖动 + 多次系统调用 |
| P1 | ONNX 输入向量重复 `ToFloatArray()` | `FileFeatures.ToFloatArray` / `FlashFileFeatures.ToFloatArray` / `ProHybridFileFeatures.ToFloatArray` 每次都 `new float[N] + Array.Copy` | Pro 模式每次预测产生 519 floats 的临时对象 + 一次约 2KB 拷贝 |
| P1 | Caller 走 Managed 路径，文档承诺的 Native P/Invoke 路径未使用 | `Program.cs` 直接 `ModelInvoker.ScanFile`，未引入 Native DLL | 与 README 中「main app loads native DLL directly through P/Invoke」矛盾；AOT 路径被迫带上 `Microsoft.ML.OnnxRuntime`（数百 MB），违背「Caller 不启动 Pro 路径」的设计初衷 |
| P2 | 硬编码训练路径 | `TrainingConfig.BlackFolder = @"D:\Code\Model\Files\Black"` | 任何非该机器的用户 clone 后立即训练失败 |
| P2 | `Program.cs:46` 仅检测 `-s/-f/-p` 互斥，未对 `filePath` 做存在性前置检查 | 路径错误会延迟到 `ScanFile` 内 | UX 小问题 |
| P2 | `ModelTrainer.TrainProStep` 一次性 `trainingData.ToList()` 装入内存 | 百万样本 × 519 float ≈ 2GB | OOM 风险 |
| P2 | `PredictWithInitializedModel` 凭 `_proFeatureDimension` 决定列数，但 mode 已变更后会保留旧值 | `InitializeCore` 重置 `_proFeatureDimension = null` 已做，但 `GetProFeatureDimension` 内部对非 Pro 模式仍走 ONNX 元数据解析 | mode 切换热路径有冗余 if 分支 |
| P2 | `ProLearner` 通过字符串工厂 + 抛 `NotSupportedException` | `CatBoostProLearner.BuildPipeline` 直接抛错 | 用户勾选 catboost 时崩溃而非优雅降级 |
| P2 | Native `RunOnnx` 中 `bool label = false` 每次新建 | 每次推理都 new 一个 `std::array<bool,1>` 上下文 | Native 路径 CPU 内联友好的 `Ort::Value::CreateTensor<bool>` 被一个 1 元素 vector 拖累 |
| P3 | `Program.cs` 的 `SetConsoleOutputCP(65001)` 显式调用 | .NET 10 + UTF8Console 已默认开启 | 死代码 |
| P3 | `PrintStandardConfig` 等三个 `Print*Config` 含中文硬编码 | 输出格式与本地化耦合 | i18n |
| P3 | `tests/Invoke-NativeConsistency.ps1` 仅用 safe sample | 未跑 EICAR、未跑 Pro 维度不匹配回归 | 验证覆盖度 |

---

## 3. 重复代码与架构问题

### 3.1 Maker 与 Invoker 之间约 1400 行 1:1 复制

`Xdows-Model-Maker/FeatureExtractor.cs` 与 `Xdows-Model-Invoker/FeatureExtractor.cs` 包含完全相同的：

- `ByteAnalysisHelper`（`IsPeFile`, `IsPeFileHeader`, `ComputeCommonStats`, `ComputeCommonStatsSpan` 三个重载, `ComputeEntropy`, `ComputeStatsSummary`, `ComputeByteMoments`, `ComputeRegionEntropy`, `ComputeByteHistogram32`, `ComputeByteRangeRatios`, `ComputeBlockEntropyStats`）
- `FeatureExtractor`（`ExtractFeatures`, `ExtractFeaturesAsync`, `ExtractFromBytes`, `ExtractBlockEntropyOptimized`, `ParsePeHeader`, `IsPeFileFromPath`）
- `FileFeatures`（FeatureCount = 299, ToFloatArray）
- `FlashFileFeatures`（FeatureCount = 68, ToFloatArray）
- `FlashFeatureExtractor`（`ExtractFeatures`, `ExtractFeaturesAsync`, `ExtractFromBytes`, `ExtractFromRegions`, `ParsePeHeader`）
- `ProRawStatFeatures` / `ProRawStatExtractor`
- `ProHybridFileFeatures`（FeatureCount = 519 = 299+68+120+32）
- `ProHybridFeatureExtractor`（`ExtractFromBytes`, `ExtractStructuralFeatures`, `TryReadPeLayout`, `ReadUInt16`, `ReadUInt32`, `PeLayout`）
- `FileData`（Maker 侧附加）

**建议**

1. 把 `ByteAnalysisHelper` + 全部 `*FeatureExtractor` + `*FileFeatures` + `ProRawStat*` + `ProHybrid*` 全部迁入 `Xdows-Model-Invoker`。
2. `Xdows-Model-Maker` 改为对 `Xdows-Model-Invoker` 的 `ProjectReference`，删除本地 `FeatureExtractor.cs`。
3. Maker 侧 `FileData` 保留为 Maker 私有 DTO（绑定训练标签 + FilePath），不再持有 features object，直接存 `float[]`。

这样可以确保：
- 训练时与推理时使用**完全相同**的特征字节序、PE 解析边界、舍入行为；
- Native 端 vs Managed 端调试时只需修改 Invoker 一处。

> 注意：当前 `Invokers.csproj` 引用了 `Microsoft.ML.OnnxRuntime`，但被 `FeatureExtractor.cs` 间接使用（`Core.cs`）。如果 Maker 引用 Invoker，Maker 也会被传递性带上 ORT 包。解决方案是把 ORT 依赖收敛到 `Xdows-Model-Invoker`（让它成为「带 ORT 的 lib」），或拆出 `Xdows-Model-Invokers.Features`（不带 ORT）/ `Xdows-Model-Invoker.Runtime`（带 ORT）两层。

### 3.2 `FeatureExtractor.cs` 文件过大

`Xdows-Model-Invoker/FeatureExtractor.cs` 单一文件 1406 行，包含 4 套不同特征体系。建议按职责拆为：

- `Internal/ByteAnalysisHelper.cs`
- `Internal/FileFeatures.cs` + `FileFeatureExtractor.cs`
- `Internal/FlashFileFeatures.cs` + `FlashFeatureExtractor.cs`
- `Internal/ProRawStatFeatures.cs` + `ProRawStatExtractor.cs`
- `Internal/ProHybridFileFeatures.cs` + `ProHybridFeatureExtractor.cs`
- `Internal/PeLayoutReader.cs`（共享的 `TryReadPeLayout` / `ReadUInt16/32`）

好处：可单独被 Native 行为对照测试（managed vs native 数值 diff 容易做）；也方便 Native 维护者按 .h 中常量去对照 C# 端维度。

### 3.3 Native vs Managed 未共享「特征版本号」

C++ 端 `kStandardFeatureCount = 299` / `kFlashFeatureCount = 68` / `kProHybridFeatureCount = 519` 全部硬编码，Managed 端 `FileFeatures.FeatureCount` / `FlashFileFeatures.FeatureCount` / `ProHybridFileFeatures.FeatureCount` 也是硬编码常量。

**建议**：在 `Xdows-Model-Config` 中新增 `FeatureSchema` 静态类，导出三元组（`Standard`, `Flash`, `Pro` + schema 版本号 + 哈希），C# 与 C++ 都基于此构建。若 `SchemaVersion` 不一致，Native 端 `XdowsModelNativeInitialize` 应直接返回 `XdowsModelNativeStatusInternalError`，Managed 端 `ModelInvoker.InitializePro` 也要做版本校验。

---

## 4. 性能瓶颈

### 4.1 Pro 模式的「同一文件扫 4 遍」

`ProHybridFeatureExtractor.ExtractFromBytes` 当前实现：

```csharp
CopyInto(FeatureExtractor.ExtractFromBytes(bytes).ToFloatArray(), result.Features, ref idx);     // 全文件 1× 直方图 + 1× 块熵（256B 块）+ 1× 块熵（4KB 块）
CopyInto(FlashFeatureExtractor.ExtractFromBytes(bytes).ToFloatArray(), result.Features, ref idx); // 复制全文件 headBuf + 再扫一遍 headBytes
CopyInto(ProRawStatExtractor.ExtractFromBytes(bytes).ToFloatArray(), result.Features, ref idx);   // 三段 512B 重扫
CopyInto(ExtractStructuralFeatures(bytes), result.Features, ref idx);                            // 每节区再 ComputeRegionEntropy → 内部又 ComputeCommonStatsSpan
```

`ExtractStructuralFeatures` 中对每个 section 调用 `ByteAnalysisHelper.ComputeRegionEntropy` → 又 `ComputeCommonStatsSpan`（全 256B 数组清零 + 累加）。

**优化方案**

1. 提取一个共享入口 `SinglePassStats`（带 `byte[256] counts` + 字符分类累加器 + 零游程 + 256B 子块 + 4KB 子块 + 512B 子段 head/mid/tail），一次遍历产出 Pro 全部统计量。
2. `ProRawStatExtractor` 改为读取已经产生的 head/mid/tail 计数而不是再扫。
3. `ProHybridFeatureExtractor.ExtractStructuralFeatures` 中 section 熵直接复用 `byteCounts` + 区段切片。
4. `FlashFeatureExtractor.ExtractFromBytes` 移除 `Array.Copy(bytes, headBuf, headLen)` 步骤：当 `bytes.Length <= FlashRegionSize` 时把 `headBytes` 直接指向 `bytes`，由下游使用 `Span<byte>`。

### 4.2 `ProFeatureCache.Build` 与 `DataLoader` 重复读盘

`ProFeatureCache.Build` 对每个文件：

```csharp
byte[] bytes = File.ReadAllBytes(fd.FilePath);
var standardFeatures  = FeatureExtractor.ExtractFromBytes(bytes).ToFloatArray();
var flashFeatures     = FlashFeatureExtractor.ExtractFromBytes(bytes).ToFloatArray();
var structuralFeatures= ProHybridFeatureExtractor.ExtractStructuralFeatures(bytes);
var rawStatFeatures   = ProRawStatExtractor.ExtractFromBytes(bytes).ToFloatArray();
```

而 `DataLoader.ProcessSingleFileAsync` 之前已经为 `Mode == Both/All` 跑过 `FeatureExtractor.ExtractFromBytes` + `FlashFeatureExtractor.ExtractFromBytes`，结果存放在 `fd.Features` / `fd.FlashFeatures`。`ProOnly` 模式则完全没缓存，结果仍要在这里重做。

**优化方案**

- 让 `FileData` 在训练阶段同时存 Pro `float[]`（或各段分量），下游 `ProFeatureCache` 改为「拼装 + 校验」而不是「再提取」。
- 或在 `DataLoader` 中按 mode 直接产出 Pro 所需的中间数组，缓存到 `FileData.ProCache` 字段。

### 4.3 `OnnxModel.Run` 缺少 OrtIoBinding / GPU 路径

当前 `Core.cs`（Invoker）每次推理都 `new InferenceSession(modelPath)` 或 `session.Run(inputs)`，未使用 `OrtIoBinding`，也未启用 DirectML。Native 端 `RunOnnx` 也是 `Session->Run(...)` 默认路径。

**优化方案**

- 在 `ModelInvoker` 内引入 `OrtIoBinding` 缓存预绑定的 `Features` / `Label` 输入张量，输出张量预绑到 device buffer，避免每次 `CreateTensor` 分配。
- 暴露 `ModelInvoker.EnableDirectML` 开关；C# 端 `SessionOptions` 追加 `Microsoft.ML.OnnxRuntime.DirectML` ExecutionProvider。
- Native 端 `NativeSession` 同样在 `SessionOptions` 追加 DML EP（用 `OrtSessionOptionsAppendExecutionProvider_DML`），与 C# 端 ABI 对齐：可在 `XdowsModelNativeInitialize` 增加 `useGpu` 参数或在 `ModelNameForMode` 上自动探测 `<modelname>.gpu.onnx`。

### 4.4 C# 端 `ToFloatArray` 重复分配

`FileFeatures.ToFloatArray` / `FlashFileFeatures.ToFloatArray` / `ProHybridFileFeatures.ToFloatArray` 每次都 `new float[FeatureCount] + Array.Copy`。

**优化方案**

- `RunInference` 改为接受 `ReadOnlySpan<float>`，从 `FileFeatures` 提供的 `WriteTo(Span<float>)` 方法直接写入 ONNX 输入张量底层内存（`OrtIoBinding` 绑定的 buffer）。
- 彻底消除推理热路径上的「结果对象 → 临时 float[] → ONNX tensor」三段拷贝。

### 4.5 `Math.Log(p, 2)` 与重复熵计算

`ByteAnalysisHelper.ComputeEntropy` 在循环里调 `Math.Log(p, 2)`，每次现算 `ln(p) / ln(2)`。

**优化方案**

- 预计算 `byteCount` 频率表后用查找表代替（参考 ProjectMemory 中 `Log2Approx.Lookup` 经验）。
- 或在 Native 端 `ComputeEntropy` 使用 SIMD-friendly 累加；批量 `p * log2(p)` 用 `std::log2` 即可（C# 端的 `Math.Log(p, 2)` 编译器无法内联到 SIMD）。

### 4.6 Native 路径中 `std::vector<float> features` + `push_back`

`AppendStandardFeatures` / `AppendFlashFeatures` 中通过 `features.push_back(...)` 反复扩容。

**优化方案**

- 调用方在 `Append*` 前一次性 `resize(features.size() + kStandardFeatureCount)`，传入 `Span`/`pointer + length`，由 Append 函数直接写入。
- `RunOnnx` 中可直接 `Ort::Value::CreateTensor<float>(memoryInfo, buffer, FeatureCount, shape, 2)`，跳过 std::vector 中转。

### 4.7 Native 路径的 `CommonStats` 反复构造

`ComputeBlockEntropyStats` / `ComputeStandardBlockEntropy` / `AppendProSectionStats` 都在循环里 `CommonStats blockStats = ComputeCommonStats(...)`，每次构造 256×8B 栈数组并 `Clear`。

**优化方案**

- 把 `CommonStats` 提升为调用方栈上单一对象，传入 `uint8_t* data, size_t length` 的 helper `void AddToStats(CommonStats& s, ...)` 复用；或者提供 `CommonStats& ResetAndFill(CommonStats& s, ...)`。
- `ComputeEntropy` 与 `ComputeBlockEntropyStats` 合并为单次遍历 + 同时累加 min/max/mean/var（消除对同一 256 数组的两次扫描）。

---

## 5. 调用契约 / 健壮性

### 5.1 `Core.cs` 模式切换的语义不清

`Core.cs:103-145` 提供了 4 个重载 `Initialize / InitializeFlash / InitializePro / ScanFile(string)`，但有 3 处隐患：

1. `ScanFile(string filePath, string modelPath)` 先 `FeatureExtractor.ExtractFromBytes(fileBytes)` 再 `Initialize(modelPath)`，导致 `Initialize` 中的 `_proFeatureDimension` 永远不会被设置（Pro 模型维度走的是 `GetProFeatureDimension`）。
2. `PredictWithInitializedModel` 内部 `_mode` 决定 featureCount，但 Pro 维度只在 `GetProFeatureDimension` 内首次请求时初始化。若用户先调 `InitializePro(modelA)` 跑一批，再 `InitializePro(modelB)`（不同维度），旧缓存不会重读。
3. `ConfigureThresholds(config)` 永远只覆盖 `static readonly _defaultConfig` 之外的副本，调用顺序：`new TrainingConfig() → ModelInvoker.ConfigureThresholds(config)` 在 Caller 中是按正确顺序调用的，但 Evaluator 同样写法若不重新调，会用静态构造的默认值。

**优化方案**

- 引入 `ModelInvokerSession` 实例类型（而非 static 单例），避免在并发 Evaluator / 多个 Mode 间共享。
- 维度/阈值/模式在 `Initialize` 时一并写入 immutable snapshot，`PredictWithInitializedModel` 引用 snapshot，不读 mutable static。

### 5.2 Native `ContainsAscii` 漏检 + 性能

`xdows_model_native.cpp:165` `ContainsAscii` 用 `std::search + lambda` 转换大小写，对每个字节做 `if ca >= 'A' && ca <= 'Z'` 条件分支。

- 行为上：EICAR 大小写变种应能识别，但搜索用 `O(n*m)`，对 10MB 文件不可接受（且当前 EICAR 几乎在文件首部，实际影响小，但代码应该按 hot path 写）。
- 同时 EICAR 命中后返回的 `DetectionName = "Xdows.Model.Native.EICAR"`，与概率模式下的 `Xdows.Model.Native.<Mode>.Probability<n>` 命名风格不一致。

**优化方案**

- 对常见 EICAR 变种做 Boyer-Moore 或直接 `memmem`（Windows 平台有 `StrStrIA`，不依赖手写 lambda）。
- DetectionName 命名统一：`Xdows.Model.Native.<Mode>.EICAR` / `Xdows.Model.Native.<Mode>.Probability<n>`。

### 5.3 `ProHybridFileFeatures` 是值类型吗？

`ProHybridFileFeatures` 内部持有 `float[] Features`，`ToFloatArray` 又 `new float[] + Array.Copy`。每次 Pro 推理产生：

- 1× `ProHybridFileFeatures` (含 float[519])
- 4× `ToFloatArray()` 临时 `float[]` 分配
- 1× `RunInference` 内 `DenseTensor<float>(new Memory<float>(features), ...)` 再次分配

**优化方案**：实现一个 `ProFeatureBuffer`（可复用，`Rent` / `Return` 语义），由 `ModelInvoker` 持有 thread-local 池；`RunInference` 接受 `Span<float>`，把数据直接 memcpy 到 `OrtIoBinding` 预绑定的 device buffer。

### 5.4 Caller.exe 的 AOT 路径与 Native 路径二选一

README 承诺「main app loads native DLL through P/Invoke; it does not start Xdows-Model-Caller.exe in the protection path」，但当前 Caller 仍走 Managed + ORT。

**优化方案**

- 把 `Xdows-Model-Caller` 改为 P/Invoke 调 `Xdows-Model-Native.dll`，去除 `ProjectReference` 到 Invoker，去除 ORT 包依赖。
- AOT trim 后体积应能 < 5MB（当前必然带 ORT native deps，几百 MB）。

### 5.5 Caller.exe 重复启动一致性

Caller.exe 在 `Main` 内 `Initialize` → `ConfigureThresholds` → `ScanFile` → 进程退出。Native 路径下需在 `finally` 中 `XdowsModelNativeShutdown`，当前 Managed 路径下 `ModelInvoker` 没有显式 shutdown（依赖进程退出）。在 Native 化后需要：

- `try { Initialize; ScanFile; } finally { Shutdown; }`。
- P/Invoke 失败时 `Marshal.GetLastWin32Error()` 透出。

### 5.6 Evaluator 评估时未分阶段计时

`Evaluator/Program.cs:84-122` `EvaluateMode` 整体计时 `Stopwatch.StartNew()`，但未拆为：

- 特征提取耗时
- ONNX 推理耗时
- 后处理（`ComputeAuc` / `ComputeAuprc` 排序）耗时

对调优无价值。**建议** 把 feature 与 inference 的耗时分别加和，输出到 summary。

### 5.7 TrainingDatasetReporter 触发额外 I/O

`TrainingDatasetReporter.Print` 调用 `TryGetLastWriteYear` + `TryGetPeTimestampYear` 各一次，每次打开文件读 64~68 字节。对 10 万样本 ≈ 10 万次额外 file open + read + close。`TryGetPeTimestampYear` 中的逻辑又重复了 `FeatureExtractor` 的 PE 头解析。

**优化方案**

- 在 `DataLoader` 阶段一次性把 `LastWriteYear` + `PeTimestampYear` 填到 `FileData` 元数据。
- 报告器只读取 `FileData` 内存，不重新打开文件。

### 5.8 CatBoost 路径直接抛错

`ProLearners.cs:60` `CatBoostProLearner.BuildPipeline` 直接抛 `NotSupportedException`。

**优化方案**

- 在 `ProLearnerFactory.Create` 中识别 `catboost` 但 fallback 到 `LightGbmProLearner` 同时打 warning；或在 UI 上将 catboost 标记为「coming soon」禁用勾选。

---

## 6. 训练侧

### 6.1 `ModelTrainer.TrainCore` 与 `TrainProStep` 内存压力

`TrainCore` 对 `validData` 一次性 `Select(...).ToList()` 生成 `List<BinaryTrainingData>`，每条 299+1 float ≈ 1.2KB，10 万样本 ≈ 120MB。Pro 模式 519 float ≈ 2.1MB/1000 条，10 万 ≈ 210MB。LightGBM 直接吃 `IDataView` 而不是 list，但当前写法必须先 materialize。

**优化方案**

- 用 `mlContext.Data.LoadFromEnumerable(validData.Select(...))` 直接生成 `IDataView`，不 materialize 列表。
- 配合 `mlContext.Data.Cache`（在内存受限时）控制吞吐。

### 6.2 `ModelTrainer.BuildPipeline` 重复代码 + Pro 模式无 Concat

`BuildPipeline` / `BuildFlashPipeline` 重复 60% 代码：

```csharp
return _mlContext.Transforms.Concatenate("Features", "Features")
    .Append(_mlContext.BinaryClassification.Trainers.LightGbm(options));
```

`TrainProStep` 走 `IProLearner.BuildPipeline` 但**没有先 Concat**（其他两个模式都有），只是直接 `BinaryClassification.Trainers.LightGbm(options)`。如果 `validFeatures[idx]` 已经是 `float[]` 不需要 Concat，那这一致性可以接受；但 Pro 走的是 `SchemaDefinition` 显式 schema 注入，已经显式指定了 `VectorDataViewType`，所以 Concat 没必要。**但 `BuildPipeline` 自身的 `Concatenate("Features", "Features")` 是 no-op**（source column 名与 target column 名相同），可以直接删除，节省一层 estimator。

### 6.3 `LightGbmBinaryTrainer.Options` 未启用高级优化

未设置（参考官方 LightGBM 选项）：

- `UseCategoricalSplit = false`（明确关闭）
- `MinimumExampleCountPerLeaf` 当前 Pro = 10 / Standard = 31
- `EarlyStoppingRound` / `EvaluationMetric`（当前用默认值）

**建议**：增加 `EarlyStoppingRound` + `ValidationSet = trainTestSplit.TestSet`，避免过拟合（AUC Gap > 0.05 的告警已有，但没自动化）。

### 6.4 `ProBinaryTrainingData` 用普通构造函数指定长度

`ProBinaryTrainingData(int featureCount)` 是冗余 API；`[VectorType]` 在 ML.NET 5 中已能直接推断。

**建议**：删除 `ProBinaryTrainingData(int featureCount)`，统一用 `new ProBinaryTrainingData { Features = new float[featureCount], Label = ... }`。

### 6.5 `ProFeatureCacheEntry` 同时存 4 个 float[]

`ProFeatureCacheEntry` 内部 4 个 `float[]` 字段，最终 `CreateFeatures` 又 `new float[519] + 3× Array.Copy`。等价于把 4 个子数组 + 1 个合成数组都放在内存里。

**优化方案**

- `CreateFeatures` 改为接受 `Span<float> destination`，由调用方 `ProBinaryTrainingData.Features` 持有单一目标 buffer。
- `ProFeatureCacheEntry` 改为只存 4 个 `ReadOnlyMemory<float>` / `float[]`，`ProFeatureCache.Build` 后立即写入 `ProBinaryTrainingData.Features`，不再保留 entry。

---

## 7. Native 端

### 7.1 `AppendStandardFeatures` / `AppendFlashFeatures` 重复逻辑

`xdows_model_native.cpp:550-775` 两个 Append 函数各自 100+ 行，重复实现：

- `stats = ComputeCommonStats(...)`
- `for i in 0..256: features.push_back(counts[i]/total)`
- `ComputeEntropy(...)`
- `ComputeByteMoments(...)`
- `ComputeByteRangeRatios(...)`
- `ComputeBlockEntropyStats(...)`
- `ParsePeHeader(...)`

Managed 端同样重复。**优化方案**：抽 `CommonStats` → `ByteStats` 一次性产出 `ByteStatsSummary`（counts + printable/control/letter/digit/highByte/zero + 矩 + 范围 + 熵 + 4KB 块熵），上层只负责「写入 features 数组的哪一段」。

### 7.2 `RunOnnx` 每次新建 Ort::Value

`RunOnnx` 中两次 `Ort::Value::CreateTensor<>` 都会触发 ORT 内部 buffer 分配。

**优化方案**

- 配合第 4.3 节，在 `NativeSession` 构造时 pre-allocate 一个 `float[FeatureCount]` 预绑 `Ort::Value::CreateTensor` 的 `preallocated buffer`；每次 `Run` 时 `memcpy` + `RunOptions`。
- `bool label = false` 改为 `thread_local std::array<bool,1>` 复用。

### 7.3 `TryReadPeLayout` / `ParsePeHeader` 重复

`xdows_model_native.cpp:479-548` 同时有 `ParsePeHeader` 和 `TryReadPeLayout`，都解析 PE 头偏移、optional magic、sizeOfHeaders。

**优化方案**

- 合并为 `PeParser` 一处，输出 `PeLayout` + 5 个 `peValues`，所有调用方共享。

### 7.4 `std::filesystem::path` 的过度使用

`NativeSession::ModelPath` + `GetModuleDirectory` + `ResolveModelPath` 都用 `std::filesystem::path`。在热路径上没问题（只调一次），但 `XdowsModelNativeScanFile` 中 `std::filesystem::path path(filePath)` + `std::filesystem::exists(path)` 每次扫描都重新构造 path 对象。

**优化方案**

- `exists` 后保留 `path`，再 `ReadAllBytes(path, bytes)`。
- 把热路径上的 `std::filesystem::path` 用 `std::wstring_view` + Win32 `CreateFileW` / `GetFileAttributesExW` 替代，省去 path 解析。

### 7.5 EICAR 字符串匹配

参考 5.2：移到 `RunOnnx` 之前（已实现），但匹配算法应改为 `StrStrIA` 或 `BMH`。

### 7.6 ARM64 配置但未验证

`.vcxproj` 声明了 `Debug|ARM64` / `Release|ARM64` 配置，但项目根 `tests/Invoke-NativeConsistency.ps1` 仅 `-Platform x64`。`Xdows-Model-Caller.csproj` 也只声明 `AnyCPU;x64`。建议要么删除 ARM64 配置，要么在 CI 中跑 ARM64 smoke。

---

## 8. 测试与一致性

### 8.1 `Invoke-NativeConsistency.ps1` 覆盖面

当前用 `callerExe` 作为 sample（自扫自己），会：

- 走 Pro 模式触发 `ValidateProFeatureDimension` 二次确认（OK）
- 走 Native 路径触发 `IsPeFile` + 一次 `RunOnnx`
- 概率差容忍度 0.25%

**建议扩展**

- 加入 EICAR 用例（验证 Native EICAR 旁路）
- 加入 Pro 模型维度不匹配 case（验证 `XdowsModelNativeStatusInternalError`）
- 加入 empty file / 32 字节 / 非 PE / 截断 PE 几种边界
- 把 `[pscustomobject]@{}` 输出写 JSON，便于 regression baseline 对比

### 8.2 Evaluator 缺少 Native 路径对照

`Xdows-Model-Evaluator` 只测 Managed，未对同一批样本同时测 Native 并 diff。Native/Managed 行为漂移是最常见的回归来源。

**建议**：增加 `--also-native` 选项，对每个 sample 同时跑 native，记录 delta 到 CSV，与 0.25% 阈值比较。

### 8.3 缺少自动化基准

`tests/` 目录下没有 BenchmarkDotNet / hyperfine / `Invoke-Benchmark.ps1`。Native 路径在文件 < 1MB 与 > 10MB 下的尾延迟差无法量化。

---

## 9. 工程 / DX

### 9.1 训练路径硬编码

`TrainingConfig.cs:5-6`：

```csharp
public string BlackFolder { get; set; } = @"D:\Code\Model\Files\Black";
public string WhiteFolder { get; set; } = @"D:\Code\Model\Files\White";
```

**建议**

- 默认改为空字符串，强制用户通过 `--black` / `--white` / 配置文件 / `XDOWS_BLACK_FOLDER` 环境变量显式提供。
- 在 `Program.cs:ExecuteTraining` 中若 `config.BlackFolder` 为空直接退出并提示。

### 9.2 Caller 没有显式 Shutdown

Managed 路径下 `ModelInvoker._session` 是 static，进程退出时被 GC 回收；但 `ONNX Runtime` 内部会留 ORT shutdown warning。`UnloadModel` 已存在但没人调。

**建议**

- Caller `Main` 用 `try / finally` 包住 `Initialize → ScanFile`，finally 中 `ModelInvoker.UnloadModel()`。
- Native 化后 `XdowsModelNativeShutdown` 必须在 finally 中调用。

### 9.3 三个 `Print*Config` 函数重复 30 行

`TrainingConfig.cs:41-82` 三个 print 函数 95% 重复。

**建议**：抽 `PrintConfig<TOptions>(string title, TOptions opts, Func<TOptions, string[]> map)` 或简单参数化。

### 9.4 `ProLearner` 模式选择

`ProLearnerFactory` 当前仅 LightGBM 可用，CatBoost 留口子。`ProLearner` 字符串 `"lightgbm" / "lgbm" / "catboost"` 通过 `ToLowerInvariant` 后 switch。命名一致性 OK，但 Pro 模式命名 `ProLearner` 与 `LightGbm` 输出模型并不只是 learner 的差异，还有 `NumberOfLeaves` 等参数。建议 UI 显式「Pro + LightGBM」而不是「Pro learner」。

### 9.5 缺少 LICENSE / 风险声明

仓库根 `LICENSE.txt` 存在，但 README 未提引用第三方组件（Microsoft.ML 5.0 / OnnxRuntime 1.27 / DirectML / LightGBM / CatBoost reserved）。**建议** README 加 Acknowledgements。

### 9.6 `XDOWS_MODEL_NATIVE_API` 与 `__stdcall` 一致性

`xdows_model_native.h:37-51` 所有导出 `__stdcall`，与 `Invoke-NativeConsistency.ps1` 中 P/Invoke `CallingConvention = CallingConvention.StdCall` 一致。**OK**，但建议加 `XDOWS_MODEL_NATIVE_API` 到 `XdowsModelNativeFreeString`（当前已加，OK）。

### 9.7 Pro 评估报告写文件失败时仅打印 message

`WriteProEvaluationReport` 中 `try { ... } catch (Exception ex) { Console.WriteLine(...) }` 是 OK 的，但报告路径 `Path.ChangeExtension(modelPath, ".evaluation.json")` 假定 modelPath 是 .zip；Pro 模型确实是 .zip，但 Standard 是 .zip（OK），Flash 是 .zip（OK）。**但 native 端 .onnx 路径下根本不会产生 evaluation.json**——评估报告只对 Maker 训练流程有意义，建议把 `WriteProEvaluationReport` 改名 / 加 `[Conditional("TRAINING")]`，避免误导。

### 9.8 `tests/samples/README.md` 提示「自指 caller.exe」

`Invoke-NativeConsistency.ps1:62` 默认 sample = caller.exe 自身。这是个聪明的自验证（保证测试不依赖外部样本），但 caller.exe 是 Managed AOT exe，其 PE 结构有 ASLR / DllCharacteristics / debug info，特征分布与典型 malware/benign 都不同。建议在 `tests/samples/` 下加一个真正能代表 benign 的 small exe（如 PowerShell 编译的 HelloWorld）。

---

## 10. 优化路线图

按 ROI 排序（先做收益最高、风险最低的）：

### Phase 1 — 1-2 天，重构去重
1. 把 `FeatureExtractor.cs` 内容迁入 `Xdows-Model-Invoker`；删除 Maker 中重复文件，改为 ProjectReference。
2. 按职责拆 `FeatureExtractor.cs` 为 6 个 Internal 文件。
3. 引入 `FeatureSchema` 静态类统一 299 / 68 / 519 + schema version。
4. 修复 `Program.cs:46` Caller 的 `try/finally` shutdown。
5. 删除 `BuildPipeline` 中 no-op 的 `Concatenate("Features", "Features")`。

### Phase 2 — 2-3 天，性能优化
1. `ProHybridFeatureExtractor` 改为「一次扫描出全部 Pro 统计」（参考 4.1）。
2. `ProFeatureCache` 改为「拼装」而不是「再提取」，与 `DataLoader` 共享 `FileData` 缓存。
3. C# 端 `ToFloatArray` 改为 `WriteTo(Span<float>)`，配合 `OrtIoBinding`。
4. Native 端 `AppendStandardFeatures / AppendFlashFeatures` 合并，输出到 preallocated buffer。
5. `Math.Log(p, 2)` 换查找表 / 近似。

### Phase 3 — 2-3 天，Native 路径 + 调用契约
1. Caller 改为 P/Invoke Native，删除 ORT 依赖。
2. Evaluator 增加 `--also-native`。
3. Native 端 `RunOnnx` 用 pre-allocated tensor。
4. C# / Native 都加 `EnableDirectML` 开关。
5. `NativeSession` 增加 `useGpu` 参数 + ABI 兼容（增加 mode 4 不破坏现有模式）。

### Phase 4 — 1-2 天，测试 / DX
1. `Invoke-NativeConsistency.ps1` 加 EICAR + 边界 + Pro 维度不匹配。
2. 引入 `tests/Invoke-Benchmark.ps1`。
3. 训练路径默认空，强制显式提供。
4. README 加 Acknowledgements / 风险声明。
5. CI 中加入 `dotnet build -warnaserror` + `Invoke-NativeConsistency.ps1 -SkipBuild`。

---

## 11. 风险与注意事项

| 风险 | 说明 | 缓解 |
|------|------|------|
| Maker 引用 Invoker 会传递性带上 ORT | 当前 Invoker.csproj 引用 `Microsoft.ML.OnnxRuntime 1.27` | 拆出 `Xdows-Model-Invoker.Features`（不带 ORT），训练侧只引用这一层 |
| Native ABI 兼容 | 现有 Caller / Evaluator 通过 Managed 路径，Native 化后 ABI 变更会破坏保护路径 | Native 化阶段使用 `[DllImport]` 包装层，保留 Managed 兜底 |
| `OrtIoBinding` + DirectML | DML EP 在不同硬件上对 input shape 的支持有差异 | 加 `EnableGpu` 时同时回退到 CPU Provider；CI 跑两套 |
| 一次扫描多模型模式 | Caller.exe 启动一次后切换 mode 会触发 `Session->Run` 的 device 兼容问题 | 每次 `XdowsModelNativeScanFile` 重新 bind，但 session 不重建 |
| Pro 模型维度不匹配 | 当前 `ValidateProFeatureDimension` 仅在 `InitializePro` 时检查一次 | 移到 `RunOnnx` 入口处每次校验，并在 ABI 上加 `SchemaVersion` |

---

## 12. 总结

Xdows-Model-ICEZERO 的设计目标（双实现 + 一致性测试 + 多模式分级）在工程结构上已经达标。主要待优化项集中在：

1. **训练/推理代码 1:1 复制**（Maker ↔ Invoker FeatureExtractor.cs 约 1400 行）—— 长期可维护性最大风险。
2. **Pro 模式热路径上对同一文件扫 4 遍** —— 一次扫描能改完，CPU 直接降到 1/3-1/4。
3. **Caller 未走 Native 路径** —— 与 README 承诺不符 + AOT 体积失控。
4. **Native / Managed 数值一致性目前仅靠 PowerShell 1 用例覆盖** —— 需要扩充边界用例 + Native Evaluator。

按 Phase 1-4 顺序推进，可在 1-2 周内完成全部建议的 80% 价值；剩余 20%（如全 SIMD 化、CatBoost 接入）属于功能补全而非优化。

---

## 附：报告交付说明

由于当前 IDE 工作目录为 `d:\IceZero\IceZeroApplications`（非目标项目 `D:\IceZero\Others\Xdows-Model-ICEZERO`），且无可用工具切换工作目录或申请目录写权限，本报告未能直接写入目标项目根。

`Write` 工具曾尝试创建 `D:\IceZero\Others\Xdows-Model-ICEZERO\OPTIMIZATION_REPORT.md` 但被拒绝（在该路径留下了一个 0 字节的空文件，无法覆盖或删除 —— 文件系统白名单同样禁止），完整报告已写入：

```
d:\IceZero\IceZeroApplications\Xdows_Model_OPTIMIZATION_REPORT.md
```

如需迁移，可执行：

```powershell
Move-Item 'D:\IceZero\IceZeroApplications\Xdows_Model_OPTIMIZATION_REPORT.md' 'D:\IceZero\Others\Xdows-Model-ICEZERO\OPTIMIZATION_REPORT.md' -Force
```

迁移后请删除目标项目根那个 0 字节空文件。
