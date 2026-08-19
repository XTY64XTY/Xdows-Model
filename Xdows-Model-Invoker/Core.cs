using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Reflection;
using Xdows_Model_Config;

namespace Xdows_Model_Invoker
{
    public enum ModelMode
    {
        Standard = 0,
        Flash = 1,
        Pro = 2,
        Adaptive = 3
    }

    public static class ModelInvoker
    {
        private const string DefaultModelFileName = "Xdows-Model.onnx";
        private const string DefaultFlashModelFileName = "Xdows-Model-Flash.onnx";
        private const string DefaultProModelFileName = "Xdows-Model-Pro.onnx";
        private static readonly TrainingConfig _defaultConfig = new();
        private static readonly object _initLock = new();
        private static InferenceSession? _session;
        private static string? _loadedModelPath;
        private static ModelMode _mode = ModelMode.Standard;
        private static int? _proFeatureDimension;
        private static ProEnsembleSession? _proEnsemble;
        private static AdaptiveModelSession? _adaptiveSession;
        private static float _standardThreshold = NormalizeThreshold((float)_defaultConfig.StandardThreshold);
        private static float _flashThreshold = NormalizeThreshold((float)_defaultConfig.FlashThreshold);
        private static float _proThreshold = NormalizeThreshold((float)_defaultConfig.ProThreshold);

        // 推荐阈值与固定阈值分开存储：固定阈值来自 TrainingConfig（判毒下限），
        // 推荐阈值来自模型旁 *.threshold.json 清单（Suspicious 区间下限）。清单缺失时回退到固定值。
        private static float _standardRecommendedThreshold = NormalizeThreshold((float)_defaultConfig.StandardThreshold);
        private static float _flashRecommendedThreshold = NormalizeThreshold((float)_defaultConfig.FlashThreshold);
        private static float _proRecommendedThreshold = NormalizeThreshold((float)_defaultConfig.ProThreshold);

        /// <summary>
        /// 是否在加载模型时自动采用模型旁 <c>*.threshold.json</c> 里训练阶段校准出的推荐阈值。
        /// 清单缺失或无效时回退到 <see cref="TrainingConfig"/> 中的固定阈值。
        /// </summary>
        public static bool AutoThresholdSelection { get; set; } = true;

        private static string EnsureModelAvailable(string fileName)
        {
            string baseDir = AppContext.BaseDirectory;
            string destPath = Path.Combine(baseDir, fileName);

            if (File.Exists(destPath))
                return destPath;

            var asm = Assembly.GetExecutingAssembly();

            var resourceName = asm.GetManifestResourceNames().FirstOrDefault(n => n.EndsWith(fileName, StringComparison.OrdinalIgnoreCase));
            if (!string.IsNullOrEmpty(resourceName))
            {
                using var rs = asm.GetManifestResourceStream(resourceName);
                if (rs != null)
                {
                    using var fs = new FileStream(destPath, FileMode.Create, FileAccess.Write);
                    rs.CopyTo(fs);
                    return destPath;
                }
            }
            var asmDir = Path.GetDirectoryName(asm.Location) ?? baseDir;
            var candidate = Path.Combine(asmDir, fileName);
            if (File.Exists(candidate))
            {
                File.Copy(candidate, destPath, overwrite: true);
                return destPath;
            }

            throw new FileNotFoundException($"Model file not found. Expected to find '{fileName}' as an embedded resource or next to the Invoker assembly.", fileName);
        }

        public static (ScanVerdict verdict, float probability) PredictWithMlNet(string modelPath, float[] features)
        {
            using var session = new InferenceSession(modelPath, CreateSessionOptions());
            return RunInference(session, features, FileFeatures.FeatureCount, _standardThreshold, _standardRecommendedThreshold);
        }

        public static (ScanVerdict verdict, float probability) ScanFile(string filePath, string modelPath)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException("找不到指定文件", filePath);

            Initialize(modelPath);

            var fileBytes = File.ReadAllBytes(filePath);
            if (!FeatureExtractor.IsPeFile(fileBytes))
                throw new NotSupportedException("不支持该文件类型");

            var features = FeatureExtractor.ExtractFromBytes(fileBytes);
            return PredictWithInitializedModel(features.ToFloatArray());
        }

        public static (ScanVerdict verdict, float probability) ScanFileFlash(string filePath, string modelPath)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException("找不到指定文件", filePath);

            InitializeFlash(modelPath);

            byte[] bytes = File.ReadAllBytes(filePath);
            if (!FeatureExtractor.IsPeFile(bytes))
                throw new NotSupportedException("不支持该文件类型");
            var features = FlashFeatureExtractor.ExtractFromBytes(bytes);
            return PredictWithInitializedModel(features.ToFloatArray());
        }

        public static (ScanVerdict verdict, float probability) ScanFilePro(string filePath, string modelPath)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException("找不到指定文件", filePath);

            InitializePro(modelPath);

            byte[] bytes = File.ReadAllBytes(filePath);
            if (!FeatureExtractor.IsPeFile(bytes))
                throw new NotSupportedException("不支持该文件类型");
            var proFeatures = ProHybridFeatureExtractor.ExtractFromBytes(bytes);
            return PredictWithInitializedModel(proFeatures.ToFloatArray());
        }

        public static void Initialize(string? modelPath = null)
        {
            InitializeCore(modelPath ?? EnsureModelAvailable(DefaultModelFileName), ModelMode.Standard);
        }

        public static void InitializeFlash(string? modelPath = null)
        {
            InitializeCore(modelPath ?? EnsureModelAvailable(DefaultFlashModelFileName), ModelMode.Flash);
        }

        public static void InitializePro(string? modelPath = null)
        {
            try
            {
                InitializeCore(modelPath ?? EnsureModelAvailable(DefaultProModelFileName), ModelMode.Pro);
            }
            catch
            {
                UnloadModel();
                throw;
            }
        }

        public static void InitializeAdaptive(string? modelDirectory = null)
        {
            string directoryKey = string.IsNullOrWhiteSpace(modelDirectory)
                ? string.Empty
                : Path.GetFullPath(modelDirectory);

            lock (_initLock)
            {
                if (_adaptiveSession != null &&
                    _mode == ModelMode.Adaptive &&
                    string.Equals(_loadedModelPath, directoryKey, StringComparison.OrdinalIgnoreCase))
                {
                    return;
                }
            }

            try
            {
                AdaptiveModelSession adaptiveSession = CreateAdaptiveSession(
                    directoryKey.Length == 0 ? null : directoryKey,
                    _defaultConfig);

                lock (_initLock)
                {
                    _session?.Dispose();
                    _proEnsemble?.Dispose();
                    _adaptiveSession?.Dispose();
                    _session = null;
                    _proEnsemble = null;
                    _adaptiveSession = adaptiveSession;
                    _loadedModelPath = directoryKey;
                    _mode = ModelMode.Adaptive;
                    _proFeatureDimension = null;
                }
            }
            catch
            {
                UnloadModel();
                throw;
            }
        }

        private static void InitializeCore(string path, ModelMode mode)
        {
            lock (_initLock)
            {
                if (_session != null && _loadedModelPath == path && _mode == mode)
                    return;

                _session?.Dispose();
                _proEnsemble?.Dispose();
                _adaptiveSession?.Dispose();
                _proEnsemble = null;
                _adaptiveSession = null;
                _session = new InferenceSession(path, CreateSessionOptions());
                _loadedModelPath = path;
                _mode = mode;
                _proFeatureDimension = null;

                ValidateFeatureDimension(mode);
                ApplyThresholdManifest(path, mode);
            }
        }

        private static void ValidateFeatureDimension(ModelMode mode)
        {
            if (_session == null)
                throw new InvalidOperationException("ModelInvoker 没有初始化");

            int expected = mode switch
            {
                ModelMode.Flash => FeatureSchema.FlashFeatureCount,
                ModelMode.Pro => FeatureSchema.ProHybridFeatureCount,
                _ => FeatureSchema.StandardFeatureCount
            };

            if (mode == ModelMode.Pro)
            {
                ValidateProFeatureDimension();
                return;
            }

            var inputMeta = _session.InputMetadata;
            if (inputMeta.TryGetValue("Features", out var nodeMeta))
            {
                var dims = nodeMeta.Dimensions;
                int actual = dims.Length == 2 && dims[1] > 0 ? dims[1]
                    : dims.Length == 1 && dims[0] > 0 ? dims[0]
                    : -1;

                bool validProDimension = mode == ModelMode.Pro && actual == FeatureSchema.ProFusionFeatureCount;
                if (actual > 0 && actual != expected && !validProDimension)
                {
                    throw new InvalidOperationException(
                        $"{mode} 模型特征维度不匹配：当前模型为 {actual} 维，期望 {expected} 维。");
                }
            }
        }

        public static bool IsInitialized => _session != null || _adaptiveSession != null;
        public static bool IsFlashMode => _mode == ModelMode.Flash;
        public static bool IsProMode => _mode == ModelMode.Pro;
        public static bool IsAdaptiveMode => _mode == ModelMode.Adaptive;
        public static ModelMode CurrentMode => _mode;

        public static void ConfigureThresholds(TrainingConfig config)
        {
            ArgumentNullException.ThrowIfNull(config);

            float standardThreshold = NormalizeThreshold((float)config.StandardThreshold);
            float flashThreshold = NormalizeThreshold((float)config.FlashThreshold);
            float proThreshold = NormalizeThreshold((float)config.ProThreshold);

            _standardThreshold = standardThreshold;
            _flashThreshold = flashThreshold;
            _proThreshold = proThreshold;
            _defaultConfig.StandardThreshold = standardThreshold;
            _defaultConfig.FlashThreshold = flashThreshold;
            _defaultConfig.ProThreshold = proThreshold;

            // 未加载清单时推荐阈值与固定阈值一致（Suspicious 区间为空，退化为二档判定）。
            _standardRecommendedThreshold = standardThreshold;
            _flashRecommendedThreshold = flashThreshold;
            _proRecommendedThreshold = proThreshold;
        }

        /// <summary>
        /// 三档判定：输出 >= 固定阈值 → <see cref="ScanVerdict.Malware"/>；
        /// 固定阈值 > 输出 >= 推荐阈值 → <see cref="ScanVerdict.Suspicious"/>；否则 <see cref="ScanVerdict.Clean"/>。
        /// 先比较固定阈值再比较推荐阈值，即使清单推荐值异常高于固定阈值也能保持正确顺序。
        /// </summary>
        public static ScanVerdict ClassifyVerdict(float probability, float fixedThreshold, float recommendedThreshold)
        {
            if (probability >= fixedThreshold)
                return ScanVerdict.Malware;
            if (probability >= recommendedThreshold)
                return ScanVerdict.Suspicious;
            return ScanVerdict.Clean;
        }

        /// <summary>
        /// 按模型旁的阈值清单自动设置该模式的推荐判毒阈值。清单不存在或无效时推荐阈值回退到固定阈值，
        /// 并在原因明确时输出诊断信息，避免线上工作点在无声中偏离预期。
        /// 固定阈值（判毒下限）永远不被清单覆盖。
        /// </summary>
        private static void ApplyThresholdManifest(string modelPath, ModelMode mode)
        {
            if (!AutoThresholdSelection)
                return;

            if (!ModelThresholdManifest.TryLoad(modelPath, mode.ToString(), out ModelThresholdManifest? manifest, out string? failureReason))
            {
                if (!string.IsNullOrEmpty(failureReason))
                    Console.Error.WriteLine($"[Xdows-Model] {mode} 阈值清单被忽略：{failureReason}");
                return;
            }

            float threshold = NormalizeThreshold((float)manifest!.RecommendedThreshold);
            SetRecommendedThreshold(mode, threshold);
        }

        private static void SetThreshold(ModelMode mode, float threshold)
        {
            switch (mode)
            {
                case ModelMode.Flash:
                    _flashThreshold = threshold;
                    _defaultConfig.FlashThreshold = threshold;
                    break;
                case ModelMode.Pro:
                    _proThreshold = threshold;
                    _defaultConfig.ProThreshold = threshold;
                    break;
                default:
                    _standardThreshold = threshold;
                    _defaultConfig.StandardThreshold = threshold;
                    break;
            }
        }

        private static void SetRecommendedThreshold(ModelMode mode, float threshold)
        {
            switch (mode)
            {
                case ModelMode.Flash:
                    _flashRecommendedThreshold = threshold;
                    break;
                case ModelMode.Pro:
                    _proRecommendedThreshold = threshold;
                    break;
                default:
                    _standardRecommendedThreshold = threshold;
                    break;
            }
        }

        public static float GetThreshold(ModelMode mode)
        {
            return mode switch
            {
                ModelMode.Flash => _flashThreshold,
                ModelMode.Pro => _proThreshold,
                ModelMode.Adaptive => _proThreshold,
                _ => _standardThreshold
            };
        }

        /// <summary>
        /// 返回当前模式的推荐阈值（Suspicious 区间下限）。清单缺失时与固定阈值一致。
        /// </summary>
        public static float GetRecommendedThreshold(ModelMode mode)
        {
            return mode switch
            {
                ModelMode.Flash => _flashRecommendedThreshold,
                ModelMode.Pro => _proRecommendedThreshold,
                ModelMode.Adaptive => _proRecommendedThreshold,
                _ => _standardRecommendedThreshold
            };
        }

        public static void UnloadModel()
        {
            lock (_initLock)
            {
                _session?.Dispose();
                _proEnsemble?.Dispose();
                _adaptiveSession?.Dispose();
                _proEnsemble = null;
                _adaptiveSession = null;
                _session = null;
                _loadedModelPath = null;
                _mode = ModelMode.Standard;
                _proFeatureDimension = null;
            }
        }

        private static int GetProFeatureDimension()
        {
            if (_proFeatureDimension.HasValue)
                return _proFeatureDimension.Value;

            if (_session == null)
                throw new InvalidOperationException("ModelInvoker 没有初始化");

            var inputMeta = _session.InputMetadata;
            if (inputMeta.TryGetValue("Features", out var nodeMeta))
            {
                var dims = nodeMeta.Dimensions;
                if (dims.Length == 2 && dims[1] > 0)
                {
                    _proFeatureDimension = dims[1];
                    return dims[1];
                }
                if (dims.Length == 1 && dims[0] > 0)
                {
                    _proFeatureDimension = dims[0];
                    return dims[0];
                }
            }

            throw new InvalidOperationException("无法从 ONNX 模型元数据中读取 Pro 模型特征维度");
        }

        private static (ScanVerdict verdict, float probability) PredictWithInitializedModel(float[] features)
        {
            if (_session == null)
                throw new InvalidOperationException("ModelInvoker 没有初始化");

            int featureCount = _mode switch
            {
                ModelMode.Flash => FeatureSchema.FlashFeatureCount,
                ModelMode.Pro => GetProFeatureDimension(),
                _ => FeatureSchema.StandardFeatureCount
            };

            if (_mode == ModelMode.Pro && _proEnsemble != null)
            {
                float probability = _proEnsemble.Predict(_session, features);
                return (ClassifyVerdict(probability, _proThreshold, _proRecommendedThreshold), probability);
            }

            return RunInference(_session, features, featureCount, GetThreshold(_mode), GetRecommendedThreshold(_mode));
        }

        public static (ScanVerdict verdict, float probability) ScanFile(string filePath)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException("找不到指定文件", filePath);

            if (_mode == ModelMode.Adaptive)
            {
                AdaptiveModelSession adaptiveSession = _adaptiveSession ??
                    throw new InvalidOperationException("ModelInvoker 没有初始化");
                AdaptiveScanResult result = adaptiveSession.ScanFile(filePath);
                return (result.Verdict, result.Probability);
            }

            if (_session == null)
                throw new InvalidOperationException("ModelInvoker 没有初始化");

            byte[] bytes = File.ReadAllBytes(filePath);
            if (!FeatureExtractor.IsPeFile(bytes))
                throw new NotSupportedException("不支持该文件类型");

            float[] floatFeatures;

            switch (_mode)
            {
                case ModelMode.Flash:
                    var flashFeatures = FlashFeatureExtractor.ExtractFromBytes(bytes);
                    floatFeatures = flashFeatures.ToFloatArray();
                    break;
                case ModelMode.Pro:
                    var proFeatures = ProHybridFeatureExtractor.ExtractFromBytes(bytes);
                    floatFeatures = proFeatures.ToFloatArray();
                    break;
                default:
                    var features = FeatureExtractor.ExtractFromBytes(bytes);
                    floatFeatures = features.ToFloatArray();
                    break;
            }

            return PredictWithInitializedModel(floatFeatures);
        }

        private static void ValidateProFeatureDimension()
        {
            int featureCount = GetProFeatureDimension();
            if (featureCount == FeatureSchema.ProFusionFeatureCount)
            {
                if (string.IsNullOrEmpty(_loadedModelPath))
                    throw new InvalidOperationException("Pro 融合模型路径不可用。");
                _proEnsemble?.Dispose();
                _proEnsemble = new ProEnsembleSession(_loadedModelPath);
                return;
            }

            if (featureCount != FeatureSchema.ProHybridFeatureCount)
            {
                throw new InvalidOperationException(
                    $"Pro 模型特征维度不匹配：当前模型为 {featureCount} 维，期望 {FeatureSchema.ProHybridFeatureCount} 维。请重新训练并导出新的 Xdows-Model-Pro.onnx。");
            }
        }

        /// <summary>
        /// 创建统一启用了全量图优化的 SessionOptions。batch=1 的树模型对 IntraOp 线程不敏感，
        /// 提升 GraphOptimizationLevel 是性价比最高的运行时优化（语义保持）。
        /// </summary>
        internal static SessionOptions CreateSessionOptions()
        {
            return new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
        }

internal static float RunProbability(InferenceSession session, float[] features, int featureCount)
        {
            var featuresTensor = new DenseTensor<float>(new Memory<float>(features, 0, featureCount), new[] { 1, featureCount });
            var labelTensor = new DenseTensor<bool>(new Memory<bool>(new bool[] { false }), new[] { 1, 1 });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("Features", featuresTensor),
                NamedOnnxValue.CreateFromTensor("Label", labelTensor)
            };

            using var results = session.Run(inputs);

            var probabilityOutput = results.FirstOrDefault(r => r.Name == "Probability.output");

            float probability = 0f;

            if (probabilityOutput != null)
            {
                var probResult = probabilityOutput.AsEnumerable<float>().ToArray();
                if (probResult.Length > 0) probability = probResult[0] * 100;
            }

            return probability;
        }

        private static (ScanVerdict verdict, float probability) RunInference(InferenceSession session, float[] features, int featureCount, float fixedThreshold, float recommendedThreshold)
        {
            float probability = RunProbability(session, features, featureCount);
            return (ClassifyVerdict(probability, fixedThreshold, recommendedThreshold), probability);
        }

        public static AdaptiveModelSession CreateAdaptiveSession(string? modelDirectory = null, TrainingConfig? config = null)
        {
            string standardPath;
            string flashPath;
            string proPath;
            if (string.IsNullOrWhiteSpace(modelDirectory))
            {
                standardPath = EnsureModelAvailable(DefaultModelFileName);
                flashPath = EnsureModelAvailable(DefaultFlashModelFileName);
                proPath = EnsureModelAvailable(DefaultProModelFileName);
            }
            else
            {
                standardPath = Path.Combine(modelDirectory, DefaultModelFileName);
                flashPath = Path.Combine(modelDirectory, DefaultFlashModelFileName);
                proPath = Path.Combine(modelDirectory, DefaultProModelFileName);
            }
            TrainingConfig effectiveConfig = config ?? new TrainingConfig();
            AdaptiveRecommendedThresholds recommended = new(
                (float)effectiveConfig.FlashThreshold,
                (float)effectiveConfig.StandardThreshold,
                (float)effectiveConfig.ProThreshold);
            if (AutoThresholdSelection)
            {
                recommended = ResolveRecommendedThresholds(effectiveConfig, standardPath, flashPath, proPath);
            }

            return new AdaptiveModelSession(flashPath, standardPath, proPath, effectiveConfig, recommended);
        }

        /// <summary>
        /// Adaptive 会同时用到三个模型，因此按各自的清单分别解析推荐阈值。
        /// 返回推荐阈值集，配置里的固定阈值保持不被覆盖。
        /// </summary>
        private static AdaptiveRecommendedThresholds ResolveRecommendedThresholds(
            TrainingConfig config,
            string standardPath,
            string flashPath,
            string proPath)
        {
            return new AdaptiveRecommendedThresholds(
                (float)ResolveManifestThreshold(flashPath, ModelMode.Flash, config.FlashThreshold),
                (float)ResolveManifestThreshold(standardPath, ModelMode.Standard, config.StandardThreshold),
                (float)ResolveManifestThreshold(proPath, ModelMode.Pro, config.ProThreshold));
        }

        private static double ResolveManifestThreshold(string modelPath, ModelMode mode, double fallbackThreshold)
        {
            if (!ModelThresholdManifest.TryLoad(modelPath, mode.ToString(), out ModelThresholdManifest? manifest, out string? failureReason))
            {
                if (!string.IsNullOrEmpty(failureReason))
                    Console.Error.WriteLine($"[Xdows-Model] {mode} 阈值清单被忽略：{failureReason}");
                return fallbackThreshold;
            }

            float threshold = NormalizeThreshold((float)manifest!.RecommendedThreshold);
            SetRecommendedThreshold(mode, threshold);
            return threshold;
        }

        private static float NormalizeThreshold(float threshold)
        {
            if (float.IsNaN(threshold) || float.IsInfinity(threshold))
                throw new ArgumentOutOfRangeException(nameof(threshold), "Threshold must be a finite percentage.");

            if (threshold < 0 || threshold > 100)
                throw new ArgumentOutOfRangeException(nameof(threshold), "Threshold must be between 0 and 100.");

            return threshold;
        }
    }
}
