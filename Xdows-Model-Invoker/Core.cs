using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Reflection;
using Xdows_Model_Config;

namespace Xdows_Model_Invoker
{
    public enum ModelMode
    {
        Standard,
        Flash,
        Pro
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
        private static float _standardThreshold = NormalizeThreshold((float)_defaultConfig.StandardThreshold);
        private static float _flashThreshold = NormalizeThreshold((float)_defaultConfig.FlashThreshold);
        private static float _proThreshold = NormalizeThreshold((float)_defaultConfig.ProThreshold);

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

        public static (bool isVirus, float probability) PredictWithMlNet(string modelPath, float[] features)
        {
            using var session = new InferenceSession(modelPath);
            return RunInference(session, features, FileFeatures.FeatureCount, _standardThreshold);
        }

        public static (bool isVirus, float probability) ScanFile(string filePath, string modelPath)
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

        public static (bool isVirus, float probability) ScanFileFlash(string filePath, string modelPath)
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

        public static (bool isVirus, float probability) ScanFilePro(string filePath, string modelPath)
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

        private static void InitializeCore(string path, ModelMode mode)
        {
            lock (_initLock)
            {
                if (_session != null && _loadedModelPath == path && _mode == mode)
                    return;

                _session?.Dispose();
                _proEnsemble?.Dispose();
                _proEnsemble = null;
                _session = new InferenceSession(path);
                _loadedModelPath = path;
                _mode = mode;
                _proFeatureDimension = null;

                ValidateFeatureDimension(mode);
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

        public static bool IsInitialized => _session != null;
        public static bool IsFlashMode => _mode == ModelMode.Flash;
        public static bool IsProMode => _mode == ModelMode.Pro;
        public static ModelMode CurrentMode => _mode;

        public static void ConfigureThresholds(TrainingConfig config)
        {
            ArgumentNullException.ThrowIfNull(config);

            _standardThreshold = NormalizeThreshold((float)config.StandardThreshold);
            _flashThreshold = NormalizeThreshold((float)config.FlashThreshold);
            _proThreshold = NormalizeThreshold((float)config.ProThreshold);
        }

        public static float GetThreshold(ModelMode mode)
        {
            return mode switch
            {
                ModelMode.Flash => _flashThreshold,
                ModelMode.Pro => _proThreshold,
                _ => _standardThreshold
            };
        }

        public static void UnloadModel()
        {
            lock (_initLock)
            {
                _session?.Dispose();
                _proEnsemble?.Dispose();
                _proEnsemble = null;
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

        private static (bool isVirus, float probability) PredictWithInitializedModel(float[] features)
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
                return (probability >= _proThreshold, probability);
            }

            return RunInference(_session, features, featureCount, GetThreshold(_mode));
        }

        public static (bool isVirus, float probability) ScanFile(string filePath)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException("找不到指定文件", filePath);

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

        internal static float RunProbability(InferenceSession session, float[] features, int featureCount)
        {
            var featuresTensor = new DenseTensor<float>(new Memory<float>(features), new[] { 1, featureCount });
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

        private static (bool isVirus, float probability) RunInference(InferenceSession session, float[] features, int featureCount, float threshold)
        {
            float probability = RunProbability(session, features, featureCount);
            return (probability >= threshold, probability);
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
            return new AdaptiveModelSession(flashPath, standardPath, proPath, config ?? new TrainingConfig());
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
