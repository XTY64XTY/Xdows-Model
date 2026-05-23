using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Reflection;

namespace Xdows_Model_Invoker
{
    public static class ModelInvoker
    {
        private const string DefaultModelFileName = "Xdows-Model.onnx";
        private const string DefaultFlashModelFileName = "Xdows-Model-Flash.onnx";
        private static readonly object _initLock = new();
        private static InferenceSession? _session;
        private static string? _loadedModelPath;
        private static bool _isFlashMode;

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
            return RunInference(session, features, FileFeatures.FeatureCount);
        }

        public static (bool isVirus, float probability) ScanFile(string filePath, string modelPath)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException("找不到指定文件", filePath);

            var fileBytes = File.ReadAllBytes(filePath);
            if (!FeatureExtractor.IsPeFile(fileBytes))
                throw new NotSupportedException("不支持该文件类型");

            var features = FeatureExtractor.ExtractFromBytes(fileBytes);
            Initialize(modelPath);
            return PredictWithInitializedModel(features.ToFloatArray());
        }

        public static (bool isVirus, float probability) ScanFileFlash(string filePath, string modelPath)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException("找不到指定文件", filePath);

            if (!FlashFeatureExtractor.IsPeFileFromPath(filePath))
                throw new NotSupportedException("不支持该文件类型");

            var features = FlashFeatureExtractor.ExtractFeatures(filePath);
            InitializeFlash(modelPath);
            return PredictWithInitializedModel(features.ToFloatArray());
        }

        public static void Initialize(string? modelPath = null)
        {
            InitializeCore(modelPath ?? EnsureModelAvailable(DefaultModelFileName), flashMode: false);
        }

        public static void InitializeFlash(string? modelPath = null)
        {
            InitializeCore(modelPath ?? EnsureModelAvailable(DefaultFlashModelFileName), flashMode: true);
        }

        private static void InitializeCore(string path, bool flashMode)
        {
            lock (_initLock)
            {
                if (_session != null && _loadedModelPath == path && _isFlashMode == flashMode)
                    return;

                _session?.Dispose();
                _session = new InferenceSession(path);
                _loadedModelPath = path;
                _isFlashMode = flashMode;
            }
        }

        public static bool IsInitialized => _session != null;
        public static bool IsFlashMode => _isFlashMode;

        public static void UnloadModel()
        {
            lock (_initLock)
            {
                _session?.Dispose();
                _session = null;
                _loadedModelPath = null;
                _isFlashMode = false;
            }
        }

        private static (bool isVirus, float probability) PredictWithInitializedModel(float[] features)
        {
            if (_session == null)
                throw new InvalidOperationException("ModelInvoker 没有初始化");

            int featureCount = _isFlashMode ? FlashFileFeatures.FeatureCount : FileFeatures.FeatureCount;
            return RunInference(_session, features, featureCount);
        }

        public static (bool isVirus, float probability) ScanFile(string filePath)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException("找不到指定文件", filePath);

            if (_session == null)
                throw new InvalidOperationException("ModelInvoker 没有初始化");

            float[] floatFeatures;

            if (_isFlashMode)
            {
                if (!FlashFeatureExtractor.IsPeFileFromPath(filePath))
                    throw new NotSupportedException("不支持该文件类型");

                var flashFeatures = FlashFeatureExtractor.ExtractFeatures(filePath);
                floatFeatures = flashFeatures.ToFloatArray();
            }
            else
            {
                var fileBytes = File.ReadAllBytes(filePath);
                if (!FeatureExtractor.IsPeFile(fileBytes))
                    throw new NotSupportedException("不支持该文件类型");

                var features = FeatureExtractor.ExtractFromBytes(fileBytes);
                floatFeatures = features.ToFloatArray();
            }

            return PredictWithInitializedModel(floatFeatures);
        }

        private static (bool isVirus, float probability) RunInference(InferenceSession session, float[] features, int featureCount)
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

            bool isVirus = false;
            float probability = 0f;

            if (probabilityOutput != null)
            {
                var probResult = probabilityOutput.AsEnumerable<float>().ToArray();
                if (probResult.Length > 0) probability = probResult[0] * 100;
            }

            isVirus = probability >= 90.0f;

            return (isVirus, probability);
        }
    }
}
