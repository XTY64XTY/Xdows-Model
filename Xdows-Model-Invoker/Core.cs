using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Reflection;

namespace Xdows_Model_Invoker
{
    public static class ModelInvoker
    {
        private const string DefaultModelFileName = "Xdows-Model.onnx";
        private static readonly object _initLock = new();
        private static SessionOptions? _sessionOptions;
        private static InferenceSession? _session;
        private static string? _loadedModelPath;
        private static string EnsureModelAvailable()
        {
            string baseDir = AppContext.BaseDirectory;
            string destPath = Path.Combine(baseDir, DefaultModelFileName);

            if (File.Exists(destPath))
                return destPath;

            var asm = Assembly.GetExecutingAssembly();

            var resourceName = asm.GetManifestResourceNames().FirstOrDefault(n => n.EndsWith(DefaultModelFileName, StringComparison.OrdinalIgnoreCase));
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
            var candidate = Path.Combine(asmDir, DefaultModelFileName);
            if (File.Exists(candidate))
            {
                File.Copy(candidate, destPath, overwrite: true);
                return destPath;
            }

            throw new FileNotFoundException("Model file not found. Expected to find '" + DefaultModelFileName + "' as an embedded resource or next to the Invoker assembly.", DefaultModelFileName);
        }

        public static (bool isVirus, float probability) PredictWithMlNet(string modelPath, float[] features)
        {
            using var session = CreateSession(modelPath);
            return RunInference(session, features);
        }

        public static (bool isVirus, float probability) ScanFile(string filePath, string modelPath)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException("File not found", filePath);

            var features = FeatureExtractor.ExtractFeatures(filePath);
            var floatFeatures = features.ToFloatArray();

            Initialize(modelPath);
            return PredictWithInitializedModel(floatFeatures);
        }
        public static void Initialize(string? modelPath = null)
        {
            lock (_initLock)
            {
                if (_session != null && _loadedModelPath == modelPath)
                    return;

                string path = modelPath ?? EnsureModelAvailable();

                _session?.Dispose();
                _sessionOptions?.Dispose();

                _sessionOptions = new SessionOptions();
                _session = CreateSession(path);
                _loadedModelPath = path;
            }
        }

        public static bool IsInitialized => _session != null;
        public static void UnloadModel()
        {
            lock (_initLock)
            {
                _session?.Dispose();
                _session = null;
                _sessionOptions?.Dispose();
                _sessionOptions = null;
                _loadedModelPath = null;
            }
        }
        private static (bool isVirus, float probability) PredictWithInitializedModel(float[] features)
        {
            if (_session == null)
                throw new InvalidOperationException("ModelInvoker is not initialized. Call Initialize() before scanning.");

            return RunInference(_session, features);
        }

        public static (bool isVirus, float probability) ScanFile(string filePath)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException("File not found", filePath);

            if (_session == null)
                throw new InvalidOperationException("ModelInvoker is not initialized");

            var features = FeatureExtractor.ExtractFeatures(filePath);
            var floatFeatures = features.ToFloatArray();
            return PredictWithInitializedModel(floatFeatures);
        }

        private static InferenceSession CreateSession(string modelPath)
        {
            var options = new SessionOptions();
            return new InferenceSession(modelPath, options);
        }

        private static (bool isVirus, float probability) RunInference(InferenceSession session, float[] features)
        {
            var featuresTensor = new DenseTensor<float>(new Memory<float>(features), new[] { 1, 279 });
            var labelTensor = new DenseTensor<bool>(new Memory<bool>(new bool[] { false }), new[] { 1, 1 });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("Features", featuresTensor),
                NamedOnnxValue.CreateFromTensor("Label", labelTensor)
            };

            using var results = session.Run(inputs);

            var predictedLabelOutput = results.FirstOrDefault(r => r.Name == "PredictedLabel.output");
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
