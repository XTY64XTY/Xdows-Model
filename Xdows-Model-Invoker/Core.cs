using Microsoft.ML;
using Microsoft.ML.Data;
using System.Reflection;

namespace Xdows_Model_Invoker
{
    public static class ModelInvoker
    {
        private const string DefaultModelFileName = "Xdows-Model.zip";
        private static readonly object _initLock = new();
        private static MLContext? _mlContext;
        private static ITransformer? _model;
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
            var mlContextLocal = new MLContext();
            ITransformer tempModel;
            using (var fileStream = new FileStream(modelPath, FileMode.Open, FileAccess.Read))
            {
                tempModel = mlContextLocal.Model.Load(fileStream, out _);
            }

            var predictionEngine = mlContextLocal.Model.CreatePredictionEngine<ModelInput, ModelOutput>(tempModel);
            var input = new ModelInput { Features = features };
            var prediction = predictionEngine.Predict(input);
            return (prediction.PredictedLabel, prediction.Probability * 100);
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
                if (_model != null && _loadedModelPath == modelPath)
                    return;

                string path = modelPath ?? EnsureModelAvailable();

                _mlContext = new MLContext();
                using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
                _model = _mlContext.Model.Load(fs, out _);
                _loadedModelPath = path;
            }
        }

        public static bool IsInitialized => _model != null;
        public static void UnloadModel()
        {
            lock (_initLock)
            {
                _model = null;
                _mlContext = null;
                _loadedModelPath = null;
            }
        }
        private static (bool isVirus, float probability) PredictWithInitializedModel(float[] features)
        {
            if (_mlContext == null || _model == null)
                throw new InvalidOperationException("ModelInvoker is not initialized. Call Initialize() before scanning.");

            var predictionEngine = _mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(_model);
            var input = new ModelInput { Features = features };
            var prediction = predictionEngine.Predict(input);
            return (prediction.PredictedLabel, prediction.Probability * 100);
        }

        public static (bool isVirus, float probability) ScanFile(string filePath)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException("File not found", filePath);

            if (_model == null)
                throw new InvalidOperationException("ModelInvoker is not initialized");

            var features = FeatureExtractor.ExtractFeatures(filePath);
            var floatFeatures = features.ToFloatArray();
            return PredictWithInitializedModel(floatFeatures);
        }
    }

    internal class ModelInput
    {
        [VectorType(279)]
        public float[] Features { get; set; } = [];
    }

    internal class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public bool PredictedLabel { get; set; }

        [ColumnName("Score")]
        public float Score { get; set; }

        [ColumnName("Probability")]
        public float Probability { get; set; }
    }
}
