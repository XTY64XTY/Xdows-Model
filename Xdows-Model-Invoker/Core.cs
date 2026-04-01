using Microsoft.ML;
using Microsoft.ML.Data;
using System.Reflection;

namespace Xdows_Model_Invoker
{
    public static class ModelInvoker
    {
        private const string DefaultModelFileName = "Xdows-Model.zip";

        // Ensure the model file is available in the application's base directory.
        // If it's not present, attempt to copy it from an embedded resource or from the assembly directory.
        private static string EnsureModelAvailable()
        {
            string baseDir = AppContext.BaseDirectory;
            string destPath = Path.Combine(baseDir, DefaultModelFileName);

            if (File.Exists(destPath))
                return destPath;

            var asm = Assembly.GetExecutingAssembly();

            // Try embedded resource
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

            // Try to find the file next to the assembly (useful when shipped alongside the DLL)
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
            var mlContext = new MLContext();

            // Load the model
            ITransformer model;
            using (var fileStream = new FileStream(modelPath, FileMode.Open, FileAccess.Read))
            {
                model = mlContext.Model.Load(fileStream, out _);
            }

            // Create prediction engine
            var predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

            // Create input
            var input = new ModelInput
            {
                Features = features
            };

            // Predict
            var prediction = predictionEngine.Predict(input);

            return (prediction.PredictedLabel, prediction.Probability * 100);
        }

        public static (bool isVirus, float probability) ScanFile(string filePath, string modelPath)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException("File not found", filePath);

            var features = FeatureExtractor.ExtractFeatures(filePath);
            var floatFeatures = features.ToFloatArray();

            return PredictWithMlNet(modelPath, floatFeatures);
        }

        // Overload that uses the default internal model and ensures it is available (copied) before prediction.
        public static (bool isVirus, float probability) ScanFile(string filePath)
        {
            string modelPath = EnsureModelAvailable();
            return ScanFile(filePath, modelPath);
        }
    }

    internal class ModelInput
    {
        [VectorType(279)]
        public float[] Features { get; set; } = System.Array.Empty<float>();
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
