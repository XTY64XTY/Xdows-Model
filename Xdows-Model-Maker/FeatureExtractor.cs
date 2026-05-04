namespace Xdows_Model_Maker;

public class FeatureExtractor
{
    public static FileFeatures ExtractFeatures(string filePath)
    {
        var features = new FileFeatures();

        var bytes = File.ReadAllBytes(filePath);

        features.FileSize = bytes.Length;

        ExtractAllFeaturesOptimized(bytes, features);

        return features;
    }

    public static async Task<FileFeatures> ExtractFeaturesAsync(string filePath)
    {
        var features = new FileFeatures();

        var bytes = await File.ReadAllBytesAsync(filePath);

        features.FileSize = bytes.Length;

        ExtractAllFeaturesOptimized(bytes, features);

        return features;
    }

    private static void ExtractAllFeaturesOptimized(byte[] bytes, FileFeatures features)
    {
        if (bytes.Length == 0)
            return;

        var byteCounts = new long[256];
        int printableCount = 0;
        int controlCount = 0;
        int whitespaceCount = 0;
        int letterCount = 0;
        int digitCount = 0;
        int maxZeroRun = 0;
        int currentZeroRun = 0;
        int highByteCount = 0;

        foreach (var b in bytes)
        {
            byteCounts[b]++;

            if (b >= 0x80 && b <= 0xFF)
            {
                highByteCount++;
            }

            if (b >= 32 && b <= 126)
            {
                printableCount++;
                if (b == 32)
                    whitespaceCount++;
                else if ((b >= 65 && b <= 90) || (b >= 97 && b <= 122))
                    letterCount++;
                else if (b >= 48 && b <= 57)
                    digitCount++;
            }
            else
            {
                controlCount++;
            }

            if (b == 0)
            {
                currentZeroRun++;
                if (currentZeroRun > maxZeroRun)
                    maxZeroRun = currentZeroRun;
            }
            else
            {
                currentZeroRun = 0;
            }
        }

        for (int i = 0; i < 256; i++)
        {
            features.ByteFrequency[i] = (double)byteCounts[i] / bytes.Length;
        }

        features.UniqueBytes = byteCounts.Count(c => c > 0);
        features.MostCommonByte = Array.IndexOf(byteCounts, byteCounts.Max());
        features.MostCommonByteRatio = (double)byteCounts.Max() / bytes.Length;
        features.LeastCommonByte = Array.IndexOf(byteCounts, byteCounts.Min());
        features.LeastCommonByteRatio = (double)byteCounts.Min() / bytes.Length;

        features.ZeroByteRatio = (double)byteCounts[0] / bytes.Length;

        features.HighEntropyRatio = (double)highByteCount / bytes.Length;

        double entropy = 0;
        foreach (var count in byteCounts)
        {
            if (count > 0)
            {
                double p = (double)count / bytes.Length;
                entropy -= p * Math.Log(p, 2);
            }
        }
        features.Entropy = entropy;

        ExtractBlockEntropyOptimized(bytes, features);

        features.PrintableCharRatio = (double)printableCount / bytes.Length;
        features.ControlCharRatio = (double)controlCount / bytes.Length;
        features.WhitespaceRatio = (double)whitespaceCount / bytes.Length;
        features.LetterRatio = (double)letterCount / bytes.Length;
        features.DigitRatio = (double)digitCount / bytes.Length;

        features.MaxZeroByteRun = maxZeroRun;
    }

    private static void ExtractBlockEntropyOptimized(byte[] bytes, FileFeatures features)
    {
        const int blockSize = 256;
        int numBlocks = (bytes.Length + blockSize - 1) / blockSize;

        if (numBlocks == 0)
        {
            features.MinBlockEntropy = 0;
            features.MaxBlockEntropy = 0;
            features.MeanBlockEntropy = 0;
            return;
        }

        double minEntropy = double.MaxValue;
        double maxEntropy = double.MinValue;
        double totalEntropy = 0;

        var blockByteCounts = new long[256];

        for (int blockIdx = 0; blockIdx < numBlocks; blockIdx++)
        {
            int start = blockIdx * blockSize;
            int end = Math.Min(start + blockSize, bytes.Length);
            int currentBlockSize = end - start;

            Array.Clear(blockByteCounts, 0, 256);

            for (int i = start; i < end; i++)
            {
                blockByteCounts[bytes[i]]++;
            }

            double blockEntropy = 0;
            foreach (var count in blockByteCounts)
            {
                if (count > 0)
                {
                    double p = (double)count / currentBlockSize;
                    blockEntropy -= p * Math.Log(p, 2);
                }
            }

            if (blockEntropy < minEntropy) minEntropy = blockEntropy;
            if (blockEntropy > maxEntropy) maxEntropy = blockEntropy;
            totalEntropy += blockEntropy;
        }

        features.MinBlockEntropy = minEntropy;
        features.MaxBlockEntropy = maxEntropy;
        features.MeanBlockEntropy = totalEntropy / numBlocks;
    }

    public static bool IsPeFile(byte[] bytes)
    {
        if (bytes.Length < 2 || bytes[0] != 'M' || bytes[1] != 'Z')
            return false;

        if (bytes.Length < 64)
            return false;

        int peOffset = BitConverter.ToInt32(bytes, 60);
        if (peOffset + 4 > bytes.Length)
            return false;

        return bytes[peOffset] == 'P' && bytes[peOffset + 1] == 'E';
    }
}

public class FileFeatures
{
    public const int FeatureCount = 274;

    public double[] ByteFrequency { get; set; } = new double[256];
    public long FileSize { get; set; }
    public double Entropy { get; set; }
    public double MinBlockEntropy { get; set; }
    public double MaxBlockEntropy { get; set; }
    public double MeanBlockEntropy { get; set; }
    public int UniqueBytes { get; set; }
    public int MostCommonByte { get; set; }
    public double MostCommonByteRatio { get; set; }
    public int LeastCommonByte { get; set; }
    public double LeastCommonByteRatio { get; set; }
    public double PrintableCharRatio { get; set; }
    public double ControlCharRatio { get; set; }
    public double WhitespaceRatio { get; set; }
    public double LetterRatio { get; set; }
    public double DigitRatio { get; set; }
    public int MaxZeroByteRun { get; set; }

    public double ZeroByteRatio { get; set; }
    public double HighEntropyRatio { get; set; }

    public float[] ToFloatArray()
    {
        var features = new float[FeatureCount];
        int idx = 0;

        for (int i = 0; i < 256; i++)
        {
            features[idx++] = (float)ByteFrequency[i];
        }

        features[idx++] = (float)Math.Log(FileSize + 1);
        features[idx++] = (float)Entropy;
        features[idx++] = (float)MinBlockEntropy;
        features[idx++] = (float)MaxBlockEntropy;
        features[idx++] = (float)MeanBlockEntropy;
        features[idx++] = UniqueBytes;
        features[idx++] = MostCommonByte;
        features[idx++] = (float)MostCommonByteRatio;
        features[idx++] = LeastCommonByte;
        features[idx++] = (float)LeastCommonByteRatio;
        features[idx++] = (float)PrintableCharRatio;
        features[idx++] = (float)ControlCharRatio;
        features[idx++] = (float)WhitespaceRatio;
        features[idx++] = (float)LetterRatio;
        features[idx++] = (float)DigitRatio;
        features[idx++] = MaxZeroByteRun;
        features[idx++] = (float)ZeroByteRatio;
        features[idx++] = (float)HighEntropyRatio;

        return features;
    }
}
