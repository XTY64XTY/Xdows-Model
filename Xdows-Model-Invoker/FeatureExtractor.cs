using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace Xdows_Model_Invoker;

public class FeatureExtractor
{
    public static FileFeatures ExtractFeatures(string filePath)
    {
        var features = new FileFeatures();
        var fileInfo = new FileInfo(filePath);
        
        var bytes = File.ReadAllBytes(filePath);

        features.FileSize = bytes.Length;

        ExtractAllFeaturesOptimized(bytes, features);

        return features;
    }

    public static async Task<FileFeatures> ExtractFeaturesAsync(string filePath)
    {
        var features = new FileFeatures();
        var fileInfo = new FileInfo(filePath);
        
        var bytes = await File.ReadAllBytesAsync(filePath);

        features.FileSize = bytes.Length;

        ExtractAllFeaturesOptimized(bytes, features);

        return features;
    }

    private static void ExtractAllFeaturesOptimized(byte[] bytes, FileFeatures features)
    {
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

        features.HasDosHeader = bytes.Length >= 2 && bytes[0] == 'M' && bytes[1] == 'Z';
        features.HasPeHeader = false;

        if (features.HasDosHeader && bytes.Length >= 64)
        {
            int peOffset = BitConverter.ToInt32(bytes, 60);
            if (peOffset + 4 <= bytes.Length && bytes[peOffset] == 'P' && bytes[peOffset + 1] == 'E')
            {
                features.HasPeHeader = true;
            }
        }

        features.HasElfHeader = bytes.Length >= 4 && bytes[0] == 0x7F && bytes[1] == 'E' && bytes[2] == 'L' && bytes[3] == 'F';
        features.HasZipHeader = bytes.Length >= 4 && bytes[0] == 0x50 && bytes[1] == 0x4B;
        features.HasRarHeader = bytes.Length >= 4 && bytes[0] == 0x52 && bytes[1] == 0x61 && bytes[2] == 0x72 && bytes[3] == 0x21;

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
}

public class FileFeatures
{
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
    public bool HasDosHeader { get; set; }
    public bool HasPeHeader { get; set; }
    public bool HasElfHeader { get; set; }
    public bool HasZipHeader { get; set; }
    public bool HasRarHeader { get; set; }
    public int MaxZeroByteRun { get; set; }
    
    public double ZeroByteRatio { get; set; }
    public double HighEntropyRatio { get; set; }

    public float[] ToFloatArray()
    {
        var features = new List<float>();
        
        foreach (var freq in ByteFrequency)
        {
            features.Add((float)freq);
        }
        
        features.Add((float)FileSize);
        features.Add((float)Entropy);
        features.Add((float)MinBlockEntropy);
        features.Add((float)MaxBlockEntropy);
        features.Add((float)MeanBlockEntropy);
        features.Add((float)UniqueBytes);
        features.Add((float)MostCommonByte);
        features.Add((float)MostCommonByteRatio);
        features.Add((float)LeastCommonByte);
        features.Add((float)LeastCommonByteRatio);
        features.Add((float)PrintableCharRatio);
        features.Add((float)ControlCharRatio);
        features.Add((float)WhitespaceRatio);
        features.Add((float)LetterRatio);
        features.Add((float)DigitRatio);
        features.Add(HasDosHeader ? 1.0f : 0.0f);
        features.Add(HasPeHeader ? 1.0f : 0.0f);
        features.Add(HasElfHeader ? 1.0f : 0.0f);
        features.Add(HasZipHeader ? 1.0f : 0.0f);
        features.Add(HasRarHeader ? 1.0f : 0.0f);
        features.Add((float)MaxZeroByteRun);
        
        features.Add((float)ZeroByteRatio);
        features.Add((float)HighEntropyRatio);
        
        return features.ToArray();
    }
}
