#include "xdows_model_native.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <Windows.h>
#include <objbase.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace
{
    struct NativeSession
    {
        int Mode;
        std::filesystem::path ModelPath;
    };

    const wchar_t* ModelNameForMode(int mode)
    {
        switch (mode)
        {
        case XdowsModelNativeModeFlash:
            return L"Xdows-Model-Flash.onnx";
        case XdowsModelNativeModePro:
            return L"Xdows-Model-Pro.onnx";
        default:
            return L"Xdows-Model.onnx";
        }
    }

    float ThresholdForMode(int mode)
    {
        switch (mode)
        {
        case XdowsModelNativeModeFlash:
            return 92.0f;
        case XdowsModelNativeModePro:
            return 80.0f;
        default:
            return 85.0f;
        }
    }

    std::filesystem::path GetModuleDirectory()
    {
        HMODULE module = nullptr;
        wchar_t path[MAX_PATH]{};
        if (GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&XdowsModelNativeInitialize),
            &module) &&
            GetModuleFileNameW(module, path, MAX_PATH) > 0)
        {
            return std::filesystem::path(path).parent_path();
        }

        return std::filesystem::current_path();
    }

    std::filesystem::path ResolveModelPath(const wchar_t* modelDirectory, int mode)
    {
        const wchar_t* modelName = ModelNameForMode(mode);

        if (modelDirectory != nullptr && modelDirectory[0] != L'\0')
        {
            std::filesystem::path explicitPath = std::filesystem::path(modelDirectory) / modelName;
            if (std::filesystem::exists(explicitPath))
                return explicitPath;
        }

        std::filesystem::path modulePath = GetModuleDirectory() / modelName;
        if (std::filesystem::exists(modulePath))
            return modulePath;

        std::filesystem::path cwdPath = std::filesystem::current_path() / modelName;
        if (std::filesystem::exists(cwdPath))
            return cwdPath;

        return {};
    }

    wchar_t* DuplicateString(const std::wstring& value)
    {
        size_t bytes = (value.size() + 1) * sizeof(wchar_t);
        auto* buffer = static_cast<wchar_t*>(CoTaskMemAlloc(bytes));
        if (buffer == nullptr)
            return nullptr;

        memcpy(buffer, value.c_str(), bytes);
        return buffer;
    }

    void ResetResult(XDOWS_MODEL_NATIVE_SCAN_RESULT* result)
    {
        if (result == nullptr)
            return;

        result->Size = sizeof(XDOWS_MODEL_NATIVE_SCAN_RESULT);
        result->Status = XdowsModelNativeStatusOk;
        result->IsThreat = 0;
        result->Probability = 0.0f;
        result->DetectionName = nullptr;
        result->ErrorMessage = nullptr;
    }

    void SetError(XDOWS_MODEL_NATIVE_SCAN_RESULT* result, int status, const std::wstring& message)
    {
        if (result == nullptr)
            return;

        result->Status = status;
        result->IsThreat = 0;
        result->Probability = 0.0f;
        result->DetectionName = nullptr;
        result->ErrorMessage = DuplicateString(message);
    }

    bool ReadAllBytes(const std::filesystem::path& path, std::vector<std::uint8_t>& bytes)
    {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        if (!stream)
            return false;

        std::streamsize size = stream.tellg();
        if (size <= 0)
            return false;

        stream.seekg(0, std::ios::beg);
        bytes.resize(static_cast<size_t>(size));
        return stream.read(reinterpret_cast<char*>(bytes.data()), size).good();
    }

    bool ContainsAscii(const std::vector<std::uint8_t>& bytes, const std::string& needle)
    {
        if (needle.empty() || bytes.size() < needle.size())
            return false;

        return std::search(
            bytes.begin(),
            bytes.end(),
            needle.begin(),
            needle.end(),
            [](std::uint8_t a, char b)
            {
                char ca = static_cast<char>(a);
                char cb = b;
                if (ca >= 'A' && ca <= 'Z')
                    ca = static_cast<char>(ca - 'A' + 'a');
                if (cb >= 'A' && cb <= 'Z')
                    cb = static_cast<char>(cb - 'A' + 'a');
                return ca == cb;
            }) != bytes.end();
    }

    bool IsPeFile(const std::vector<std::uint8_t>& bytes)
    {
        if (bytes.size() < 64 || bytes[0] != 'M' || bytes[1] != 'Z')
            return false;

        std::int32_t peOffset = 0;
        memcpy(&peOffset, bytes.data() + 60, sizeof(peOffset));
        if (peOffset < 0 || static_cast<size_t>(peOffset) + 4 > bytes.size())
            return false;

        return bytes[peOffset] == 'P' && bytes[peOffset + 1] == 'E';
    }

    double ComputeEntropy(const std::vector<std::uint8_t>& bytes, double& highByteRatio)
    {
        std::array<size_t, 256> counts{};
        size_t highBytes = 0;

        for (std::uint8_t b : bytes)
        {
            counts[b]++;
            if (b >= 0x80)
                highBytes++;
        }

        double entropy = 0.0;
        const double total = static_cast<double>(bytes.size());
        for (size_t count : counts)
        {
            if (count == 0)
                continue;

            double p = static_cast<double>(count) / total;
            entropy -= p * std::log2(p);
        }

        highByteRatio = total > 0.0 ? static_cast<double>(highBytes) / total : 0.0;
        return entropy;
    }

    int CountSuspiciousStrings(const std::vector<std::uint8_t>& bytes)
    {
        static const std::array<const char*, 10> needles{
            "virtualalloc",
            "writeprocessmemory",
            "createremotethread",
            "setwindowshookex",
            "loadlibrary",
            "winexec",
            "shellexecute",
            "powershell",
            "rundll32",
            "cmd.exe"
        };

        int count = 0;
        for (const char* needle : needles)
        {
            if (ContainsAscii(bytes, needle))
                count++;
        }
        return count;
    }

    std::wstring ModeName(int mode)
    {
        switch (mode)
        {
        case XdowsModelNativeModeFlash:
            return L"Flash";
        case XdowsModelNativeModePro:
            return L"Pro";
        default:
            return L"Standard";
        }
    }
}

extern "C" XDOWS_MODEL_NATIVE_API int __stdcall XdowsModelNativeInitialize(
    const wchar_t* modelDirectory,
    int mode,
    void** session)
{
    if (session == nullptr)
        return XdowsModelNativeStatusInvalidArgument;

    *session = nullptr;

    std::filesystem::path modelPath = ResolveModelPath(modelDirectory, mode);
    if (modelPath.empty())
        return XdowsModelNativeStatusModelNotFound;

    try
    {
        auto* nativeSession = new NativeSession();
        nativeSession->Mode = mode;
        nativeSession->ModelPath = modelPath;
        *session = nativeSession;
        return XdowsModelNativeStatusOk;
    }
    catch (...)
    {
        return XdowsModelNativeStatusInternalError;
    }
}

extern "C" XDOWS_MODEL_NATIVE_API int __stdcall XdowsModelNativeScanFile(
    void* session,
    const wchar_t* filePath,
    XDOWS_MODEL_NATIVE_SCAN_RESULT* result)
{
    ResetResult(result);

    if (session == nullptr || filePath == nullptr || result == nullptr)
        return XdowsModelNativeStatusInvalidArgument;

    auto* nativeSession = static_cast<NativeSession*>(session);
    std::filesystem::path path(filePath);
    if (!std::filesystem::exists(path))
    {
        SetError(result, XdowsModelNativeStatusFileNotFound, L"file-not-found");
        return XdowsModelNativeStatusFileNotFound;
    }

    std::vector<std::uint8_t> bytes;
    if (!ReadAllBytes(path, bytes))
    {
        SetError(result, XdowsModelNativeStatusInternalError, L"read-failed");
        return XdowsModelNativeStatusInternalError;
    }

    if (ContainsAscii(bytes, "EICAR-STANDARD-ANTIVIRUS-TEST-FILE"))
    {
        result->Status = XdowsModelNativeStatusOk;
        result->IsThreat = 1;
        result->Probability = 100.0f;
        result->DetectionName = DuplicateString(L"Xdows.Model.Native.EICAR");
        return XdowsModelNativeStatusOk;
    }

    if (!IsPeFile(bytes))
    {
        result->Status = XdowsModelNativeStatusOk;
        result->IsThreat = 0;
        result->Probability = 0.0f;
        return XdowsModelNativeStatusOk;
    }

    double highByteRatio = 0.0;
    double entropy = ComputeEntropy(bytes, highByteRatio);
    int suspiciousStrings = CountSuspiciousStrings(bytes);

    double sizeFactor = std::min(12.0, std::log(static_cast<double>(bytes.size()) + 1.0));
    double probability = (entropy * 8.0) + (highByteRatio * 20.0) + (suspiciousStrings * 7.0) + sizeFactor;
    probability = std::clamp(probability, 0.0, 100.0);

    result->Status = XdowsModelNativeStatusOk;
    result->Probability = static_cast<float>(probability);

    if (probability >= ThresholdForMode(nativeSession->Mode))
    {
        result->IsThreat = 1;
        result->DetectionName = DuplicateString(
            L"Xdows.Model.Native." + ModeName(nativeSession->Mode) + L".Heuristic");
    }
    else
    {
        result->IsThreat = 0;
    }

    return XdowsModelNativeStatusOk;
}

extern "C" XDOWS_MODEL_NATIVE_API void __stdcall XdowsModelNativeShutdown(
    void* session)
{
    auto* nativeSession = static_cast<NativeSession*>(session);
    delete nativeSession;
}

extern "C" XDOWS_MODEL_NATIVE_API void __stdcall XdowsModelNativeFreeString(
    wchar_t* value)
{
    if (value != nullptr)
        CoTaskMemFree(value);
}
