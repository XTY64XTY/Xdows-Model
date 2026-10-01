#pragma once

#ifdef XDOWS_MODEL_NATIVE_EXPORTS
#define XDOWS_MODEL_NATIVE_API __declspec(dllexport)
#else
#define XDOWS_MODEL_NATIVE_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum XDOWS_MODEL_NATIVE_MODE {
    XdowsModelNativeModeStandard = 0,
    XdowsModelNativeModeFlash = 1,
    XdowsModelNativeModePro = 2,
    XdowsModelNativeModeAdaptive = 3
} XDOWS_MODEL_NATIVE_MODE;

typedef enum XDOWS_MODEL_NATIVE_STATUS {
    XdowsModelNativeStatusOk = 0,
    XdowsModelNativeStatusInvalidArgument = 1,
    XdowsModelNativeStatusFileNotFound = 2,
    XdowsModelNativeStatusUnsupportedFile = 3,
    XdowsModelNativeStatusModelNotFound = 4,
    XdowsModelNativeStatusInternalError = 5,
    // Pro 模型清单校验失败（对齐 Managed ProModelManifest.Validate 抛出的异常）。
    XdowsModelNativeStatusModelManifestInvalid = 6
} XDOWS_MODEL_NATIVE_STATUS;

typedef enum XDOWS_MODEL_NATIVE_VERDICT {
    XdowsModelNativeVerdictClean = 0,
    XdowsModelNativeVerdictSuspicious = 1,
    XdowsModelNativeVerdictMalware = 2
} XDOWS_MODEL_NATIVE_VERDICT;

// 版本化协议：调用方应先将 Size 设为 sizeof(XDOWS_MODEL_NATIVE_SCAN_RESULT)。
// 当 Size 不足（旧调用方按旧布局编译）时，DLL 只写旧字段（不含 Verdict），避免越界写内存。
typedef struct XDOWS_MODEL_NATIVE_SCAN_RESULT {
    int Size;
    int Status;
    int IsThreat;
    float Probability;
    wchar_t* DetectionName;
    wchar_t* ErrorMessage;
    int Verdict;
} XDOWS_MODEL_NATIVE_SCAN_RESULT;

XDOWS_MODEL_NATIVE_API int __stdcall XdowsModelNativeInitialize(
    const wchar_t* modelDirectory,
    int mode,
    void** session);

XDOWS_MODEL_NATIVE_API int __stdcall XdowsModelNativeScanFile(
    void* session,
    const wchar_t* filePath,
    XDOWS_MODEL_NATIVE_SCAN_RESULT* result);

XDOWS_MODEL_NATIVE_API void __stdcall XdowsModelNativeShutdown(
    void* session);

XDOWS_MODEL_NATIVE_API void __stdcall XdowsModelNativeFreeString(
    wchar_t* value);

// 会话信息。与扫描结果同属版本化协议：调用方必须先把 Size 设为
// sizeof(XDOWS_MODEL_NATIVE_SESSION_INFO)，否则调用会被拒绝。
typedef struct XDOWS_MODEL_NATIVE_SESSION_INFO {
    int Size;
    int Mode;                       // XDOWS_MODEL_NATIVE_MODE
    int FeatureCount;               // 会话特征维度；Pro 为融合模型输入维度（=分支数），Adaptive 为 0
    int AutoThresholdSelection;
    float FixedStandard;
    float FixedFlash;
    float FixedPro;
    float RecommendedStandard;
    float RecommendedFlash;
    float RecommendedPro;
    wchar_t* ModelPath;             // 由 DLL 分配，调用方用 XdowsModelNativeFreeString 释放
} XDOWS_MODEL_NATIVE_SESSION_INFO;

// 进程级阈值配置，对齐 Managed 的静态 ConfigureThresholds / AutoThresholdSelection。
// fixedThresholds 指向 3 个百分比（Standard, Flash, Pro），须为有限 0..100 值。
// 只影响之后新建的会话。
XDOWS_MODEL_NATIVE_API int __stdcall XdowsModelNativeConfigureThresholds(
    const float* fixedThresholds,
    int autoThresholdSelection);

// 按调用方给定的特征向量直接推理，对齐 Managed PredictWithMlNet / RunProbability。
// Standard 需 299 维、Flash 需 68 维、Pro 可为混合特征（519/5143）或融合向量（4/5）；
// Adaptive 会话不支持。结果复用 XDOWS_MODEL_NATIVE_SCAN_RESULT，DetectionName 恒为 NULL。
XDOWS_MODEL_NATIVE_API int __stdcall XdowsModelNativePredict(
    void* session,
    const float* features,
    int featureCount,
    XDOWS_MODEL_NATIVE_SCAN_RESULT* result);

// 查询会话状态与生效阈值，对齐 Managed IsInitialized / CurrentMode / GetThreshold 系列。
XDOWS_MODEL_NATIVE_API int __stdcall XdowsModelNativeGetSessionInfo(
    void* session,
    XDOWS_MODEL_NATIVE_SESSION_INFO* info);

#ifdef __cplusplus
}
#endif
