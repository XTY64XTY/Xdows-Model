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
    XdowsModelNativeModePro = 2
} XDOWS_MODEL_NATIVE_MODE;

typedef enum XDOWS_MODEL_NATIVE_STATUS {
    XdowsModelNativeStatusOk = 0,
    XdowsModelNativeStatusInvalidArgument = 1,
    XdowsModelNativeStatusFileNotFound = 2,
    XdowsModelNativeStatusUnsupportedFile = 3,
    XdowsModelNativeStatusModelNotFound = 4,
    XdowsModelNativeStatusInternalError = 5
} XDOWS_MODEL_NATIVE_STATUS;

typedef struct XDOWS_MODEL_NATIVE_SCAN_RESULT {
    int Size;
    int Status;
    int IsThreat;
    float Probability;
    wchar_t* DetectionName;
    wchar_t* ErrorMessage;
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

#ifdef __cplusplus
}
#endif
