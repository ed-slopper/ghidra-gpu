package com.xebyte.gpu.cuda;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Platform;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.IntByReference;
import com.sun.jna.ptr.LongByReference;
import com.sun.jna.ptr.PointerByReference;

/**
 * The slice of the CUDA <em>driver</em> API ghidra-gpu uses, bound through JNA.
 *
 * <p>The driver API ships with the NVIDIA display driver itself ({@code nvcuda.dll} on
 * Windows, {@code libcuda.so.1} on Linux), so nothing beyond a driver has to be installed:
 * no CUDA toolkit, no runtime DLL, no JCuda. JNA is already on Ghidra's classpath
 * ({@code Ghidra/Framework/Pty/lib}), so the extension adds no jar either.
 *
 * <p>Device pointers ({@code CUdeviceptr}) are 64-bit integers and are passed as
 * {@code long}; handles (context, module, function) are opaque {@link Pointer}s.
 */
public interface CudaDriver extends Library {

    static CudaDriver load() {
        return Native.load(Platform.isWindows() ? "nvcuda" : "cuda", CudaDriver.class);
    }

    int CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1;
    int CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;
    int CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13;
    int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
    int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;

    int CU_JIT_ERROR_LOG_BUFFER = 5;
    int CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6;

    int cuInit(int flags);
    int cuDriverGetVersion(IntByReference version);
    int cuDeviceGetCount(IntByReference count);
    int cuDeviceGet(IntByReference device, int ordinal);
    int cuDeviceGetName(byte[] name, int len, int device);
    int cuDeviceGetAttribute(IntByReference value, int attribute, int device);
    int cuDeviceTotalMem_v2(LongByReference bytes, int device);

    int cuCtxCreate_v2(PointerByReference ctx, int flags, int device);
    int cuCtxSetCurrent(Pointer ctx);
    int cuCtxSynchronize();
    int cuMemGetInfo_v2(LongByReference free, LongByReference total);

    int cuModuleLoadDataEx(PointerByReference module, Pointer image, int numOptions, int[] options, Pointer[] optionValues);
    int cuModuleGetFunction(PointerByReference function, Pointer module, String name);

    int cuMemAlloc_v2(LongByReference dptr, long bytes);
    int cuMemFree_v2(long dptr);
    int cuMemcpyHtoD_v2(long dst, Pointer src, long bytes);
    int cuMemcpyDtoH_v2(Pointer dst, long src, long bytes);
    int cuMemsetD8_v2(long dst, byte value, long count);

    // Primitive-array overloads: JNA marshals the array to native memory for the call
    // (and copies it back afterwards for the DtoH forms), so no Memory staging copy.
    int cuMemcpyHtoD_v2(long dst, byte[] src, long bytes);
    int cuMemcpyHtoD_v2(long dst, int[] src, long bytes);
    int cuMemcpyHtoD_v2(long dst, long[] src, long bytes);
    int cuMemcpyHtoD_v2(long dst, float[] src, long bytes);
    int cuMemcpyHtoD_v2(long dst, double[] src, long bytes);
    int cuMemcpyDtoH_v2(byte[] dst, long src, long bytes);
    int cuMemcpyDtoH_v2(int[] dst, long src, long bytes);
    int cuMemcpyDtoH_v2(long[] dst, long src, long bytes);
    int cuMemcpyDtoH_v2(float[] dst, long src, long bytes);
    int cuMemcpyDtoH_v2(double[] dst, long src, long bytes);

    int cuLaunchKernel(Pointer f, int gridX, int gridY, int gridZ, int blockX, int blockY, int blockZ,
                       int sharedMemBytes, Pointer stream, Pointer kernelParams, Pointer extra);

    int cuGetErrorName(int error, PointerByReference str);
    int cuGetErrorString(int error, PointerByReference str);
}
