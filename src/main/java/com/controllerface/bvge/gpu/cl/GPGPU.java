package com.controllerface.bvge.gpu.cl;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.editor.Editor;
import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes;
import com.controllerface.bvge.memory.GPUCoreMemory;
import org.lwjgl.BufferUtils;
import org.lwjgl.opencl.CL10;
import org.lwjgl.system.MemoryStack;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;

import static org.lwjgl.opencl.CL12.*;
import static org.lwjgl.opencl.CL12GL.clCreateFromGLBuffer;
import static org.lwjgl.opencl.CL20.*;

/**
 * Core class used for executing General Purpose GPU (GPGPU) functions
 */
public class GPGPU
{
    //#region Constants

    private static final long FLAGS_WRITE_GPU = CL_MEM_READ_WRITE;
    private static final long FLAGS_WRITE_CPU_COPY = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
    private static final long FLAGS_READ_CPU_COPY = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;

    /**
     * A convenience object, used when clearing out buffers to fill them with zeroes
     */
    private static final ByteBuffer ZERO_PATTERN_BUFFER = BufferUtils.createByteBuffer(4).order(ByteOrder.nativeOrder());
    private static final ByteBuffer NEGATIVE_ONE_PATTERN_BUFFER = BufferUtils.createByteBuffer(4).order(ByteOrder.nativeOrder());

    static
    {
        ZERO_PATTERN_BUFFER.putInt(0,0);
        NEGATIVE_ONE_PATTERN_BUFFER.putInt(0, -1);
    }

    //#endregion

    //#region Workgroup Variables

    public static GPUCoreMemory core_memory;
    public static CL_ComputeController compute;

    //#endregion

    //#region Init Methods

//    private static long init_device()
//    {
//        // The platform, device type and device number
//        // that will be used
//        long deviceType = CL_DEVICE_TYPE_GPU;
//
//        // Obtain the number of platforms
//        int[] numPlatformsArray = new int[1];
//        clGetPlatformIDs(null, numPlatformsArray);
//        int numPlatforms = numPlatformsArray[0];
//
//        // Obtain a platform ID
//        var platform_buffer = MemoryUtil.memAllocPointer(numPlatforms);
//        clGetPlatformIDs(platform_buffer, (IntBuffer) null);
//        var platform = platform_buffer.get();
//        MemoryUtil.memFree(platform_buffer);
//
//        // Obtain the number of devices for the platform
//        int[] numDevicesArray = new int[1];
//        clGetDeviceIDs(platform, deviceType, null, numDevicesArray);
//        int numDevices = numDevicesArray[0];
//
//        // Obtain a device ID
//        var device_buffer = MemoryUtil.memAllocPointer(numDevices);
//        clGetDeviceIDs(platform, deviceType, device_buffer, (IntBuffer) null);
//        long device = device_buffer.get();
//        MemoryUtil.memFree(device_buffer);
//
//        var dc = wglGetCurrentDC();
//        var ctx = wglGetCurrentContext();
//
//        // todo: the above code is windows specific add linux code path,
//        //  should look something like this:
//        // var ctx = glXGetCurrentContext();
//        // var dc = glXGetCurrentDrawable(); OR glfwGetX11Display();
//        // contextProperties.addProperty(CL_GLX_DISPLAY_KHR, dc);
//
//        // Create a context for the selected device
//        var ctx_props_buffer = MemoryUtil.memAllocPointer(7);
//        ctx_props_buffer.put(CL_CONTEXT_PLATFORM)
//            .put(platform)
//            .put(CL_GL_CONTEXT_KHR)
//            .put(ctx)
//            .put(CL_WGL_HDC_KHR)
//            .put(dc)
//            .put(0L)
//            .flip();
//
//        ptr_context = clCreateContext(ctx_props_buffer,
//            device, null, 0L, null);
//
//        // Create a command-queue for the selected device
//        ptr_compute_queue = clCreateCommandQueue(ptr_context,
//            device, 0, (IntBuffer) null);
//
//        ptr_render_queue = clCreateCommandQueue(ptr_context,
//            device, 0, (IntBuffer) null);
//
//        ptr_sector_queue = clCreateCommandQueue(ptr_context,
//            device, 0, (IntBuffer) null);
//
//        MemoryUtil.memFree(ctx_props_buffer);
//
//        return device;
//    }

    //#endregion

    //#region Utility Methods

    public static void cl_read_int_buffer(long queue_ptr, long src_ptr, int[] dst)
    {
        int result = clEnqueueReadBuffer(queue_ptr,
            src_ptr,
            true,
            0,
            dst,
            null,
            null);
        if (result != CL_SUCCESS)
        {
            System.out.println("Error on int buffer read : " + result);
            throw new RuntimeException("Error on int buffer read : " + result);
        }
    }

    public static void cl_read_float_buffer(long queue_ptr, long src_ptr, float[] dst)
    {
        int result = clEnqueueReadBuffer(queue_ptr,
            src_ptr,
            true,
            0,
            dst,
            null,
            null);
        if (result != CL_SUCCESS)
        {
            System.out.println("Error on float buffer read : " + result);
            throw new RuntimeException("Error on float buffer read : " + result);
        }
    }

    public static void cl_read_short_buffer(long queue_ptr, long src_ptr, short[] dst)
    {
        int result = clEnqueueReadBuffer(queue_ptr,
            src_ptr,
            true,
            0,
            dst,
            null,
            null);
        if (result != CL_SUCCESS)
        {
            System.out.println("Error on short buffer read : " + result);
            throw new RuntimeException("Error on short buffer read : " + result);
        }
    }

    public static long cl_new_buffer(long size)
    {
        try (var stack = MemoryStack.stackPush())
        {
            var status = stack.mallocInt(1);
            long ptr = clCreateBuffer(compute.context.ptr(), FLAGS_WRITE_GPU, size, status);
            int result = status.get(0);
            if (result != CL_SUCCESS)
            {
                System.out.println("Error on new buffer creation : " + result);
                throw new RuntimeException("Error on new buffer creation : " + result);
            }
            return ptr;
        }
    }

    public static long cl_new_int_arg_buffer(int[] src)
    {
        int[] status = new int[1];
        long ptr = clCreateBuffer(compute.context.ptr(), FLAGS_WRITE_CPU_COPY, src, status);
        int result = status[0];
        if (result != CL_SUCCESS)
        {
            System.out.println("Error on new int arg buffer creation : " + result);
            throw new RuntimeException("Error on new int arg buffer creation : " + result);
        }
        return ptr;
    }

    public static long cl_new_cpu_copy_buffer(float[] src)
    {
        int[] status = new int[1];
        long ptr = clCreateBuffer(compute.context.ptr(), FLAGS_READ_CPU_COPY, src, status);
        int result = status[0];
        if (result != CL_SUCCESS)
        {
            System.out.println("Error on new cpu copy buffer creation : " + result);
            throw new RuntimeException("Error on new cpu copy buffer creation : " + result);
        }
        return ptr;
    }

    public static void cl_zero_buffer(long queue_ptr, long buffer_ptr, long buffer_size)
    {
        int result = clEnqueueFillBuffer(queue_ptr,
            buffer_ptr,
            ZERO_PATTERN_BUFFER,
            0,
            buffer_size,
            null,
            null);
        if (result != CL_SUCCESS)
        {
            System.out.println("Error on filling buffer with 0 int value: " + result);
            throw new RuntimeException("Error on filling buffer with 0 int value: " + result);
        }
    }

    public static void cl_negative_one_buffer(long queue_ptr, long buffer_ptr, long buffer_size)
    {
        int result = clEnqueueFillBuffer(queue_ptr,
            buffer_ptr,
            NEGATIVE_ONE_PATTERN_BUFFER,
            0,
            buffer_size,
            null,
            null);
        if (result != CL_SUCCESS)
        {
            System.out.println("Error on filling buffer with -1 int value: " + result);
            throw new RuntimeException("Error on filling buffer with -1 int value: " + result);
        }
    }

    public static long cl_new_pinned_buffer(long size)
    {
        try (var stack = MemoryStack.stackPush())
        {
            var status = stack.mallocInt(1);
            long flags = CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;
            long ptr = clCreateBuffer(compute.context.ptr(), flags, size, status);
            int result = status.get(0);
            if (result != CL_SUCCESS)
            {
                System.out.println("Error on creating new pinned int buffer: " + result);
                throw new RuntimeException("Error on creating new pinned int buffer: " + result);
            }
            return ptr;
        }
    }

    public static void cl_map_read_int_buffer(long queue_ptr, long pinned_ptr, long size, int count, int[] output)
    {
        try (var stack = MemoryStack.stackPush())
        {
            var status = stack.mallocInt(1);
            var out = clEnqueueMapBuffer(queue_ptr,
                pinned_ptr,
                true,
                CL_MAP_READ,
                0,
                size * (long) count,
                null,
                null,
                status,
                null);

            assert out != null;
            int result = status.get(0);
            if (result != CL_SUCCESS)
            {
                System.out.println("Error on int array map: " + result);
                throw new RuntimeException("Error on int array map: " + result);
            }
            var int_buffer = out.order(ByteOrder.LITTLE_ENDIAN).asIntBuffer();
            for (int i = 0; i < count; i++)
            {
                output[i] = int_buffer.get(i);
            }
            result = clEnqueueUnmapMemObject(queue_ptr, pinned_ptr, out, null, null);
            if (result != CL_SUCCESS)
            {
                System.out.println("Error on int array unmap: " + result);
                throw new RuntimeException("Error on int array unmap: " + result);
            }
        }
    }

    public static void cl_map_read_float_buffer(long queue_ptr, long pinned_ptr, long size, int count, float[] output)
    {
        try (var stack = MemoryStack.stackPush())
        {
            var status = stack.mallocInt(1);
            var out = clEnqueueMapBuffer(queue_ptr,
                pinned_ptr,
                true,
                CL_MAP_READ,
                0,
                size * (long) count,
                null,
                null,
                status,
                null);

            assert out != null;

            int result = status.get(0);
            if (result != CL_SUCCESS)
            {
                System.out.println("Error on float array map: " + result);
                throw new RuntimeException("Error on float array map: " + result);
            }

            var float_buffer = out.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            for (int i = 0; i < count; i++)
            {
                output[i] = float_buffer.get(i);
            }
            result = clEnqueueUnmapMemObject(queue_ptr, pinned_ptr, out, null, null);
            if (result != CL_SUCCESS)
            {
                System.out.println("Error on float array map: " + result);
                throw new RuntimeException("Error on float array map: " + result);
            }
        }
    }

    public static void cl_map_read_short_buffer(long queue_ptr, long pinned_ptr, long size, int count, short[] output)
    {
        try (var stack = MemoryStack.stackPush())
        {
            var status = stack.mallocInt(1);
            var out = clEnqueueMapBuffer(queue_ptr,
                pinned_ptr,
                true,
                CL_MAP_READ,
                0,
                size * (long) count,
                null,
                null,
                status,
                null);

            assert out != null;
            int result = status.get(0);
            if (result != CL_SUCCESS)
            {
                System.out.println("Error on short array map: " + result);
                throw new RuntimeException("Error on short array map: " + result);
            }
            var short_buffer = out.order(ByteOrder.LITTLE_ENDIAN).asShortBuffer();
            for (int i = 0; i < count; i++)
            {
                output[i] = short_buffer.get(i);
            }
            result = clEnqueueUnmapMemObject(queue_ptr, pinned_ptr, out, null, null);
            if (result != CL_SUCCESS)
            {
                System.out.println("Error on short array unmap: " + result);
                throw new RuntimeException("Error on short array unmap: " + result);
            }
        }
    }

    public static int[] cl_read_pinned_int_buffer(long queue_ptr, long pinned_ptr, long size, int count)
    {
        try (var stack = MemoryStack.stackPush())
        {
            var status = stack.mallocInt(1);
            var out = clEnqueueMapBuffer(queue_ptr,
                pinned_ptr,
                true,
                CL_MAP_READ,
                0,
                size * (long) count,
                null,
                null,
                status,
                null);

            assert out != null;
            int result = status.get(0);
            if (result != CL_SUCCESS)
            {
                System.out.println("Error on pinned int array map: " + result);
                throw new RuntimeException("Error on pinned int array map: " + result);
            }
            int[] value = new int[count];
            var int_buffer = out.order(ByteOrder.LITTLE_ENDIAN).asIntBuffer();
            for (int i = 0; i < count; i++)
            {
                value[i] = int_buffer.get(i);
            }
            result = clEnqueueUnmapMemObject(queue_ptr, pinned_ptr, out, null, null);
            if (result != CL_SUCCESS)
            {
                System.out.println("Error on pinned int array unmap: " + result);
                throw new RuntimeException("Error on pinned int array unmap: " + result);
            }
            return value;
        }
    }

    public static float[] cl_read_pinned_float_buffer(long queue_ptr, long pinned_ptr, long size, int count)
    {
        return cl_read_pinned_float_buffer(queue_ptr, pinned_ptr, size, count, new float[count]);
    }

    public static float[] cl_read_pinned_float_buffer(long queue_ptr, long pinned_ptr, long size, int count, float[] output)
    {
        try (var stack = MemoryStack.stackPush())
        {
            var status = stack.mallocInt(1);
            var out = clEnqueueMapBuffer(queue_ptr,
                pinned_ptr,
                true,
                CL_MAP_READ,
                0,
                size * (long) count,
                null,
                null,
                status,
                null);

            assert out != null;

            int result = status.get(0);
            if (result != CL_SUCCESS)
            {
                System.out.println("Error on pinned float array map: " + result);
                throw new RuntimeException("Error on pinned float array map: " + result);
            }

            var float_buffer = out.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            for (int i = 0; i < count; i++)
            {
                output[i] = float_buffer.get(i);
            }
            result = clEnqueueUnmapMemObject(queue_ptr, pinned_ptr, out, null, null);
            if (result != CL_SUCCESS)
            {
                System.out.println("Error on pinned float array unmap: " + result);
                throw new RuntimeException("Error on pinned float array unmap: " + result);
            }
            return output;
        }
    }

    public static long cl_new_pinned_int()
    {
        try (var stack = MemoryStack.stackPush())
        {
            var status = stack.mallocInt(1);
            long flags = CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;
            long ptr = CL10.clCreateBuffer(compute.context.ptr(), flags, CL_DataTypes.cl_int.size(), status);
            int result = status.get(0);
            if (result != CL_SUCCESS)
            {
                System.out.println("Error on new pinned int creation: " + result);
                throw new RuntimeException("Error on new pinned int creation: " + result);
            }
            return ptr;
        }
    }

    public static long cl_new_unpinned_int()
    {
        try (var stack = MemoryStack.stackPush())
        {
            var status = stack.mallocInt(1);
            long ptr = CL10.clCreateBuffer(compute.context.ptr(), CL_MEM_HOST_READ_ONLY, CL_DataTypes.cl_int.size(), status);
            int result = status.get(0);
            if (result != CL_SUCCESS)
            {
                System.out.println("Error on new unpinned int creation: " + result);
                throw new RuntimeException("Error on new unpinned int creation: " + result);
            }
            return ptr;
        }
    }

    public static ByteBuffer cl_new_svm_int()
    {
        return clSVMAlloc(compute.context.ptr(), CL_MEM_READ_WRITE, CL_DataTypes.cl_int.size(), 0);
    }

    public static int cl_read_svm_int(long queue_ptr, ByteBuffer svm_buffer)
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;
        int result = clEnqueueSVMMap(queue_ptr, true, CL_MAP_READ, svm_buffer, null, null);
        if (result != CL_SUCCESS)
        {
            System.out.println("Error on svm buffer map: " + result);
            throw new RuntimeException("Error on svm buffer map: " + result);
        }
        int v = svm_buffer.getInt(0);
        result = clEnqueueSVMUnmap(queue_ptr, svm_buffer, null, null);
        if (result != CL_SUCCESS)
        {
            System.out.println("Error on svm buffer unmap: " + result);
            throw new RuntimeException("Error on svm buffer unmap: " + result);
        }
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("cl_read_svm_int", String.valueOf(e));
        }
        return v;
    }

    public static int cl_read_unpinned_int(long queue_ptr, long pinned_ptr)
    {
        try (var stack = MemoryStack.stackPush())
        {
            var pb = stack.mallocInt(1);
            int result = clEnqueueReadBuffer(queue_ptr, pinned_ptr, true, 0, pb, null, null);
            if (result != CL_SUCCESS)
            {
                System.out.println("Error on int read: " + result);
                throw new RuntimeException("Error on int read: " + result);
            }
            return pb.get(0);
        }
    }

    public static int cl_read_pinned_int(long queue_ptr, long pinned_ptr)
    {
        try (var stack = MemoryStack.stackPush())
        {
            var status = stack.mallocInt(1);
            var out = CL10.clEnqueueMapBuffer(queue_ptr,
                pinned_ptr,
                true,
                CL_MAP_READ,
                0,
                CL_DataTypes.cl_int.size(),
                null,
                null,
                status,
                null);

            assert out != null;

            int result = status.get(0);
            if (result != CL_SUCCESS)
            {
                System.out.println("Error on pinned int map: " + result);
                throw new RuntimeException("Error on pinned int map: " + result);
            }

            int value = out.order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(0);
            result = clEnqueueUnmapMemObject(queue_ptr, pinned_ptr, out, null, null);
            if (result != CL_SUCCESS)
            {
                System.out.println("Error on pinned int unmap: " + result);
                throw new RuntimeException("Error on pinned int unmap: " + result);
            }
            return value;
        }
    }

    public static void cl_transfer_buffer(long queue_ptr, long src_ptr, long dst_ptr, long size)
    {
        int result = clEnqueueCopyBuffer(queue_ptr, src_ptr, dst_ptr, 0, 0, size, null, null);
        if (result != CL_SUCCESS)
        {
            System.out.println("Error on buffer copy: " + result);
            throw new RuntimeException("Error on buffer copy: " + result);
        }
    }

    //#endregion

    //#region GL Interop

    public static long share_memory(int vboID)
    {
        try (var stack = MemoryStack.stackPush())
        {
            var status = stack.mallocInt(1);
            long ptr = clCreateFromGLBuffer(compute.context.ptr(), FLAGS_WRITE_GPU, vboID, status);
            int result = status.get(0);
            if (result != CL_SUCCESS)
            {
                System.out.println("Error on GL memory share: " + result);
                throw new RuntimeException("Error on GL memory share: " + result);
            }
            return ptr;
        }
    }

    //#endregion

    //#region Misc. Public API

    public static long build_gpu_program(List<String> src_strings)
    {
        String[] src = src_strings.toArray(new String[]{});
        return CLUtils.cl_p(compute.context.ptr(), compute.device.ptr(), src);
    }

    public static long new_mutable_buffer(int[] src)
    {
        int[] status = new int[1];
        long ptr = clCreateBuffer(compute.context.ptr(), FLAGS_READ_CPU_COPY, src, status);
        int result = status[0];
        if (result != CL_SUCCESS)
        {
            System.out.println("Error on mutable buffer creation: " + result);
            throw new RuntimeException("Error on mutable buffer creation: " + result);
        }
        return ptr;
    }

    public static long new_empty_buffer(long queue_ptr, long size)
    {
        var new_buffer_ptr = cl_new_buffer(size);
        cl_zero_buffer(queue_ptr, new_buffer_ptr, size);
        return new_buffer_ptr;
    }

    public static void cl_release_buffer(long mem_ptr)
    {
        int result = clReleaseMemObject(mem_ptr);
        if (result != CL_SUCCESS)
        {
            System.out.println("Error on buffer release: " + result);
            throw new RuntimeException("Error on buffer release: " + result);
        }
    }

    public static void cl_release_kernel(long mem_ptr)
    {
        int result = clReleaseKernel(mem_ptr);
        if (result != CL_SUCCESS)
        {
            System.out.println("Error on kernel release: " + result);
            throw new RuntimeException("Error on kernel release: " + result);
        }
    }

    public static void cl_release_buffer(ByteBuffer mem_ptr)
    {
        clSVMFree(compute.context.ptr(), mem_ptr);
    }


    public static void init(ECS ecs)
    {
        compute = GPU.CL.init_cl();
        core_memory = new GPUCoreMemory(ecs);
    }

    public static void destroy()
    {
        core_memory.release();
        compute.release();
    }

    //#endregion
}
