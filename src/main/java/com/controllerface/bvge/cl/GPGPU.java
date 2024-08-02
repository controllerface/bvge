package com.controllerface.bvge.cl;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.editor.Editor;
import org.lwjgl.BufferUtils;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.List;

import static com.controllerface.bvge.cl.CLData.*;
import static com.controllerface.bvge.cl.CLUtils.*;
import static org.lwjgl.opencl.AMDDeviceAttributeQuery.CL_DEVICE_WAVEFRONT_WIDTH_AMD;
import static org.lwjgl.opencl.CL12.*;
import static org.lwjgl.opencl.CL12GL.clCreateFromGLBuffer;
import static org.lwjgl.opencl.CL20.*;
import static org.lwjgl.opencl.KHRGLSharing.CL_GL_CONTEXT_KHR;
import static org.lwjgl.opencl.KHRGLSharing.CL_WGL_HDC_KHR;
import static org.lwjgl.opencl.NVDeviceAttributeQuery.CL_DEVICE_WARP_SIZE_NV;
import static org.lwjgl.opengl.WGL.wglGetCurrentContext;
import static org.lwjgl.opengl.WGL.wglGetCurrentDC;

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

    /*
      These values are re-calculated at startup to match the user's hardware.
     */

    /**
     * The largest group of calculations that can be done in a single "warp" or "wave" of GPU processing.
     */
    public static long max_work_group_size = 0;

    /**
     * Used for the prefix scan kernels and their variants.
     */
    public static long max_scan_block_size = 0;

    public static long[] preferred_work_size = arg_long(0);
    public static int preferred_work_size_int = 0;

    /**
     * The max group size formatted as a single element array, making it simpler to use for Open Cl calls.
     */
    public static long[] local_work_default = arg_long(0);

    /**
     * This convenience array defines a work group size of 1, used primarily for setting up data buffers at
     * startup. Kernels of this size should be used sparingly, favor making bulk calls. However, there are
     * specific use cases where it makes sense to perform a singular operation on GPU memory.
     */
    public static final long[] global_single_size = arg_long(1);

    //#endregion

    //#region Class Variables

    /**
     * The Open CL command queue that this class uses to issue GPU commands.
     */
    public static long ptr_compute_queue;

    public static long ptr_render_queue;

    public static long ptr_sector_queue;

    /**
     * The Open CL context associated with this class.
     */
    private static long ptr_context;

    /**
     * An array of devices that support being used with Open CL. In practice, this should
     * only ever have single element, and that device should be the main GPU in the system.
     */
    private static long ptr_device_id;

    public static GPUCoreMemory core_memory;

    //#endregion

    //#region Init Methods

    private static long init_device()
    {
        // The platform, device type and device number
        // that will be used
        long deviceType = CL_DEVICE_TYPE_GPU;

        // Obtain the number of platforms
        int[] numPlatformsArray = new int[1];
        clGetPlatformIDs(null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];

        // Obtain a platform ID
        var platform_buffer = MemoryUtil.memAllocPointer(numPlatforms);
        clGetPlatformIDs(platform_buffer, (IntBuffer) null);
        var platform = platform_buffer.get();
        MemoryUtil.memFree(platform_buffer);

        // Obtain the number of devices for the platform
        int[] numDevicesArray = new int[1];
        clGetDeviceIDs(platform, deviceType, null, numDevicesArray);
        int numDevices = numDevicesArray[0];

        // Obtain a device ID
        var device_buffer = MemoryUtil.memAllocPointer(numDevices);
        clGetDeviceIDs(platform, deviceType, device_buffer, (IntBuffer) null);
        long device = device_buffer.get();
        MemoryUtil.memFree(device_buffer);

        var dc = wglGetCurrentDC();
        var ctx = wglGetCurrentContext();

        // todo: the above code is windows specific add linux code path,
        //  should look something like this:
        // var ctx = glXGetCurrentContext();
        // var dc = glXGetCurrentDrawable(); OR glfwGetX11Display();
        // contextProperties.addProperty(CL_GLX_DISPLAY_KHR, dc);

        // Create a context for the selected device
        var ctx_props_buffer = MemoryUtil.memAllocPointer(7);
        ctx_props_buffer.put(CL_CONTEXT_PLATFORM)
            .put(platform)
            .put(CL_GL_CONTEXT_KHR)
            .put(ctx)
            .put(CL_WGL_HDC_KHR)
            .put(dc)
            .put(0L)
            .flip();

        ptr_context = clCreateContext(ctx_props_buffer,
            device, null, 0L, null);

        // Create a command-queue for the selected device
        ptr_compute_queue = clCreateCommandQueue(ptr_context,
            device, 0, (IntBuffer) null);

        ptr_render_queue = clCreateCommandQueue(ptr_context,
            device, 0, (IntBuffer) null);

        ptr_sector_queue = clCreateCommandQueue(ptr_context,
            device, 0, (IntBuffer) null);

        MemoryUtil.memFree(ctx_props_buffer);

        return device;
    }

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
            long ptr = clCreateBuffer(ptr_context, FLAGS_WRITE_GPU, size, status);
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
        long ptr = clCreateBuffer(ptr_context, FLAGS_WRITE_CPU_COPY, src, status);
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
        long ptr = clCreateBuffer(ptr_context, FLAGS_READ_CPU_COPY, src, status);
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
            long ptr = clCreateBuffer(ptr_context, flags, size, status);
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
            long ptr = clCreateBuffer(ptr_context, flags, cl_int.size(), status);
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
            long ptr = clCreateBuffer(ptr_context, CL_MEM_HOST_READ_ONLY, cl_int.size(), status);
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
        return clSVMAlloc(ptr_context, CL_MEM_READ_WRITE, cl_int.size(), 0);
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
            var out = clEnqueueMapBuffer(queue_ptr,
                pinned_ptr,
                true,
                CL_MAP_READ,
                0,
                cl_int.size(),
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

    public static int work_group_count(int n)
    {
        return (int) Math.ceil((float) n / (float) max_scan_block_size);
    }

    //#endregion

    //#region GL Interop

    public static long share_memory(int vboID)
    {
        try (var stack = MemoryStack.stackPush())
        {
            var status = stack.mallocInt(1);
            long ptr = clCreateFromGLBuffer(ptr_context, FLAGS_WRITE_GPU, vboID, status);
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
        return CLUtils.cl_p(ptr_context, ptr_device_id, src);
    }

    public static long new_mutable_buffer(int[] src)
    {
        int[] status = new int[1];
        long ptr = clCreateBuffer(ptr_context, FLAGS_READ_CPU_COPY, src, status);
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
        clSVMFree(ptr_context, mem_ptr);
    }

    public static int calculate_preferred_global_size(int globalWorkSize)
    {
        int remainder = globalWorkSize % preferred_work_size_int;
        if (remainder != 0)
        {
            globalWorkSize += (preferred_work_size_int - remainder);
        }
        return globalWorkSize;
    }

    public static void init(ECS ecs)
    {
        ptr_device_id = init_device();

        System.out.println("-------- OPEN CL DEVICE -----------");
        System.out.println(get_device_string(ptr_device_id, CL_DEVICE_VENDOR));
        System.out.println(get_device_string(ptr_device_id, CL_DEVICE_NAME));
        System.out.println(get_device_string(ptr_device_id, CL_DRIVER_VERSION));
        System.out.println("-----------------------------------\n");

//        long svm_caps = get_device_long(device_id_ptr, CL_DEVICE_SVM_CAPABILITIES);
//
//        if ((svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) != 0) {
//            System.out.println("Device supports coarse-grained buffer SVM\n");
//        }K
//        if ((svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) != 0) {
//            System.out.println("Device supports fine-grained buffer SVM\n");
//        }
//        if ((svm_caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) != 0) {
//            System.out.println("Device supports fine-grained system SVM\n");
//        }
//        if ((svm_caps & CL_DEVICE_SVM_ATOMICS) != 0) {
//            System.out.println("Device supports SVM atomics\n");
//        }
//        System.out.println("SVM: " + svm_caps);

        // At runtime, local buffers are used to perform prefix scan operations.
        // It is vital that the max scan block size does not exceed the maximum
        // local buffer size of the GPU. In order to ensure this doesn't happen,
        // the following logic halves the effective max workgroup size if needed
        // to ensure that at runtime, the amount of local buffer storage requested
        // does not meet or exceed the local memory size.
        /*
         * The maximum size of a local buffer that can be used as a __local prefixed, GPU allocated
         * buffer within a kernel. Note that in practice, local memory buffers should be _less_ than
         * this value. Even though it is given a maximum, tests have shown that trying to allocate
         * exactly this amount can fail, likely due to some small amount of the local buffer being
         * used by the hardware either for individual arguments, or some other internal data.
         */
        long max_local_buffer_size = get_device_long(ptr_device_id, CL_DEVICE_LOCAL_MEM_SIZE);
        long current_max_group_size = get_device_long(ptr_device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE);
        long compute_unit_count = get_device_long(ptr_device_id, CL_DEVICE_MAX_COMPUTE_UNITS);
        long wavefront_width = get_device_long(ptr_device_id, CL_DEVICE_WAVEFRONT_WIDTH_AMD);
        long warp_width = get_device_long(ptr_device_id, CL_DEVICE_WARP_SIZE_NV);

        preferred_work_size = arg_long(wavefront_width != -1
            ? wavefront_width
            : warp_width != -1
                ? warp_width
                : 32);

        preferred_work_size_int = (int)preferred_work_size[0];

        long current_max_block_size = current_max_group_size * 2;

        long max_mem = get_device_long(ptr_device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
        long sz_char = get_device_long(ptr_device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR);
        long sz_flt = get_device_long(ptr_device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT);
        boolean non_uniform = get_device_boolean(ptr_device_id, CL_DEVICE_HOST_UNIFIED_MEMORY);

        System.out.println("CL_DEVICE_MAX_COMPUTE_UNITS: " + compute_unit_count);
        System.out.println("CL_DEVICE_WAVEFRONT_WIDTH_AMD: " + wavefront_width);
        System.out.println("CL_DEVICE_WARP_SIZE_NV: " + warp_width);

        System.out.println("CL_DEVICE_LOCAL_MEM_SIZE: " + max_local_buffer_size);
        System.out.println("CL_DEVICE_MAX_WORK_GROUP_SIZE: " + current_max_group_size);
        System.out.println("CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT: " + non_uniform);

        System.out.println("CL_DEVICE_MAX_MEM_ALLOC_SIZE: " + max_mem);
        System.out.println("preferred float: " + sz_flt);
        System.out.println("preferred char: " + sz_char);

        long int2_max = cl_int2.size() * current_max_block_size;
        long int4_max = cl_int4.size() * current_max_block_size;
        long size_cap = int2_max + int4_max;

        while (size_cap >= max_local_buffer_size)
        {
            current_max_group_size /= 2;
            current_max_block_size = current_max_group_size * 2;
            int2_max = cl_int2.size() * current_max_block_size;
            int4_max = cl_int4.size() * current_max_block_size;
            size_cap = int2_max + int4_max;
        }

        assert current_max_group_size > 0 : "Invalid Group Size";

        System.out.println("final local mem max: " + max_local_buffer_size);
        System.out.println("calculated size cap: " + size_cap);

        max_work_group_size = current_max_group_size;
        max_scan_block_size = current_max_block_size;
        local_work_default = arg_long(max_work_group_size);

        //OpenCLUtils.debugDeviceDetails(device_ids);

        core_memory = new GPUCoreMemory(ecs);
    }

    public static void destroy()
    {
        core_memory.destroy();

        clReleaseCommandQueue(ptr_compute_queue);
        clReleaseCommandQueue(ptr_render_queue);
        clReleaseCommandQueue(ptr_sector_queue);
        clReleaseContext(ptr_context);
    }

    //#endregion
}
