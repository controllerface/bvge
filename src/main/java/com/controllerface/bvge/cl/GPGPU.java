package com.controllerface.bvge.cl;

import com.controllerface.bvge.cl.kernels.*;
import com.controllerface.bvge.cl.programs.*;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.List;

import static com.controllerface.bvge.cl.CLUtils.*;
import static org.lwjgl.opencl.CL12.*;
import static org.lwjgl.opencl.CL12GL.clCreateFromGLBuffer;
import static org.lwjgl.opencl.CL20.*;
import static org.lwjgl.opencl.KHRGLSharing.CL_GL_CONTEXT_KHR;
import static org.lwjgl.opencl.KHRGLSharing.CL_WGL_HDC_KHR;
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
    private static final ByteBuffer ZERO_PATTERN_BUFFER = MemoryUtil.memCalloc(1);

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
    public static long cl_cmd_queue_ptr;

    public static long gl_cmd_queue_ptr;

    /**
     * The Open CL context associated with this class.
     */
    private static long context_ptr;

    /**
     * An array of devices that support being used with Open CL. In practice, this should
     * only ever have single element, and that device should be the main GPU in the system.
     */
    private static long device_id_ptr;

    public static GPUCoreMemory core_memory;

    //#endregion

    //#region Program Objects

    private enum Program
    {
        scan_int2_array(new ScanInt2Array()),
        scan_int4_array(new ScanInt4Array()),
        scan_int_array(new ScanIntArray()),
        scan_int_array_out(new ScanIntArrayOut()),

        ;

        public final GPUProgram gpu;

        Program(GPUProgram program)
        {
            this.gpu = program;
        }
    }

    //#endregion

    //#region Kernel Objects

    private static GPUKernel scan_int_single_block_k;
    private static GPUKernel scan_int_multi_block_k;
    private static GPUKernel complete_int_multi_block_k;
    private static GPUKernel scan_int2_single_block_k;
    private static GPUKernel scan_int2_multi_block_k;
    private static GPUKernel complete_int2_multi_block_k;
    private static GPUKernel scan_int4_single_block_k;
    private static GPUKernel scan_int4_multi_block_k;
    private static GPUKernel complete_int4_multi_block_k;
    private static GPUKernel scan_int_single_block_out_k;
    private static GPUKernel scan_int_multi_block_out_k;
    private static GPUKernel complete_int_multi_block_out_k;

    //#endregion

    //#region Init Methods

    private static long init_device()
    {
        // TODO: may need some updates for cases where there's more than one possible device

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

        context_ptr = clCreateContext(ctx_props_buffer,
            device, null, 0L, null);

        // Create a command-queue for the selected device
        cl_cmd_queue_ptr = clCreateCommandQueue(context_ptr,
            device, 0, (IntBuffer) null);

        gl_cmd_queue_ptr = clCreateCommandQueue(context_ptr,
            device, 0, (IntBuffer) null);

        MemoryUtil.memFree(ctx_props_buffer);

        return device;
    }

    private static void init_memory()
    {
        core_memory = new GPUCoreMemory();
    }

    /**
     * Creates reusable GPUKernel objects that the individual API methods use to implement
     * the CPU-GPU transition layer. Pre-generating kernels this way helps to reduce calls
     * to set kernel arguments, which can be expensive. Where possible, kernel arguments
     * can be set once, and then subsequent calls to that kernel do not require setting
     * the argument again. Only arguments with data that changes need to be updated.
     * Generally, kernel functions operate on large arrays of data, which can be defined
     * as arguments only once, even if the contents of these arrays changes often.
     */
    private static void init_kernels()
    {
        // integer exclusive scan in-place

        long scan_int_array_single_ptr = Program.scan_int_array.gpu.kernel_ptr(Kernel.scan_int_single_block);
        long scan_int_array_multi_ptr = Program.scan_int_array.gpu.kernel_ptr(Kernel.scan_int_multi_block);
        long scan_int_array_comp_ptr = Program.scan_int_array.gpu.kernel_ptr(Kernel.complete_int_multi_block);
        scan_int_single_block_k = new ScanIntSingleBlock_k(cl_cmd_queue_ptr, scan_int_array_single_ptr);
        scan_int_multi_block_k = new ScanIntMultiBlock_k(cl_cmd_queue_ptr, scan_int_array_multi_ptr);
        complete_int_multi_block_k = new CompleteIntMultiBlock_k(cl_cmd_queue_ptr, scan_int_array_comp_ptr);

        // 2D vector integer exclusive scan in-place

        long scan_int2_array_single_ptr = Program.scan_int2_array.gpu.kernel_ptr(Kernel.scan_int2_single_block);
        long scan_int2_array_multi_ptr = Program.scan_int2_array.gpu.kernel_ptr(Kernel.scan_int2_multi_block);
        long scan_int2_array_comp_ptr = Program.scan_int2_array.gpu.kernel_ptr(Kernel.complete_int2_multi_block);
        scan_int2_single_block_k = new ScanInt2SingleBlock_k(cl_cmd_queue_ptr, scan_int2_array_single_ptr);
        scan_int2_multi_block_k = new ScanInt2MultiBlock_k(cl_cmd_queue_ptr, scan_int2_array_multi_ptr);
        complete_int2_multi_block_k = new CompleteInt2MultiBlock_k(cl_cmd_queue_ptr, scan_int2_array_comp_ptr);

        // 4D vector integer exclusive scan in-place

        long scan_int4_array_single_ptr = Program.scan_int4_array.gpu.kernel_ptr(Kernel.scan_int4_single_block);
        long scan_int4_array_multi_ptr = Program.scan_int4_array.gpu.kernel_ptr(Kernel.scan_int4_multi_block);
        long scan_int4_array_comp_ptr = Program.scan_int4_array.gpu.kernel_ptr(Kernel.complete_int4_multi_block);
        scan_int4_single_block_k = new ScanInt4SingleBlock_k(cl_cmd_queue_ptr, scan_int4_array_single_ptr);
        scan_int4_multi_block_k = new ScanInt4MultiBlock_k(cl_cmd_queue_ptr, scan_int4_array_multi_ptr);
        complete_int4_multi_block_k = new CompleteInt4MultiBlock_k(cl_cmd_queue_ptr, scan_int4_array_comp_ptr);

        // integer exclusive scan to output buffer

        long scan_int_array_out_single_ptr = Program.scan_int_array_out.gpu.kernel_ptr(Kernel.scan_int_single_block_out);
        long scan_int_array_out_multi_ptr = Program.scan_int_array_out.gpu.kernel_ptr(Kernel.scan_int_multi_block_out);
        long scan_int_array_out_comp_ptr = Program.scan_int_array_out.gpu.kernel_ptr(Kernel.complete_int_multi_block_out);
        scan_int_single_block_out_k = new ScanIntSingleBlockOut_k(cl_cmd_queue_ptr, scan_int_array_out_single_ptr);
        scan_int_multi_block_out_k = new ScanIntMultiBlockOut_k(cl_cmd_queue_ptr, scan_int_array_out_multi_ptr);
        complete_int_multi_block_out_k = new CompleteIntMultiBlockOut_k(cl_cmd_queue_ptr, scan_int_array_out_comp_ptr);
    }

    //#endregion

    //#region Utility Methods

    public static void cl_read_int_buffer(long queue_ptr, long src_ptr, int[] dst)
    {
        clEnqueueReadBuffer(queue_ptr,
            src_ptr,
            true,
            0,
            dst,
            null,
            null);
    }

    public static void cl_read_float_buffer(long queue_ptr, long src_ptr, float[] dst)
    {
        clEnqueueReadBuffer(queue_ptr,
            src_ptr,
            true,
            0,
            dst,
            null,
            null);
    }

    public static long cl_new_buffer(long size)
    {
        return clCreateBuffer(context_ptr, FLAGS_WRITE_GPU, size, null);
    }

    public static long cl_new_int_arg_buffer(int[] src)
    {
        return clCreateBuffer(context_ptr, FLAGS_WRITE_CPU_COPY, src, null);
    }

    public static long cl_new_cpu_copy_buffer(float[] src)
    {
        return clCreateBuffer(context_ptr, FLAGS_READ_CPU_COPY, src, null);
    }

    public static void cl_zero_buffer(long queue_ptr, long buffer_ptr, long buffer_size)
    {
        clEnqueueFillBuffer(queue_ptr,
            buffer_ptr,
            ZERO_PATTERN_BUFFER,
            0,
            buffer_size,
            null,
            null);
    }

    public static void cl_zero_buffer(long queue_ptr, ByteBuffer buffer_ptr, long buffer_size)
    {
        clEnqueueSVMMemFill(queue_ptr, buffer_ptr, ZERO_PATTERN_BUFFER, null, null);
    }

    public static long cl_new_pinned_buffer(long size)
    {
        long flags = CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;
        return clCreateBuffer(context_ptr, flags, size, null);
    }

    public static int[] cl_read_pinned_int_buffer(long queue_ptr, long pinned_ptr, long size, int count)
    {
        var out = clEnqueueMapBuffer(queue_ptr,
            pinned_ptr,
            true,
            CL_MAP_READ,
            0,
            size,
            null,
            null,
            (IntBuffer) null,
            null);

        assert out != null;

        int[] result = new int[count];
        var int_buffer = out.order(ByteOrder.LITTLE_ENDIAN).asIntBuffer();
        for (int i = 0; i < count; i++)
        {
            result[i] = int_buffer.get(i);
        }
        clEnqueueUnmapMemObject(queue_ptr, pinned_ptr, out, null, null);
        return result;
    }

    public static float[] cl_read_pinned_float_buffer(long queue_ptr, long pinned_ptr, long size, int count)
    {
        var out = clEnqueueMapBuffer(queue_ptr,
            pinned_ptr,
            true,
            CL_MAP_READ,
            0,
            size,
            null,
            null,
            (IntBuffer) null,
            null);

        assert out != null;

        float[] result = new float[count];
        var float_buffer = out.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        for (int i = 0; i < count; i++)
        {
            result[i] = float_buffer.get(i);
        }
        clEnqueueUnmapMemObject(queue_ptr, pinned_ptr, out, null, null);
        return result;
    }

    public static long cl_new_pinned_int()
    {
        long flags = CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;
        return clCreateBuffer(context_ptr, flags, CLSize.cl_int, null);
    }

    public static long cl_new_unpinned_int()
    {
        long flags = CL_MEM_HOST_READ_ONLY;
        return clCreateBuffer(context_ptr, flags, CLSize.cl_int, null);
    }

    public static ByteBuffer cl_new_svm_int()
    {
        long flags = CL_MEM_READ_WRITE;
        return clSVMAlloc(context_ptr, flags, CLSize.cl_int, 0);
    }
    public static ByteBuffer cl_new_svm_buffer(long size)
    {
        long flags = CL_MEM_READ_WRITE;
        return clSVMAlloc(context_ptr, flags, size, 0);
    }

    public static void cl_read_svm_int_buffer(long queue_ptr, ByteBuffer svm_buffer, int[] dst)
    {
        int result = clEnqueueSVMMap(queue_ptr, true, CL_MAP_READ, svm_buffer, null, null);
        if (result != CL_SUCCESS)
        {
            System.out.println("Error on scm buffer creation: " + result);
            System.exit(1);
        }
        for (int i = 0; i < dst.length; i++)
        {
            dst[i] = svm_buffer.getInt(i);
        }
        clEnqueueSVMUnmap(queue_ptr, svm_buffer, null, null);
    }

    public static void cl_read_svm_float_buffer(long queue_ptr, ByteBuffer svm_buffer, float[] dst)
    {
        int result = clEnqueueSVMMap(queue_ptr, true, CL_MAP_READ, svm_buffer, null, null);
        if (result != CL_SUCCESS)
        {
            System.out.println("Error on scm buffer creation: " + result);
            System.exit(1);
        }
        for (int i = 0; i < dst.length; i++)
        {
            dst[i] = svm_buffer.getFloat(i);
        }
        clEnqueueSVMUnmap(queue_ptr, svm_buffer, null, null);
    }

    public static int cl_read_svm_int(long queue_ptr, ByteBuffer svm_buffer)
    {
        int result = clEnqueueSVMMap(queue_ptr, true, CL_MAP_READ, svm_buffer, null, null);
        if (result != CL_SUCCESS)
        {
            System.out.println("Error on scm buffer creation: " + result);
            System.exit(1);
        }
        int v = svm_buffer.getInt(0);
        clEnqueueSVMUnmap(queue_ptr, svm_buffer, null, null);
        return v;
    }

    public static int cl_read_unpinned_int(long queue_ptr, long pinned_ptr)
    {
        try(var stack = MemoryStack.stackPush())
        {
            var pb = stack.mallocInt(1);
            clEnqueueReadBuffer(queue_ptr, pinned_ptr, true, 0, pb, null, null);
            return pb.get(0);
        }
    }

    public static int cl_read_pinned_int(long queue_ptr, long pinned_ptr)
    {
        var out = clEnqueueMapBuffer(queue_ptr,
            pinned_ptr,
            true,
            CL_MAP_READ,
            0,
            CLSize.cl_int,
            null,
            null,
            (IntBuffer) null,
            null);

        assert out != null;

        int result = out.order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(0);
        clEnqueueUnmapMemObject(queue_ptr, pinned_ptr, out, null, null);
        return result;
    }

    public static void cl_transfer_buffer(long queue_ptr, long src_ptr, long dst_ptr, long size)
    {
        int result = clEnqueueCopyBuffer(queue_ptr, src_ptr, dst_ptr, 0, 0, size, null, null);
        if (result != CL_SUCCESS)
        {
            System.out.println("Error on buffer copy: " + result);
            System.exit(1);
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
        return clCreateFromGLBuffer(context_ptr, FLAGS_WRITE_GPU, vboID, (IntBuffer) null);
    }

    //#endregion

    //#region Exclusive scan variants

    public static void scan_int(long data_ptr, int n)
    {
        int k = work_group_count(n);
        if (k == 1)
        {
            scan_single_block_int(data_ptr, n);
        }
        else
        {
            scan_multi_block_int(data_ptr, n, k);
        }
    }

    public static void scan_int2(long data_ptr, int n)
    {
        int k = work_group_count(n);
        if (k == 1)
        {
            scan_single_block_int2(data_ptr, n);
        }
        else
        {
            scan_multi_block_int2(data_ptr, n, k);
        }
    }

    public static void scan_int4(long data_ptr, int n)
    {
        int k = work_group_count(n);
        if (k == 1)
        {
            scan_single_block_int4(data_ptr, n);
        }
        else
        {
            scan_multi_block_int4(data_ptr, n, k);
        }
    }

    public static void scan_int_out(long data_ptr, long o_data_ptr, int n)
    {
        int k = work_group_count(n);
        if (k == 1)
        {
            scan_single_block_int_out(data_ptr, o_data_ptr, n);
        }
        else
        {
            scan_multi_block_int_out(data_ptr, o_data_ptr, n, k);
        }
    }

    private static void scan_single_block_int(long data_ptr, int n)
    {
        long local_buffer_size = CLSize.cl_int * max_scan_block_size;

        scan_int_single_block_k
            .ptr_arg(ScanIntSingleBlock_k.Args.data, data_ptr)
            .loc_arg(ScanIntSingleBlock_k.Args.buffer, local_buffer_size)
            .set_arg(ScanIntSingleBlock_k.Args.n, n)
            .call(local_work_default, local_work_default);
    }

    private static void scan_multi_block_int(long data_ptr, int n, int k)
    {
        long local_buffer_size = CLSize.cl_int * max_scan_block_size;
        long gx = k * max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) CLSize.cl_int * ((long) part_size));

        var part_data = cl_new_buffer(part_buf_size);

        scan_int_multi_block_k
            .ptr_arg(ScanIntMultiBlock_k.Args.data, data_ptr)
            .loc_arg(ScanIntMultiBlock_k.Args.buffer, local_buffer_size)
            .ptr_arg(ScanIntMultiBlock_k.Args.part, part_data)
            .set_arg(ScanIntMultiBlock_k.Args.n, n)
            .call(global_work_size, local_work_default);

        scan_int(part_data, part_size);

        complete_int_multi_block_k
            .ptr_arg(CompleteIntMultiBlock_k.Args.data, data_ptr)
            .loc_arg(CompleteIntMultiBlock_k.Args.buffer, local_buffer_size)
            .ptr_arg(CompleteIntMultiBlock_k.Args.part, part_data)
            .set_arg(CompleteIntMultiBlock_k.Args.n, n)
            .call(global_work_size, local_work_default);

        cl_release_buffer(part_data);
    }

    private static void scan_single_block_int2(long data_ptr, int n)
    {
        long local_buffer_size = CLSize.cl_int2 * max_scan_block_size;

       scan_int2_single_block_k
            .ptr_arg(ScanInt2SingleBlock_k.Args.data, data_ptr)
            .loc_arg(ScanInt2SingleBlock_k.Args.buffer, local_buffer_size)
            .set_arg(ScanInt2SingleBlock_k.Args.n, n)
            .call(local_work_default, local_work_default);
    }

    private static void scan_multi_block_int2(long data_ptr, int n, int k)
    {
        long local_buffer_size = CLSize.cl_int2 * max_scan_block_size;
        long gx = k * max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) CLSize.cl_int2 * ((long) part_size));

        var part_data = cl_new_buffer(part_buf_size);

        scan_int2_multi_block_k
            .ptr_arg(ScanInt2MultiBlock_k.Args.data, data_ptr)
            .loc_arg(ScanInt2MultiBlock_k.Args.buffer, local_buffer_size)
            .ptr_arg(ScanInt2MultiBlock_k.Args.part, part_data)
            .set_arg(ScanInt2MultiBlock_k.Args.n, n)
            .call(global_work_size, local_work_default);

        scan_int2(part_data, part_size);

        complete_int2_multi_block_k
            .ptr_arg(CompleteInt2MultiBlock_k.Args.data, data_ptr)
            .loc_arg(CompleteInt2MultiBlock_k.Args.buffer, local_buffer_size)
            .ptr_arg(CompleteInt2MultiBlock_k.Args.part, part_data)
            .set_arg(CompleteInt2MultiBlock_k.Args.n, n)
            .call(global_work_size, local_work_default);

        cl_release_buffer(part_data);
    }

    private static void scan_single_block_int4(long data_ptr, int n)
    {
        long local_buffer_size = CLSize.cl_int4 * max_scan_block_size;

        scan_int4_single_block_k
            .ptr_arg(ScanInt4SingleBlock_k.Args.data, data_ptr)
            .loc_arg(ScanInt4SingleBlock_k.Args.buffer, local_buffer_size)
            .set_arg(ScanInt4SingleBlock_k.Args.n, n)
            .call(local_work_default, local_work_default);
    }

    private static void scan_multi_block_int4(long data_ptr, int n, int k)
    {
        long local_buffer_size = CLSize.cl_int4 * max_scan_block_size;
        long gx = k * max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) CLSize.cl_int4 * ((long) part_size));

        var part_data = cl_new_buffer(part_buf_size);

        scan_int4_multi_block_k
            .ptr_arg(ScanInt4MultiBlock_k.Args.data, data_ptr)
            .loc_arg(ScanInt4MultiBlock_k.Args.buffer, local_buffer_size)
            .ptr_arg(ScanInt4MultiBlock_k.Args.part, part_data)
            .set_arg(ScanInt4MultiBlock_k.Args.n, n)
            .call(global_work_size, local_work_default);

        scan_int4(part_data, part_size);

        complete_int4_multi_block_k
            .ptr_arg(CompleteInt4MultiBlock_k.Args.data, data_ptr)
            .loc_arg(CompleteInt4MultiBlock_k.Args.buffer, local_buffer_size)
            .ptr_arg(CompleteInt4MultiBlock_k.Args.part, part_data)
            .set_arg(CompleteInt4MultiBlock_k.Args.n, n)
            .call(global_work_size, local_work_default);

        cl_release_buffer(part_data);
    }

    private static void scan_single_block_int_out(long data_ptr, long o_data_ptr, int n)
    {
        long local_buffer_size = CLSize.cl_int * max_scan_block_size;

        scan_int_single_block_out_k
            .ptr_arg(ScanIntSingleBlockOut_k.Args.input, data_ptr)
            .ptr_arg(ScanIntSingleBlockOut_k.Args.output, o_data_ptr)
            .loc_arg(ScanIntSingleBlockOut_k.Args.buffer, local_buffer_size)
            .set_arg(ScanIntSingleBlockOut_k.Args.n, n)
            .call(local_work_default, local_work_default);
    }

    private static void scan_multi_block_int_out(long data_ptr, long o_data_ptr, int n, int k)
    {
        long local_buffer_size = CLSize.cl_int * max_scan_block_size;
        long gx = k * max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) CLSize.cl_int * ((long) part_size));
        var part_data = cl_new_buffer(part_buf_size);

        scan_int_multi_block_out_k
            .ptr_arg(ScanIntMultiBlockOut_k.Args.input, data_ptr)
            .ptr_arg(ScanIntMultiBlockOut_k.Args.output, o_data_ptr)
            .loc_arg(ScanIntMultiBlockOut_k.Args.buffer, local_buffer_size)
            .ptr_arg(ScanIntMultiBlockOut_k.Args.part, part_data)
            .set_arg(ScanIntMultiBlockOut_k.Args.n, n)
            .call(global_work_size, local_work_default);

        scan_int(part_data, part_size);

        complete_int_multi_block_out_k
            .ptr_arg(CompleteIntMultiBlockOut_k.Args.output, o_data_ptr)
            .loc_arg(CompleteIntMultiBlockOut_k.Args.buffer, local_buffer_size)
            .ptr_arg(CompleteIntMultiBlockOut_k.Args.part, part_data)
            .set_arg(CompleteIntMultiBlockOut_k.Args.n, n)
            .call(global_work_size, local_work_default);

        cl_release_buffer(part_data);
    }

    //#endregion

    //#region Misc. Public API

    public static long build_gpu_program(List<String> src_strings)
    {
        String[] src = src_strings.toArray(new String[]{});
        return CLUtils.cl_p(context_ptr, device_id_ptr, src);
    }

    public static long new_mutable_buffer(int[] src)
    {
        return clCreateBuffer(context_ptr, FLAGS_READ_CPU_COPY, src, null);
    }

    public static long new_empty_buffer(long queue_ptr, long size)
    {
        var new_buffer_ptr = cl_new_buffer(size);
        cl_zero_buffer(queue_ptr, new_buffer_ptr, size);
        return new_buffer_ptr;
    }

    public static void cl_release_buffer(long mem_ptr)
    {
        clReleaseMemObject(mem_ptr);
    }

    public static void cl_release_buffer(ByteBuffer mem_ptr)
    {
        clSVMFree(context_ptr, mem_ptr);
    }

    public static void init()
    {
        device_id_ptr = init_device();

        System.out.println("-------- OPEN CL DEVICE -----------");
        System.out.println(get_device_string(device_id_ptr, CL_DEVICE_VENDOR));
        System.out.println(get_device_string(device_id_ptr, CL_DEVICE_NAME));
        System.out.println(get_device_string(device_id_ptr, CL_DRIVER_VERSION));
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
        // the following logic halves the effective max workgroup size, if needed
        // to ensure that at runtime, the amount of local buffer storage requested
        // does not meet or exceed the local memory size.
        /*
         * The maximum size of a local buffer that can be used as a __local prefixed, GPU allocated
         * buffer within a kernel. Note that in practice, local memory buffers should be _less_ than
         * this value. Even though it is given a maximum, tests have shown that trying to allocate
         * exactly this amount can fail, likely due to some small amount of the local buffer being
         * used by the hardware either for individual arguments, or some other internal data.
         */
        long max_local_buffer_size = get_device_long(device_id_ptr, CL_DEVICE_LOCAL_MEM_SIZE);
        long current_max_group_size = get_device_long(device_id_ptr, CL_DEVICE_MAX_WORK_GROUP_SIZE);
        long current_max_block_size = current_max_group_size * 2;

        long max_mem = get_device_long(device_id_ptr, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
        long sz_char = get_device_long(device_id_ptr, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR);
        long sz_flt = get_device_long(device_id_ptr, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT);

        System.out.println("max mem: " + max_mem);
        System.out.println("preferred float: " + sz_flt);
        System.out.println("preferred char: " + sz_char);

        long int2_max = CLSize.cl_int2 * current_max_block_size;
        long int4_max = CLSize.cl_int4 * current_max_block_size;
        long size_cap = int2_max + int4_max;

        while (size_cap >= max_local_buffer_size)
        {
            current_max_group_size /= 2;
            current_max_block_size = current_max_group_size * 2;
            int2_max = CLSize.cl_int2 * current_max_block_size;
            int4_max = CLSize.cl_int4 * current_max_block_size;
            size_cap = int2_max + int4_max;
        }

        assert current_max_group_size > 0 : "Invalid Group Size";

        max_work_group_size = current_max_group_size;
        max_scan_block_size = current_max_block_size;
        local_work_default = arg_long(max_work_group_size);

        // initialize gpu programs
        for (var program : Program.values())
        {
            program.gpu.init();
        }

        //OpenCLUtils.debugDeviceDetails(device_ids);

        // create memory buffers
        init_memory();

        // Create re-usable kernel objects
        init_kernels();
    }

    public static void destroy()
    {
        for (Program program : Program.values())
        {
            if (program.gpu != null) program.gpu.destroy();
        }

        core_memory.destroy();

        clReleaseCommandQueue(cl_cmd_queue_ptr);
        clReleaseCommandQueue(gl_cmd_queue_ptr);
        clReleaseContext(context_ptr);
        MemoryUtil.memFree(ZERO_PATTERN_BUFFER);
    }

    //#endregion
}
