import org.jocl.*;

import java.util.Arrays;

import static com.controllerface.bvge.cl.OCLFunctions.readSrc;
import static org.jocl.CL.*;

public class ExclusiveScanTest
{
    private static int wx = 256;
    private static int m = wx * 2;

    static cl_context context;
    static cl_kernel k_scan_single_block;
    static cl_kernel k_scan_multi_block;
    static cl_kernel k_complete_multi_block;

    static cl_program program;
    static cl_command_queue commandQueue;
    private static final String source = readSrc("exclusive_scan.cl");

    public static void main(String[] args)
    {
        // Input array
        int size = 100_000_000;
        System.out.println("debug: " + size * Sizeof.cl_int);
        int[] input = new int[size];
        for (int i = 0; i<size; i++)
        {
            input[i] = i + 1;
        }

        // CPU based exclusive scan
        long s = System.currentTimeMillis();
        int[] t = new int[size];
        for (int i = 1; i < input.length; i ++)
        {
            t[i] = t[i - 1] + input[i - 1];
        }
        System.out.println("t1: " + (System.currentTimeMillis() - s));
        //System.out.println("CPU Output: " + Arrays.toString(t));


        // Initialize the JOCL library
        CL.setExceptionsEnabled(true);
        cl_platform_id[] platforms = new cl_platform_id[1];
        CL.clGetPlatformIDs(1, platforms, null);
        cl_platform_id platform = platforms[0];
        cl_device_id[] devices = new cl_device_id[1];
        CL.clGetDeviceIDs(platform, CL.CL_DEVICE_TYPE_GPU, 1, devices, null);
        cl_device_id device = devices[0];
        context = CL.clCreateContext(null, 1, new cl_device_id[]{device}, null, null, null);
        commandQueue = CL.clCreateCommandQueue(context, device, 0, null);

        // Create and build the kernel program
        program = CL.clCreateProgramWithSource(context, 1, new String[]{source}, null, null);
        CL.clBuildProgram(program, 0, null, null, null, null);

        // Create kernel
        k_scan_single_block = CL.clCreateKernel(program, "scan_single_block", null);
        k_scan_multi_block = CL.clCreateKernel(program, "scan_multi_block", null);
        k_complete_multi_block = CL.clCreateKernel(program, "complete_multi_block", null);


        // GPU based exclusive scan
        s = System.currentTimeMillis();
        int n = input.length;
        int k = (int) Math.ceil((float)n / (float)m);
        cl_mem d_data;
        if (k == 1)
        {
            long x = ((long)Sizeof.cl_int * (long)k * (long)m);
            d_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR, x, Pointer.to(input), null);
        }
        else
        {
            long x = ((long)Sizeof.cl_int * (long)k * (long)m);
            d_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR, x, Pointer.to(input), null);
        }
        long data_buf_size = (long)Sizeof.cl_int * n;
        Pointer dst_data = Pointer.to(input);
        scan(d_data, n);

        // transfer results into local memory.
        // Note: this is only needed once, after all operations are done
        clEnqueueReadBuffer(commandQueue, d_data, CL_TRUE, 0,
            data_buf_size, dst_data, 0, null, null);

        clReleaseMemObject(d_data);
        System.out.println("t2: " + (System.currentTimeMillis() - s));

        // Print the output
        //System.out.println("GPU Output: " + Arrays.toString(input));

        CL.clReleaseKernel(k_scan_single_block);
        CL.clReleaseProgram(program);
        CL.clReleaseCommandQueue(commandQueue);
        CL.clReleaseContext(context);

        System.out.println("Variance: " + Arrays.compare(t, input));
    }
    private static void scan(cl_mem d_data, int n)
    {
        int k = (int) Math.ceil((float)n / (float)m);
        if (k == 1)
        {
            scan_single_block(d_data, n);
        }
        else
        {
            scan_multi_block(d_data, n, k);
        }
    }

    private static void scan_single_block(cl_mem d_data, int n)
    {
        // set up buffers
        int localBufferSize = Sizeof.cl_int * m;
        Pointer src_data = Pointer.to(d_data);

        // pass in arguments
        clSetKernelArg(k_scan_single_block, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_scan_single_block, 1, localBufferSize,null);
        clSetKernelArg(k_scan_single_block, 2, Sizeof.cl_int, Pointer.to(new int[]{n}));

        // call kernel
        clEnqueueNDRangeKernel(commandQueue, k_scan_single_block, 1, null,
            new long[]{wx}, new long[]{wx}, 0, null, null);
    }

    private static void scan_multi_block(cl_mem d_data, int n, int k)
    {
        // set up buffers
        int localBufferSize = Sizeof.cl_int * m;
        int gx = k * m;
        long part_buf_size = ((long)Sizeof.cl_int * ((long)k * 2));
        int[] partial_sums = new int[k * 2];
        cl_mem p_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR, part_buf_size, Pointer.to(partial_sums), null);
        Pointer src_data = Pointer.to(d_data);
        Pointer src_part = Pointer.to(p_data);

        // pass in arguments
        clSetKernelArg(k_scan_multi_block, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_scan_multi_block, 1, localBufferSize,null);
        clSetKernelArg(k_scan_multi_block, 2, Sizeof.cl_mem, src_part);
        clSetKernelArg(k_scan_multi_block, 3, Sizeof.cl_int, Pointer.to(new int[]{n}));

        // call kernel
        clEnqueueNDRangeKernel(commandQueue, k_scan_multi_block, 1, null,
            new long[]{gx}, new long[]{wx}, 0, null, null);

        // do scan on partial/block sums
        scan(p_data, partial_sums.length);

        // pass in arguments
        clSetKernelArg(k_complete_multi_block, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_complete_multi_block, 1, localBufferSize,null);
        clSetKernelArg(k_complete_multi_block, 2, Sizeof.cl_mem, src_part);
        clSetKernelArg(k_complete_multi_block, 3, Sizeof.cl_int, Pointer.to(new int[]{n}));

        // call kernel
        clEnqueueNDRangeKernel(commandQueue, k_complete_multi_block, 1, null,
            new long[]{gx}, new long[]{wx}, 0, null, null);

        clReleaseMemObject(p_data);
    }
}
