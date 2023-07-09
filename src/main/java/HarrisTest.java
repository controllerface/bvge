import org.jocl.*;

import java.nio.IntBuffer;
import java.util.Arrays;

import static com.controllerface.bvge.cl.OCLFunctions.readSrc;
import static org.jocl.CL.*;
import static org.jocl.CL.CL_TRUE;

public class HarrisTest
{
    private static int wx = 256;
    private static int m = wx * 2;

    static cl_context context;
    static cl_kernel k_scan_pad_to_pow2;
    static cl_kernel k_scan_subarrays;
    static cl_kernel k_scan_inc_subarrays;


    static cl_program program;
    static cl_command_queue commandQueue;
    private static final String source = readSrc("harris_test.cl");

    public static void main(String[] args)
    {
        // Input array
        int size = 512 + 2;
        int[] input = new int[size];
        for (int i = 0; i<size; i++)
        {
            input[i] = i+1;
        }

        int[] t = new int[size];
        for (int i = 1; i < input.length; i ++)
        {
            t[i] = t[i-1] + input[i-1];
        }
        System.out.println("CPU Output: " + Arrays.toString(t));



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

        // Create input and output buffers
        int n = input.length;
        //int[] output = new int[n];
        //cl_mem inputBuffer = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY | CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_int * n, Pointer.to(input), null);
        //cl_mem outputBuffer = CL.clCreateBuffer(context, CL.CL_MEM_WRITE_ONLY, Sizeof.cl_int * n, null, null);


        // Create kernel
        k_scan_pad_to_pow2 = CL.clCreateKernel(program, "scan_pad_to_pow2", null);
        k_scan_subarrays = CL.clCreateKernel(program, "scan_subarrays", null);
        k_scan_inc_subarrays = CL.clCreateKernel(program, "scan_inc_subarrays", null);

        int k = (int) Math.ceil((float)n / (float)m);
        long x = ((long)Sizeof.cl_int * (long)k * (long)m);
        cl_mem d_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR, x, Pointer.to(input), null);
        recursive_scan(d_data, input, n);

        // Print the output
        System.out.println("GPU Output: " + Arrays.toString(input));


        // Release resources
        CL.clReleaseMemObject(d_data);
        //CL.clReleaseMemObject(outputBuffer);
        CL.clReleaseKernel(k_scan_pad_to_pow2);
        CL.clReleaseProgram(program);
        CL.clReleaseCommandQueue(commandQueue);
        CL.clReleaseContext(context);


        System.out.println("Variance: " + Arrays.compare(t, input));
    }


    private static void recursive_scan(cl_mem d_data, int[] output, int n)
    {
        int k = (int) Math.ceil((float)n/(float)m);
        var outputBuffer = IntBuffer.wrap(output);
        //size of each subarray stored in local memory
        int localBufferSize = Sizeof.cl_int * m;
        if (k == 1)
        {
            long data_buf_size = (long)Sizeof.cl_int * n;
            Pointer dst_data = Pointer.to(outputBuffer);
            Pointer src_data = Pointer.to(d_data);
            clSetKernelArg(k_scan_pad_to_pow2, 0, Sizeof.cl_mem, src_data);
            clSetKernelArg(k_scan_pad_to_pow2, 1, localBufferSize,null);
            clSetKernelArg(k_scan_pad_to_pow2, 2, Sizeof.cl_int, Pointer.to(new int[]{n}));
            clEnqueueNDRangeKernel(commandQueue, k_scan_pad_to_pow2, 1, null,
                new long[]{wx}, new long[]{wx}, 0, null, null);
            clEnqueueReadBuffer(commandQueue, d_data, CL_TRUE, 0,
                data_buf_size, dst_data, 0, null, null);
        }
        else {
            int gx = k * wx;
            long data_buf_size = (long)Sizeof.cl_int * gx;
            long part_buf_size = (long)Sizeof.cl_int * k;
            int[] partial_bytes = new int[(int)part_buf_size];
            Pointer dst_part = Pointer.to(partial_bytes);
            cl_mem d_partial = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR, part_buf_size, dst_part, null);
            Pointer src_part = Pointer.to(d_partial);
            //clw.dev_malloc(sizeof(int)*k);

            Pointer dst_data = Pointer.to(outputBuffer);
            Pointer src_data = Pointer.to(d_data);
            clSetKernelArg(k_scan_subarrays, 0, Sizeof.cl_mem, src_data);
            clSetKernelArg(k_scan_subarrays, 1, localBufferSize,null);
            clSetKernelArg(k_scan_subarrays, 2, Sizeof.cl_mem, src_part);
            clSetKernelArg(k_scan_subarrays, 3, Sizeof.cl_int, Pointer.to(new int[]{n}));

            clEnqueueNDRangeKernel(commandQueue, k_scan_subarrays, 1, null,
                new long[]{gx}, new long[]{wx}, 0, null, null);
            clEnqueueReadBuffer(commandQueue, d_data, CL_TRUE, 0,
                data_buf_size, dst_data, 0, null, null);
            clEnqueueReadBuffer(commandQueue, d_partial, CL_TRUE, 0,
                part_buf_size, dst_part, 0, null, null);
//            clw.kernel_arg(scan_subarrays,
//                d_data, bufsize, d_partial, n);
//            k1 += clw.run_kernel_with_timing(scan_subarrays, /*dim=*/1, &gx, &wx);

            var remaining = n - gx;
            var nextOffset = output.length - remaining;
            //output[nextOffset] = nb[0];
            //System.arraycopy(output, nextOffset, nb, 1, Math.min(remaining, nb.length-1));

            //System.out.println("debug no inner: " + nextOffset + " :: " + output[nextOffset]);

            recursive_scan(d_partial, output, k);
//            recursive_scan(d_partial, k);


            clSetKernelArg(k_scan_inc_subarrays, 0, Sizeof.cl_mem, src_data);
            clSetKernelArg(k_scan_inc_subarrays, 1, localBufferSize,null);
            clSetKernelArg(k_scan_inc_subarrays, 2, Sizeof.cl_mem, src_part);
            clSetKernelArg(k_scan_inc_subarrays, 3, Sizeof.cl_int, Pointer.to(new int[]{n}));

            clEnqueueNDRangeKernel(commandQueue, k_scan_inc_subarrays, 1, null,
                new long[]{gx}, new long[]{wx}, 0, null, null);
            clEnqueueReadBuffer(commandQueue, d_data, CL_TRUE, 0,
                data_buf_size, dst_data, 0, null, null);
            clEnqueueReadBuffer(commandQueue, d_partial, CL_TRUE, 0,
                part_buf_size, dst_part, 0, null, null);

//            clw.kernel_arg(scan_inc_subarrays,
//                d_data, bufsize, d_partial, n);
//            k2 += clw.run_kernel_with_timing(scan_inc_subarrays, /*dim=*/1, &gx, &wx);
//
//            clw.dev_free(d_partial);

            CL.clReleaseMemObject(d_partial);
        }
    }
}
