import org.jocl.*;

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

    static cl_program program;
    static cl_command_queue commandQueue;
    private static final String source = readSrc("harris_test.cl");

    public static void main(String[] args) {
        // Input array
        int size = 512;
        int[] input = new int[size];
        for (int i = 0; i<size; i++)
        {
            input[i] = i+1;
        }

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
        int[] output = new int[n];
        //cl_mem inputBuffer = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY | CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_int * n, Pointer.to(input), null);
        //cl_mem outputBuffer = CL.clCreateBuffer(context, CL.CL_MEM_WRITE_ONLY, Sizeof.cl_int * n, null, null);



        // Create kernel
        k_scan_pad_to_pow2 = CL.clCreateKernel(program, "scan_pad_to_pow2", null);
        k_scan_subarrays = CL.clCreateKernel(program, "scan_subarrays", null);





        // todo: convert this to JOCL
        int k = (int) Math.ceil((float)n / (float)m);
        long x = ((long)Sizeof.cl_int * (long)k * (long)m);
        cl_mem d_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR, x, Pointer.to(input), null);
            //clw.dev_malloc(Sizeof.cl_int * k * m);
        //m0 += clw.memcpy_to_dev(d_data, sizeof(int)*n, data);
        recursive_scan(d_data, output, n);
        System.out.println("hey");
        //m1 += clw.memcpy_from_dev(d_data, sizeof(int)*n, data);
        //clw.dev_free(d_data);


        // todo: remove below



//        CL.clSetKernelArg(scanKernel, 0, Sizeof.cl_mem, Pointer.to(inputBuffer));
//        CL.clSetKernelArg(scanKernel, 1, Sizeof.cl_mem, Pointer.to(outputBuffer));
//        CL.clSetKernelArg(scanKernel, 2, Sizeof.cl_float * (2 * 256), null);
//        CL.clSetKernelArg(scanKernel, 3, Sizeof.cl_int, Pointer.to(new int[]{n}));
//
//        // Execute the kernel
//        long globalWorkSize[] = new long[]{n};
//        long localWorkSize[] = new long[]{Math.min(n, 256)}; // Adjust this based on device capabilities
//
//
//        CL.clEnqueueNDRangeKernel(commandQueue, scanKernel, 1, null, globalWorkSize, localWorkSize, 0, null, null);
//
//        // Read the output from the device
//        CL.clEnqueueReadBuffer(commandQueue, outputBuffer, CL.CL_TRUE, 0, Sizeof.cl_int * n, Pointer.to(output), 0, null, null);
//
//        // Perform the final scan operation
//        int carry = 0;
//        for (int i = 0; i < n; i++) {
//            output[i] = carry;
//            carry += input[i];
//        }

        // Print the output
        System.out.println("GPU Output: " + Arrays.toString(output));


        // Release resources
        CL.clReleaseMemObject(d_data);
        //CL.clReleaseMemObject(outputBuffer);
        CL.clReleaseKernel(k_scan_pad_to_pow2);
        CL.clReleaseProgram(program);
        CL.clReleaseCommandQueue(commandQueue);
        CL.clReleaseContext(context);

        int[] t = new int[n];
        for (int i = 1; i < input.length; i ++)
        {
            t[i] = t[i-1] + input[i-1];
        }
        System.out.println("CPU Output: " + Arrays.toString(t));
        //System.out.println("DEBUG: c=" + Arrays.compare(t, output));
    }


    private static void recursive_scan(cl_mem d_data, int[] output, int n)
    {
        int k = (int) Math.ceil((float)n/(float)m);
        //size of each subarray stored in local memory
        int bufsize = Sizeof.cl_int * m;
        if (k == 1)
        {
            long data_buf_size = (long)Sizeof.cl_int * n;
            Pointer dst_data = Pointer.to(output);
            Pointer src_data = Pointer.to(d_data);
            clSetKernelArg(k_scan_pad_to_pow2, 0, Sizeof.cl_mem, src_data);
            clSetKernelArg(k_scan_pad_to_pow2, 1, bufsize,null);
            clSetKernelArg(k_scan_pad_to_pow2, 2, Sizeof.cl_int, Pointer.to(new int[]{n}));
            clEnqueueNDRangeKernel(commandQueue, k_scan_pad_to_pow2, 1, null,
                new long[]{wx}, new long[]{wx}, 0, null, null);
            clEnqueueReadBuffer(commandQueue, d_data, CL_TRUE, 0,
                data_buf_size, dst_data, 0, null, null);
            System.out.println("test 1");

//            clw.kernel_arg(scan_pad_to_pow2,
//                d_data, bufsize, n);
//            k0 += clw.run_kernel_with_timing(scan_pad_to_pow2, /*dim=*/1, &wx, &wx);
        }
        else {
            int gx = k * wx;
            long data_buf_size = (long)Sizeof.cl_int * k;
            int[] nb = new int[k];
            Pointer nb_data = Pointer.to(nb);
            cl_mem d_partial = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR, data_buf_size, nb_data, null);
            Pointer dst_data = Pointer.to(output);
            Pointer src_data = Pointer.to(d_data);
            clSetKernelArg(k_scan_subarrays, 0, Sizeof.cl_mem, src_data);
            clSetKernelArg(k_scan_subarrays, 1, bufsize,null);
            clSetKernelArg(k_scan_subarrays, 2, Sizeof.cl_mem, nb_data);
            clSetKernelArg(k_scan_subarrays, 3, Sizeof.cl_int, Pointer.to(new int[]{n}));

            clEnqueueNDRangeKernel(commandQueue, k_scan_subarrays, 1, null,
                new long[]{gx}, new long[]{wx}, 0, null, null);
            clEnqueueReadBuffer(commandQueue, d_data, CL_TRUE, 0,
                data_buf_size, dst_data, 0, null, null);
            System.out.println("test 2");
            //recursive_scan(d_partial, output, k);
            //clw.dev_malloc(sizeof(int)*k);
//            clw.kernel_arg(scan_subarrays,
//                d_data, bufsize, d_partial, n);
//            k1 += clw.run_kernel_with_timing(scan_subarrays, /*dim=*/1, &gx, &wx);
//            recursive_scan(d_partial, k);
//            clw.kernel_arg(scan_inc_subarrays,
//                d_data, bufsize, d_partial, n);
//            k2 += clw.run_kernel_with_timing(scan_inc_subarrays, /*dim=*/1, &gx, &wx);
//
//            clw.dev_free(d_partial);

            CL.clReleaseMemObject(d_partial);
        }
    }
}
