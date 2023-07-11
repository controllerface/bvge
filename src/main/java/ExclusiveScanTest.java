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

//    static cl_kernel k_scan_pad_to_pow2;
//    static cl_kernel k_scan_subarrays;
//    static cl_kernel k_scan_inc_subarrays;

    static cl_program program;
    static cl_command_queue commandQueue;
    private static final String source = readSrc("exclusive_scan.cl");

    public static void main(String[] args)
    {
        // Input array
        int size = 512;
        System.out.println("debug: " + size * Sizeof.cl_int);
        int[] input = new int[size];
        for (int i = 0; i<size; i++)
        {
            input[i] = i+1;
        }

        long s = System.nanoTime();
        int[] t = new int[size];
        for (int i = 1; i < input.length; i ++)
        {
            t[i] = t[i-1] + input[i-1];
        }
        System.out.println("t1: " + (System.nanoTime() - s));
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

        //int[] output = new int[n];
        //cl_mem inputBuffer = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY | CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_int * n, Pointer.to(input), null);
        //cl_mem outputBuffer = CL.clCreateBuffer(context, CL.CL_MEM_WRITE_ONLY, Sizeof.cl_int * n, null, null);


        // Create kernel
        k_scan_single_block = CL.clCreateKernel(program, "scan_single_block", null);
        //k_scan_subarrays = CL.clCreateKernel(program, "scan_subarrays", null);
        //k_scan_inc_subarrays = CL.clCreateKernel(program, "scan_inc_subarrays", null);

        s = System.nanoTime();
        scan(input);
        System.out.println("t2: " + (System.nanoTime() - s));

//        int k = (int) Math.ceil((float)n / (float)m);
//        long x = ((long)Sizeof.cl_int * (long)k * (long)m);
//        cl_mem d_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR, x, Pointer.to(input), null);
//        recursive_scan(d_data, input, n);

        // Print the output
        System.out.println("GPU Output: " + Arrays.toString(input));


        // Release resources
//        CL.clReleaseMemObject(d_data);
        //CL.clReleaseMemObject(outputBuffer);
        CL.clReleaseKernel(k_scan_single_block);
        CL.clReleaseProgram(program);
        CL.clReleaseCommandQueue(commandQueue);
        CL.clReleaseContext(context);


        System.out.println("Variance: " + Arrays.compare(t, input));
    }
    private static void scan(int[] input)
    {
        int n = input.length;
        int k = (int) Math.ceil((float)n / (float)m);
        long x = ((long)Sizeof.cl_int * (long)k * (long)m);
        cl_mem d_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR, x, Pointer.to(input), null);

        if (n <= m)
        {
            scan_single_block(d_data, input, n);
        }
//        else
//        {
//            int remaining = n;
//            int nextOffset = 0;
//            int nxint_n = -1;
//            int lcount = 0;
//            while (remaining > 0)
//            {
//                int next_chunk_size = Math.min(m, remaining);
//                if (next_chunk_size != remaining)
//                {
//                    lcount++;
//                    // in order for the chunk scan to work, the block size must be one more than
//                    // the maximum. In addition, when we are on an intermediate chunk, we have to scan
//                    // backward one entry. When this occurs, we must add back 1 to the remaining count,
//                    // to allow for the one entry overage we lost from accounting for the last value.
//                    int padding = nxint_n == -1 ? 0 : 1;
//                    int next_extra = next_chunk_size + 1;
//                    int[] next_chunk = new int[next_extra];
//                    System.arraycopy(input, nextOffset - padding, next_chunk, 0, next_extra);
//                    if (padding !=0)
//                    {
//                        next_chunk[0] = nxint_n;
//                    }
//                    long x2 = ((long)Sizeof.cl_int * (long)next_extra);
//                    cl_mem p_data2 = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR, x2, Pointer.to(next_chunk), null);
//                    int nxint = scan_chunk(p_data2, next_chunk, next_extra);
//                    System.arraycopy(next_chunk, padding, input, nextOffset, next_chunk_size);
//                    //input[next_chunk_size] = nxint;
//                    nxint_n = nxint;
//                    // todo: add the final output of scan chunk to the next index
//                    remaining -= next_chunk_size - padding;
//                    nextOffset += next_chunk_size - padding;
//                    CL.clReleaseMemObject(p_data2);
//                }
//                else
//                {
//                    // the final pass, we have to "fake" backward one entry to get the actual full block
//                    // using the "partial" value from the last block
//                    int next_under = next_chunk_size + 1;
//                    int[] next_chunk = new int[next_under];
//                    System.arraycopy(input, nextOffset, next_chunk, 1, next_chunk_size);
//                    next_chunk[0] = nxint_n;
//                    long x2 = ((long)Sizeof.cl_int * (long)next_under);
//                    cl_mem p_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR, x2, Pointer.to(next_chunk), null);
//                    int nxint = scan_chunk(p_data, next_chunk, next_under);
//                    nxint_n = nxint;
////                    input[input.length-1] = nxint;
//                    // note; not using first value
//                    System.arraycopy(next_chunk, 1, input, nextOffset, next_chunk_size);
//                    CL.clReleaseMemObject(p_data);
//                    remaining -= next_chunk_size - 1;
//                    if (next_chunk_size == m)
//                    {
//                        input[input.length-1] = nxint;
//                    }
//                    //System.out.println("remaining: " + next_chunk_size);
//                    remaining = 0;
//                    //System.out.println("debg: " + remaining);
//                    //nextOffset += next_chunk_size;
//                    //remaining = 0;
//                    //nextOffset += next_chunk_size;
//                    // do final check
//                }
//
//            }
//            System.out.println("debug: " + nxint_n + " lcount: " + lcount);
//            //input[input.length-1] = nxint_n;
//            //recursive_scan(d_data, input, n);
//        }
        CL.clReleaseMemObject(d_data);
    }

    private static void scan_single_block(cl_mem d_data, int[] output, int n)
    {
        int localBufferSize = Sizeof.cl_int * m;
        long data_buf_size = (long)Sizeof.cl_int * n;
        Pointer dst_data = Pointer.to(output);
        Pointer src_data = Pointer.to(d_data);
        clSetKernelArg(k_scan_single_block, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_scan_single_block, 1, localBufferSize,null);
        clSetKernelArg(k_scan_single_block, 2, Sizeof.cl_int, Pointer.to(new int[]{n}));
        clEnqueueNDRangeKernel(commandQueue, k_scan_single_block, 1, null,
            new long[]{wx}, new long[]{wx}, 0, null, null);
        clEnqueueReadBuffer(commandQueue, d_data, CL_TRUE, 0,
            data_buf_size, dst_data, 0, null, null);
        System.out.println("---");
    }

//    private static int scan_chunk(cl_mem d_data, int[] output, int n)
//    {
//        int k = (int) Math.ceil((float)n/(float)m);
//        //size of each subarray stored in local memory
//        int localBufferSize = Sizeof.cl_int * m;
//
//            // note: as a hack, you can double this value to push all objects into the first chunk
//            //  but it caps out at 262_144 items before a CL error is thrown
//            int gx = k * wx; // * 2
//            int xf = k == 1 ? n : gx;
//            long data_buf_size = (long)Sizeof.cl_int * xf;
//            long part_buf_size = (long)Sizeof.cl_int * k * 2;
//            int[] partial_bytes = new int[(int)part_buf_size];
//            Pointer dst_part = Pointer.to(partial_bytes);
//            cl_mem d_partial = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR, part_buf_size, dst_part, null);
//            Pointer src_part = Pointer.to(d_partial);
//            //clw.dev_malloc(sizeof(int)*k);
//
//            Pointer dst_data = Pointer.to(output);
//            Pointer src_data = Pointer.to(d_data);
//            clSetKernelArg(k_scan_subarrays, 0, Sizeof.cl_mem, src_data);
//            clSetKernelArg(k_scan_subarrays, 1, localBufferSize,null);
//            clSetKernelArg(k_scan_subarrays, 2, Sizeof.cl_mem, src_part);
//            clSetKernelArg(k_scan_subarrays, 3, Sizeof.cl_int, Pointer.to(new int[]{n}));
//
//            clEnqueueNDRangeKernel(commandQueue, k_scan_subarrays, 1, null,
//                new long[]{gx}, new long[]{wx}, 0, null, null);
//            clEnqueueReadBuffer(commandQueue, d_data, CL_TRUE, 0,
//                data_buf_size, dst_data, 0, null, null);
//            clEnqueueReadBuffer(commandQueue, d_partial, CL_TRUE, 0,
//                part_buf_size, dst_part, 0, null, null);
////            clw.kernel_arg(scan_subarrays,
////                d_data, bufsize, d_partial, n);
////            k1 += clw.run_kernel_with_timing(scan_subarrays, /*dim=*/1, &gx, &wx);
//
//
////            clSetKernelArg(k_scan_inc_subarrays, 0, Sizeof.cl_mem, src_data);
////            clSetKernelArg(k_scan_inc_subarrays, 1, localBufferSize,null);
////            clSetKernelArg(k_scan_inc_subarrays, 2, Sizeof.cl_mem, src_part);
////            clSetKernelArg(k_scan_inc_subarrays, 3, Sizeof.cl_int, Pointer.to(new int[]{n}));
////
////            clEnqueueNDRangeKernel(commandQueue, k_scan_inc_subarrays, 1, null,
////                new long[]{gx}, new long[]{wx}, 0, null, null);
////            clEnqueueReadBuffer(commandQueue, d_data, CL_TRUE, 0,
////                data_buf_size, dst_data, 0, null, null);
////            clEnqueueReadBuffer(commandQueue, d_partial, CL_TRUE, 0,
////                part_buf_size, dst_part, 0, null, null);
//
////            clw.kernel_arg(scan_inc_subarrays,
////                d_data, bufsize, d_partial, n);
////            k2 += clw.run_kernel_with_timing(scan_inc_subarrays, /*dim=*/1, &gx, &wx);
////
////            clw.dev_free(d_partial);
//
//            CL.clReleaseMemObject(d_partial);
//        return partial_bytes[0];
//    }
}
