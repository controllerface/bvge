import org.jocl.*;

import java.util.Arrays;

public class CLMain2
{
    public static void main(String[] args) {
        // Input array
        int size = 101;
        int[] input = new int[size];
        for (int i=0; i< size; i++)
        {
            input[i] = i +1;
        }

        // Initialize the JOCL library
        CL.setExceptionsEnabled(true);
        cl_platform_id[] platforms = new cl_platform_id[1];
        CL.clGetPlatformIDs(1, platforms, null);
        cl_platform_id platform = platforms[0];
        cl_device_id[] devices = new cl_device_id[1];
        CL.clGetDeviceIDs(platform, CL.CL_DEVICE_TYPE_GPU, 1, devices, null);
        cl_device_id device = devices[0];
        cl_context context = CL.clCreateContext(null, 1, new cl_device_id[]{device}, null, null, null);
        cl_command_queue commandQueue = CL.clCreateCommandQueue(context, device, 0, null);

        // Create and build the kernel program
        String source =
            "#define BLOCK_SIZE 256\n" +
                "\n" +
                "__kernel void scan(__global int* input, __global int* output, int n) {\n" +
                "    __local int temp[2 * BLOCK_SIZE];\n" +
                "    int gid = get_global_id(0);\n" +
                "    int lid = get_local_id(0);\n" +
                "    int lsize = get_local_size(0);\n" +
                "    int lblock = lid / 2;\n" +
                "    int offset = 1;\n" +
                "    temp[2 * lid] = (gid < n) ? input[gid] : 0;\n" +
                "    temp[2 * lid + 1] = 0;\n" +
                "    for (int d = lsize >> 1; d > 0; d >>= 1) {\n" +
                "        barrier(CLK_LOCAL_MEM_FENCE);\n" +
                "        if (lid < d) {\n" +
                "            int ai = lblock * 2 * BLOCK_SIZE + lid * offset - 1;\n" +
                "            int bi = ai + offset;\n" +
                "            temp[bi] += temp[ai];\n" +
                "        }\n" +
                "        offset <<= 1;\n" +
                "    }\n" +
                "    if (lid == 0) {\n" +
                "        temp[lsize - 1] = 0;\n" +
                "    }\n" +
                "    for (int d = 1; d < lsize; d <<= 1) {\n" +
                "        offset >>= 1;\n" +
                "        barrier(CLK_LOCAL_MEM_FENCE);\n" +
                "        if (lid < d) {\n" +
                "            int ai = lblock * 2 * BLOCK_SIZE + lid * offset - 1;\n" +
                "            int bi = ai + offset;\n" +
                "            int t = temp[ai];\n" +
                "            temp[ai] = temp[bi];\n" +
                "            temp[bi] += t;\n" +
                "        }\n" +
                "    }\n" +
                "    barrier(CLK_LOCAL_MEM_FENCE);\n" +
                "    if (gid < n) {\n" +
                "        output[gid] = temp[2 * lid];\n" +
                "    }\n" +
                "}\n";

        cl_program program = CL.clCreateProgramWithSource(context, 1, new String[]{source}, null, null);
        CL.clBuildProgram(program, 0, null, null, null, null);

        // Create input and output buffers
        int n = input.length;
        int[] output = new int[n];
        cl_mem inputBuffer = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY | CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_int * n, Pointer.to(input), null);
        cl_mem outputBuffer = CL.clCreateBuffer(context, CL.CL_MEM_WRITE_ONLY, Sizeof.cl_int * n, null, null);

        // Create kernel
        cl_kernel scanKernel = CL.clCreateKernel(program, "scan", null);
        CL.clSetKernelArg(scanKernel, 0, Sizeof.cl_mem, Pointer.to(inputBuffer));
        CL.clSetKernelArg(scanKernel, 1, Sizeof.cl_mem, Pointer.to(outputBuffer));
        CL.clSetKernelArg(scanKernel, 2, Sizeof.cl_int, Pointer.to(new int[]{n}));

        // Execute the kernel
        long globalWorkSize[] = new long[]{n};
        long localWorkSize[] = new long[]{Math.min(n, 2 * 256)}; // Adjust this based on device capabilities
        CL.clEnqueueNDRangeKernel(commandQueue, scanKernel, 1, null, globalWorkSize, localWorkSize, 0, null, null);

        // Read the output from the device
        CL.clEnqueueReadBuffer(commandQueue, outputBuffer, CL.CL_TRUE, 0, Sizeof.cl_int * n, Pointer.to(output), 0, null, null);

        // Perform the final scan operation
        int carry = 0;
        for (int i = 0; i < n; i++) {
            int t = output[i];
            output[i] = carry;
            carry += t;
        }

        // Print the output
        System.out.println("GPU Output: " + Arrays.toString(output));


        // Release resources
        CL.clReleaseMemObject(inputBuffer);
        CL.clReleaseMemObject(outputBuffer);
        CL.clReleaseKernel(scanKernel);
        CL.clReleaseProgram(program);
        CL.clReleaseCommandQueue(commandQueue);
        CL.clReleaseContext(context);

        int[] t = new int[n];
        for (int i = 1; i < input.length; i ++)
        {
            t[i] = t[i-1] + input[i-1];
        }
        System.out.println("CPU Output: " + Arrays.toString(t));
    }
}
