import org.jocl.*;

import java.util.Arrays;

import static org.jocl.CL.*;
public class CLMain
{
    public static void main(String[] args) {
        // Input array
        int[] input = {4, 1, 3, 2, 5, 7, 6};

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
            "__kernel void blelloch_scan(__global int* input, __global int* output, int n) {\n" +
                "    __local int temp[1024];\n" +
                "    int gid = get_global_id(0);\n" +
                "    int lid = get_local_id(0);\n" +
                "    temp[lid] = (gid < n) ? input[gid] : 0;\n" +
                "    barrier(CLK_LOCAL_MEM_FENCE);\n" +
                "    for (int offset = 1; offset < n; offset *= 2) {\n" +
                "        int t = temp[lid];\n" +
                "        barrier(CLK_LOCAL_MEM_FENCE);\n" +
                "        if (lid >= offset)\n" +
                "            t += temp[lid - offset];\n" +
                "        temp[lid] = t;\n" +
                "        barrier(CLK_LOCAL_MEM_FENCE);\n" +
                "    }\n" +
                "    if (gid < n)\n" +
                "        output[gid] = temp[lid];\n" +
                "}\n";
        cl_program program = CL.clCreateProgramWithSource(context, 1, new String[]{source}, null, null);
        CL.clBuildProgram(program, 0, null, null, null, null);

        // Create input and output buffers
        int n = input.length;
        int[] output = new int[n];
        cl_mem inputBuffer = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY | CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_int * n, Pointer.to(input), null);
        cl_mem outputBuffer = CL.clCreateBuffer(context, CL.CL_MEM_WRITE_ONLY, Sizeof.cl_int * n, null, null);

        // Create kernel
        cl_kernel kernel = CL.clCreateKernel(program, "blelloch_scan", null);
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(inputBuffer));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(outputBuffer));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[]{n}));

        // Execute the kernel
        long globalWorkSize[] = new long[]{n};
        CL.clEnqueueWriteBuffer(commandQueue, inputBuffer, CL.CL_TRUE, 0, Sizeof.cl_int * n, Pointer.to(input), 0, null, null);
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, globalWorkSize, null, 0, null, null);
        CL.clEnqueueReadBuffer(commandQueue, outputBuffer, CL.CL_TRUE, 0, Sizeof.cl_int * n, Pointer.to(output), 0, null, null);

        System.out.println("GPU Output: " + Arrays.toString(output));


        // Release resources
        CL.clReleaseMemObject(inputBuffer);
        CL.clReleaseMemObject(outputBuffer);
        CL.clReleaseKernel(kernel);
        CL.clReleaseProgram(program);
        CL.clReleaseCommandQueue(commandQueue);
        CL.clReleaseContext(context);

        float[] t = new float[n];
        for (int i = 1; i < input.length; i ++)
        {
            t[i] = t[i-1] + input[i-1];
        }
        System.out.println("CPU Output: " + Arrays.toString(t));
    }
}
