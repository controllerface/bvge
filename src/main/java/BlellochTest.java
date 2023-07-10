import org.jocl.*;

import java.util.Arrays;

import static com.controllerface.bvge.cl.OCLFunctions.readSrc;

public class BlellochTest
{
    static cl_kernel scanKernel;
    static cl_program program;
    static cl_command_queue commandQueue;
    private static final String source = readSrc("blelloch_test.cl");

    public static void main(String[] args) {
        // Input array
        int size = 10;
        float[] input = new float[size];
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
        cl_context context = CL.clCreateContext(null, 1, new cl_device_id[]{device}, null, null, null);
        commandQueue = CL.clCreateCommandQueue(context, device, 0, null);

        program = CL.clCreateProgramWithSource(context, 1, new String[]{source}, null, null);
        CL.clBuildProgram(program, 0, null, null, null, null);

        // Create input and output buffers
        int n = input.length;
        float[] output = new float[n];
        cl_mem inputBuffer = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY | CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * n, Pointer.to(input), null);
        cl_mem outputBuffer = CL.clCreateBuffer(context, CL.CL_MEM_WRITE_ONLY, Sizeof.cl_float * n, null, null);

        System.out.println("GPU Output (1): " + Arrays.toString(output));


        // Create kernel
        scanKernel = CL.clCreateKernel(program, "scan", null);
        CL.clSetKernelArg(scanKernel, 0, Sizeof.cl_mem, Pointer.to(inputBuffer));
        CL.clSetKernelArg(scanKernel, 1, Sizeof.cl_mem, Pointer.to(outputBuffer));
        CL.clSetKernelArg(scanKernel, 2, Sizeof.cl_float * 256, null);
        CL.clSetKernelArg(scanKernel, 3, Sizeof.cl_int, Pointer.to(new int[]{n}));

        // Execute the kernel
        long globalWorkSize[] = new long[]{n};
        long localWorkSize[] = new long[]{Math.min(n, 256)}; // Adjust this based on device capabilities


        CL.clEnqueueNDRangeKernel(commandQueue, scanKernel, 1, null, globalWorkSize, localWorkSize, 0, null, null);

        // Read the output from the device
        CL.clEnqueueReadBuffer(commandQueue, outputBuffer, CL.CL_TRUE, 0, Sizeof.cl_int * n, Pointer.to(output), 0, null, null);

        // Perform the final scan operation
//        int carry = 0;
//        for (int i = 0; i < n; i++) {
//            output[i] = carry;
//            carry += input[i];
//        }

        // Print the output
        System.out.println("GPU Output: " + Arrays.toString(output));


        // Release resources
        CL.clReleaseMemObject(inputBuffer);
        CL.clReleaseMemObject(outputBuffer);
        CL.clReleaseKernel(scanKernel);
        CL.clReleaseProgram(program);
        CL.clReleaseCommandQueue(commandQueue);
        CL.clReleaseContext(context);

        float[] t = new float[n];
        for (int i = 1; i < input.length; i ++)
        {
            t[i] = t[i-1] + input[i-1];
        }
        System.out.println("CPU Output: " + Arrays.toString(t));
        System.out.println("DEBUG: c=" + Arrays.compare(t, output));
    }
}
