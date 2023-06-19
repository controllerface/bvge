package com.controllerface.bvge;

import org.jocl.*;

import static org.jocl.CL.*;

public class TestBed
{
    private static String programSource =
        "__kernel void "+
            "sampleKernel(__global float *a)"+
            "{"+
            "    int gidX = get_global_id(0);"+
            "    int gidY = get_global_id(1);"+
            "    a[gidX+3*gidY] *= 2;"+
            "}";

    private static cl_context context;
    private static cl_command_queue commandQueue;
    private static cl_kernel kernel;
    private static cl_program program;

    public static void main(String args[])
    {
        defaultInitialization();

        // Create input array
        float array[][] = {{1,2},{4,5},{7,8}};

        // Allocate the memory object
        cl_mem mem = clCreateBuffer(context, CL_MEM_READ_WRITE,
            Sizeof.cl_float * 2 * 3, null, null);

        // Write the source array into the buffer
        writeBuffer2D(commandQueue, mem, array);

        // Execute the kernel
        clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(mem));
        clEnqueueNDRangeKernel(commandQueue, kernel, 2, null,
            new long[]{2,3}, null, 0, null, null);

        // Read the buffer back to the array
        readBuffer2D(commandQueue, mem, array);

        // Release kernel, program, and memory objects
        clReleaseMemObject(mem);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);

        System.out.println("Result: ");
        for (int r=0; r<array.length; r++)
        {
            System.out.println(java.util.Arrays.toString(array[r]));
        }
    }

    private static void writeBuffer2D(cl_command_queue commandQueue, cl_mem buffer, float array[][])
    {
        long byteOffset = 0;
        for (int r=0; r<array.length; r++)
        {
            int bytes = array[r].length * Sizeof.cl_float;
            clEnqueueWriteBuffer(
                commandQueue, buffer, CL_TRUE, byteOffset, bytes,
                Pointer.to(array[r]), 0, null, null);
            byteOffset += bytes;
        }
    }

    private static void readBuffer2D(cl_command_queue commandQueue, cl_mem buffer, float array[][])
    {
        long byteOffset = 0;
        for (int r=0; r<array.length; r++)
        {
            int bytes = array[r].length * Sizeof.cl_float;
            clEnqueueReadBuffer(
                commandQueue, buffer, CL_TRUE, byteOffset, bytes,
                Pointer.to(array[r]), 0, null, null);
            byteOffset += bytes;
        }
    }


    private static void defaultInitialization()
    {
        // Obtain the platform IDs and initialize the context properties
        cl_platform_id platforms[] = new cl_platform_id[1];
        clGetPlatformIDs(platforms.length, platforms, null);
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platforms[0]);

        // Create an OpenCL context on a GPU device
        context = clCreateContextFromType(
            contextProperties, CL_DEVICE_TYPE_GPU, null, null, null);
        if (context == null)
        {
            // If no context for a GPU device could be created,
            // try to create one for a CPU device.
            context = clCreateContextFromType(
                contextProperties, CL_DEVICE_TYPE_CPU, null, null, null);

            if (context == null)
            {
                System.out.println("Unable to create a context");
                return;
            }
        }

        // Enable exceptions and subsequently omit error checks in this sample
        CL.setExceptionsEnabled(true);

        // Get the list of GPU devices associated with the context
        long numBytes[] = new long[1];
        clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, null, numBytes);

        // Obtain the cl_device_id for the first device
        int numDevices = (int) numBytes[0] / Sizeof.cl_device_id;
        cl_device_id devices[] = new cl_device_id[numDevices];
        clGetContextInfo(context, CL_CONTEXT_DEVICES, numBytes[0],
            Pointer.to(devices), null);

        // Create a command-queue
        commandQueue =
            clCreateCommandQueue(context, devices[0], 0, null);

        // Create the program from the source code
        program = clCreateProgramWithSource(context,
            1, new String[]{ programSource }, null, null);

        // Build the program
        clBuildProgram(program, 0, null, null, null, null);

        // Create the kernel
        kernel = clCreateKernel(program, "sampleKernel", null);
    }
}
