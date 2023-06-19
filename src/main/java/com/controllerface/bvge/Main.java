package com.controllerface.bvge;

import com.controllerface.bvge.window.Window;
import org.jocl.*;
import org.lwjgl.system.FunctionProviderLocal;

import java.util.Arrays;

import static com.controllerface.bvge.util.InfoUtil.getDeviceInfoStringUTF8;
import static com.controllerface.bvge.util.InfoUtil.getPlatformInfoStringUTF8;
import static org.jocl.CL.*;
import static org.jocl.Sizeof.cl_float2;


public class Main
{
    public static void main(String[] args)
    {
        //test2();
        CLInstance.init();
        Window window = Window.get();
        window.run();
        CLInstance.destroy();
    }

    private static String programSource =
        "__kernel void "+
            "sampleKernel(__global const float *a,"+
            "             __global const float *b,"+
            "             __global float *c)"+
            "{"+
            "    int gid = get_global_id(0);"+
            "    c[gid] = a[gid] + b[gid];"+
            "}";


    private static String programSource2 =
        "__kernel void "+
            "sampleKernel(__global const float2 *a,"+
            "             __global const float2 *b,"+
            "             __global float2 *c)"+
            "{"+
            "    int gid = get_global_id(0);"+
            "    c[gid] = distance(a[gid], b[gid]);"+
            "}";


    private static void test2()
    {



        // The platform, device type and device number
        // that will be used
        final int platformIndex = 0;
        final long deviceType = CL_DEVICE_TYPE_ALL;
        final int deviceIndex = 0;

        // Enable exceptions and subsequently omit error checks in this sample
        org.jocl.CL.setExceptionsEnabled(true);

        // Obtain the number of platforms
        int numPlatformsArray[] = new int[1];
        clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];

        // Obtain a platform ID
        cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
        clGetPlatformIDs(platforms.length, platforms, null);
        cl_platform_id platform = platforms[platformIndex];

        // Initialize the context properties
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

        // Obtain the number of devices for the platform
        int numDevicesArray[] = new int[1];
        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
        int numDevices = numDevicesArray[0];

        // Obtain a device ID
        cl_device_id devices[] = new cl_device_id[numDevices];
        clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
        cl_device_id device = devices[deviceIndex];

        // Create a context for the selected device
        cl_context context = clCreateContext(
            contextProperties, 1, new cl_device_id[]{device},
            null, null, null);

        // Create a command-queue for the selected device
        cl_queue_properties properties = new cl_queue_properties();
        cl_command_queue commandQueue = clCreateCommandQueueWithProperties(
            context, device, properties, null);



        // Create the program from the source code
        cl_program program = clCreateProgramWithSource(context,
            1, new String[]{ programSource2 }, null, null);

        // Build the program
        clBuildProgram(program, 0, null, null, null, null);

        // Create the kernel
        cl_kernel kernel = clCreateKernel(program, "sampleKernel", null);










        // Create input- and output data
        int n = 6;
//        float srcArrayA[] = new float[n];
//        float srcArrayB[] = new float[n];
//        float dstArray[] = new float[n];

//        srcArrayA[0] = 1;
//        srcArrayA[1] = 2;
//        srcArrayA[2] = 0;
//        srcArrayA[3] = 0;
//        srcArrayA[4] = 10;
//        srcArrayA[5] = 10;
//
//        srcArrayB[0] = 4;
//        srcArrayB[1] = 3;
//        srcArrayB[2] = 5;
//        srcArrayB[3] = 5;
//        srcArrayB[4] = 0;
//        srcArrayB[5] = 0;

//        Pointer srcA = Pointer.to(srcArrayA);
//        Pointer srcB = Pointer.to(srcArrayB);
//        Pointer dst = Pointer.to(dstArray);
//
//        // Allocate the memory objects for the input- and output data
//        cl_mem srcMemA = clCreateBuffer(context,
//            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
//            Sizeof.cl_float * n, srcA, null);
//
//        cl_mem srcMemB = clCreateBuffer(context,
//            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
//            Sizeof.cl_float * n, srcB, null);
//
//        cl_mem dstMem = clCreateBuffer(context,
//            CL_MEM_READ_WRITE,
//            Sizeof.cl_float * n, null, null);



        // Set the work-item dimensions
        long global_work_size[] = new long[]{n};






        for (int i = 0; i < 5; i++)
        {

            float srcArrayA[] = new float[n];
            float srcArrayB[] = new float[n];
            float dstArray[] = new float[n];

            srcArrayA[0] = 1 + i;
            srcArrayA[1] = 2 + i;
            srcArrayA[2] = 0 + i;
            srcArrayA[3] = 0 + i;
            srcArrayA[4] = 10 + i;
            srcArrayA[5] = 10 + i;

            srcArrayB[0] = 4 - i;
            srcArrayB[1] = 3 - i;
            srcArrayB[2] = 5 - i;
            srcArrayB[3] = 5 - i;
            srcArrayB[4] = 0 - i;
            srcArrayB[5] = 0 - i;

            Pointer srcA = Pointer.to(srcArrayA);
            Pointer srcB = Pointer.to(srcArrayB);
            Pointer dst = Pointer.to(dstArray);

            // Allocate the memory objects for the input- and output data
            cl_mem srcMemA = clCreateBuffer(context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * n, srcA, null);

            cl_mem srcMemB = clCreateBuffer(context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * n, srcB, null);

            cl_mem dstMem = clCreateBuffer(context,
                CL_MEM_READ_WRITE,
                Sizeof.cl_float * n, null, null);


            // Set the arguments for the kernel
            int a = 0;
            clSetKernelArg(kernel, a++, Sizeof.cl_mem, Pointer.to(srcMemA));
            clSetKernelArg(kernel, a++, Sizeof.cl_mem, Pointer.to(srcMemB));
            clSetKernelArg(kernel, a++, Sizeof.cl_mem, Pointer.to(dstMem));



            // Execute the kernel
            clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
                global_work_size, null, 0, null, null);

            // Read the output data
            clEnqueueReadBuffer(commandQueue, dstMem, CL_TRUE, 0,
                n * Sizeof.cl_float, dst, 0, null, null);






            clReleaseMemObject(srcMemA);
            clReleaseMemObject(srcMemB);
            clReleaseMemObject(dstMem);

            System.out.println("A: " + Arrays.toString(srcArrayA));
            System.out.println("B: " + Arrays.toString(srcArrayB));
            System.out.println("Result: " + Arrays.toString(dstArray));
        }







        // Release kernel, program, and memory objects
//        clReleaseMemObject(srcMemA);
//        clReleaseMemObject(srcMemB);
//        clReleaseMemObject(dstMem);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
    }
















    public static void get(FunctionProviderLocal provider, long platform, String name) {
        System.out.println(name + ": " + provider.getFunctionAddress(platform, name));
    }

    private static void printPlatformInfo(long platform, String param_name, int param) {
        System.out.println("\t" + param_name + " = " + getPlatformInfoStringUTF8(platform, param));
    }

    private static void printDeviceInfo(long device, String param_name, int param) {
        System.out.println("\t" + param_name + " = " + getDeviceInfoStringUTF8(device, param));
    }

    private static String getEventStatusName(int status) {
        switch (status) {
            case CL_QUEUED:
                return "CL_QUEUED";
            case CL_SUBMITTED:
                return "CL_SUBMITTED";
            case CL_RUNNING:
                return "CL_RUNNING";
            case CL_COMPLETE:
                return "CL_COMPLETE";
            default:
                throw new IllegalArgumentException(String.format("Unsupported event status: 0x%X", status));
        }
    }
}

