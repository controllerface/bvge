package com.controllerface.bvge.cl;

import org.jocl.cl_context;
import org.jocl.cl_device_id;
import org.jocl.cl_kernel;
import org.jocl.cl_program;

import java.io.IOException;
import java.nio.charset.StandardCharsets;

import static org.jocl.CL.*;

public class OpenCLUtils
{
    public static String read_src(String file)
    {
        var stream = OpenCL.class.getResourceAsStream("/cl/" + file);
        try
        {
            byte [] bytes = stream.readAllBytes();
            return new String(bytes, StandardCharsets.UTF_8);
        }
        catch (IOException e)
        {
            throw new RuntimeException(e);
        }
    }

    public static cl_program cl_p(cl_context context, cl_device_id[] device_ids, String ... src)
    {
        var program = clCreateProgramWithSource(context, src.length, src, null, null);
        clBuildProgram(program, 1, device_ids, null, null, null);
        return program;
    }

    public static cl_kernel cl_k(cl_program program, String kernel_name)
    {
        return clCreateKernel(program, kernel_name, null);
    }
}
