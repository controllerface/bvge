package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GpuKernel;

import static com.controllerface.bvge.cl.OpenCL.*;
import static com.controllerface.bvge.cl.OpenCLUtils.read_src;

public class GenerateKeys extends GpuKernel
{
    @Override
    protected void init()
    {
        var source = read_src("kernels/generate_keys.cl");
        this.program = cl_p(prag_int32_base_atomics, func_calculate_key_index, source);
        this.kernels.put(kn_generate_keys, cl_k(program, kn_generate_keys));
    }
}
