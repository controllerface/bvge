package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GpuKernel;

import static com.controllerface.bvge.cl.OpenCL.*;
import static com.controllerface.bvge.cl.OpenCLUtils.read_src;

public class BuildKeyMap extends GpuKernel
{
    @Override
    protected void init()
    {
        var source = read_src("kernels/build_key_map.cl");
        this.program = cl_p(prag_int32_base_atomics, func_calculate_key_index, source);
        this.kernels.put(kn_build_key_map, cl_k(program, kn_build_key_map));
    }
}
