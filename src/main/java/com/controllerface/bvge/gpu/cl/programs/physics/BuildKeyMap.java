package com.controllerface.bvge.gpu.cl.programs.physics;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class BuildKeyMap extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(prag_int32_base_atomics);
        src.add(func_calculate_key_index);
        src.add(GPU.CL.read_src("programs/build_key_map.cl"));

        make_program();

        load_kernel(KernelType.build_key_map);

        return this;
    }
}
