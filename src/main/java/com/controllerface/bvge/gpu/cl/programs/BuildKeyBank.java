package com.controllerface.bvge.gpu.cl.programs;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.CLUtils;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;

public class BuildKeyBank extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(prag_int32_base_atomics);
        src.add(func_calculate_key_index);
        src.add(GPU.CL.read_src("programs/build_key_bank.cl"));

        make_program();

        load_kernel(KernelType.build_key_bank);

        return this;
    }
}
