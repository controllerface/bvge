package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUProgram;

public class GenerateKeys extends GPUProgram
{
    @Override
    public void init()
    {
        src.add(prag_int32_base_atomics);
        src.add(func_calculate_key_index);
        src.add(CLUtils.read_src("programs/generate_keys.cl"));

        make_program();

        load_kernel(GPU.Kernel.generate_keys);
    }
}
