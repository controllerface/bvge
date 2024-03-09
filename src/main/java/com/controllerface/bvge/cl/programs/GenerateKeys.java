package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.CLUtils.read_src;
import static com.controllerface.bvge.cl.GPU.Kernel;

public class GenerateKeys extends GPUProgram
{
    @Override
    public void init()
    {
        src.add(prag_int32_base_atomics);
        src.add(func_calculate_key_index);
        src.add(read_src("programs/generate_keys.cl"));

        make_program();

        load_kernel(Kernel.generate_keys);
    }
}
