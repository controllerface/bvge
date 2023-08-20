package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.GPU.*;
import static com.controllerface.bvge.cl.CLUtils.read_src;

public class GenerateKeys extends GPUProgram
{
    @Override
    protected void init()
    {
        src.add(prag_int32_base_atomics);
        src.add(func_calculate_key_index);
        src.add(read_src("kernels/generate_keys.cl"));

        make_program();

        load_kernel(Kernel.generate_keys);
    }
}
