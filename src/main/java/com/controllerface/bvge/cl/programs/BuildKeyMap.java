package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.GPU.*;
import static com.controllerface.bvge.cl.CLUtils.read_src;

public class BuildKeyMap extends GPUProgram
{
    @Override
    protected void init()
    {
        add_src(prag_int32_base_atomics);
        add_src(func_calculate_key_index);
        add_src(read_src("kernels/build_key_map.cl"));

        make_program();

        make_kernel(Kernel.build_key_map);
    }
}
