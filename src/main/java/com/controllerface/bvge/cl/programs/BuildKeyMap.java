package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.Kernel;

public class BuildKeyMap extends GPUProgram
{
    @Override
    public void init()
    {
        src.add(prag_int32_base_atomics);
        src.add(func_calculate_key_index);
        src.add(CLUtils.read_src("programs/build_key_map.cl"));

        make_program();

        load_kernel(Kernel.build_key_map);
    }
}
