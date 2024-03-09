package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.CLUtils.read_src;
import static com.controllerface.bvge.cl.GPU.Kernel;

public class PrepareBones extends GPUProgram
{
    @Override
    public void init()
    {
        src.add(func_matrix_transform);
        src.add(read_src("programs/prepare_bones.cl"));

        make_program();

        load_kernel(Kernel.prepare_bones);
    }
}
