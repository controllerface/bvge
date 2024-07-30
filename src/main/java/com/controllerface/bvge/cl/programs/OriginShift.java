package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.Kernel;

public class OriginShift extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(CLUtils.read_src("programs/origin_shift.cl"));

        make_program();

        load_kernel(Kernel.shift_points);
        load_kernel(Kernel.shift_hulls);
        load_kernel(Kernel.shift_entities);

        return this;
    }
}
