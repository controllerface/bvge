package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.GPUProgram;
import com.controllerface.bvge.cl.Kernel;

public class ControlEntities extends GPUProgram
{
    @Override
    public void init()
    {
        src.add(CLUtils.read_src("programs/control_entities.cl"));

        make_program();

        load_kernel(Kernel.set_control_points);
        load_kernel(Kernel.handle_movement);
    }
}