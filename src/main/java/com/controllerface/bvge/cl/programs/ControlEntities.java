package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.Kernel;

public class ControlEntities extends GPUProgram
{
    @Override
    public void init()
    {
        src.add(const_control_flags);
        src.add(const_hull_flags);
        src.add(const_entity_flags);
        src.add(const_animation_states);
        src.add(const_animation_lookup_table);
        src.add(CLUtils.read_src("programs/control_entities.cl"));

        make_program();

        load_kernel(Kernel.set_control_points);
        load_kernel(Kernel.handle_movement);
    }
}
