package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.animation.AnimationSettings;
import com.controllerface.bvge.animation.AnimationState;
import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.Kernel;

public class ControlEntities extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(const_control_flags);
        src.add(const_hull_flags);
        src.add(const_entity_flags);
        src.add(AnimationState.cl_constants());
        src.add(AnimationSettings.cl_lookup_table());
        src.add(CLUtils.read_src("programs/control_entities.cl"));

        make_program();

        load_kernel(Kernel.set_control_points);
        load_kernel(Kernel.handle_movement);

        return this;
    }
}
