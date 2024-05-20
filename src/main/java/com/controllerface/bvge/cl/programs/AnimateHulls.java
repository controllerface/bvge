package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.GPUProgram;
import com.controllerface.bvge.cl.Kernel;

public class AnimateHulls extends GPUProgram
{
    @Override
    public void init()
    {
        src.add(const_entity_flags);
        src.add(const_hull_flags);
        src.add(const_identity_matrix);
        src.add(func_matrix_transform);
        src.add(func_matrix_multiply);
        src.add(func_matrix_mul_affine);
        src.add(func_pos_vector_to_matrix);
        src.add(func_scl_vector_to_matrix);
        src.add(func_rot_quaternion_to_matrix);
        src.add(func_vector_lerp);
        src.add(func_quaternion_lerp);
        src.add(CLUtils.read_src("programs/animate_hulls.cl"));

        make_program();

        load_kernel(Kernel.animate_entities);
        load_kernel(Kernel.animate_bones);
        load_kernel(Kernel.animate_points);
    }
}
