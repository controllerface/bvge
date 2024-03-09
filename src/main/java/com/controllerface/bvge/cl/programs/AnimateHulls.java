package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.CLUtils.read_src;

public class AnimateHulls extends GPUProgram
{
    @Override
    public void init()
    {
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
        src.add(read_src("programs/animate_hulls.cl"));

        make_program();

        load_kernel(GPU.Kernel.animate_armatures);
        load_kernel(GPU.Kernel.animate_bones);
        load_kernel(GPU.Kernel.animate_points);
    }
}
