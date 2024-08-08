package com.controllerface.bvge.gpu.cl.programs;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.CLUtils;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;

public class AnimateHulls extends GPUProgram
{
    @Override
    public GPUProgram init()
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
        src.add(GPU.CL.read_src("programs/animate_hulls.cl"));

        make_program();

        load_kernel(KernelType.animate_entities);
        load_kernel(KernelType.animate_bones);
        load_kernel(KernelType.animate_points);

        return this;
    }
}
