package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class CreateArmature_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.gpu_crud;
    private static final GPU.Kernel kernel = GPU.Kernel.create_armature;

    public enum Args
    {
        armatures,
        armature_flags,
        hull_tables,
        armature_masses,
        armature_animation_indices,
        armature_animation_elapsed,
        target,
        new_armature,
        new_armature_flags,
        new_hull_table,
        new_armature_mass,
        new_armature_animation_index,
        new_armature_animation_time;
    }

    public CreateArmature_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
