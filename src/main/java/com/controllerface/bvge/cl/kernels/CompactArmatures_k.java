package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class CompactArmatures_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.scan_deletes;
    private static final GPU.Kernel kernel = GPU.Kernel.compact_armatures;

    public enum Args
    {
        buffer_in,
        buffer_in_2,
        armatures,
        armature_accel,
        armature_flags,
        armature_animation_indices,
        armature_animation_elapsed,
        hull_tables,
        hulls,
        hull_flags,
        element_tables,
        points,
        vertex_tables,
        bone_tables,
        bone_bind_tables,
        bone_index_tables,
        edges,
        bone_shift,
        point_shift,
        edge_shift,
        hull_shift,
        bone_bind_shift;
    }

    public CompactArmatures_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.gpu.kernel_ptr(kernel));
    }
}
