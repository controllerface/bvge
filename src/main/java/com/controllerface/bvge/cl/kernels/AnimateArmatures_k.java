package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class AnimateArmatures_k extends GPUKernel<AnimateArmatures_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        armature_bones(Sizeof.cl_mem),
        bone_bind_poses(Sizeof.cl_mem),
        model_transforms(Sizeof.cl_mem),
        bone_bind_tables(Sizeof.cl_mem),
        bone_channel_tables(Sizeof.cl_mem),
        bone_pos_channel_tables(Sizeof.cl_mem),
        bone_rot_channel_tables(Sizeof.cl_mem),
        bone_scl_channel_tables(Sizeof.cl_mem),
        armature_flags(Sizeof.cl_mem),
        hull_tables(Sizeof.cl_mem),
        key_frames(Sizeof.cl_mem),
        frame_times(Sizeof.cl_mem),
        animation_timing_indices(Sizeof.cl_mem),
        animation_timings(Sizeof.cl_mem),
        armature_animation_indices(Sizeof.cl_mem),
        armature_animation_elapsed(Sizeof.cl_mem),
        delta_time(Sizeof.cl_float);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public AnimateArmatures_k(cl_command_queue command_queue)
    {
        super(command_queue, GPU.Program.animate_hulls.gpu.kernels().get(GPU.Kernel.animate_armatures), Args.values());
    }
}
