package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CompactArmatures_k extends GPUKernel<CompactArmatures_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        buffer_in(Sizeof.cl_mem),
        buffer_in_2(Sizeof.cl_mem),
        armatures(Sizeof.cl_mem),
        armature_accel(Sizeof.cl_mem),
        armature_flags(Sizeof.cl_mem),
        armature_animation_indices(Sizeof.cl_mem),
        armature_animation_elapsed(Sizeof.cl_mem),
        hull_tables(Sizeof.cl_mem),
        hulls(Sizeof.cl_mem),
        hull_flags(Sizeof.cl_mem),
        element_tables(Sizeof.cl_mem),
        points(Sizeof.cl_mem),
        vertex_tables(Sizeof.cl_mem),
        bone_tables(Sizeof.cl_mem),
        bone_bind_tables(Sizeof.cl_mem),
        bone_index_tables(Sizeof.cl_mem),
        edges(Sizeof.cl_mem),
        bone_shift(Sizeof.cl_mem),
        point_shift(Sizeof.cl_mem),
        edge_shift(Sizeof.cl_mem),
        hull_shift(Sizeof.cl_mem),
        bone_bind_shift(Sizeof.cl_mem);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CompactArmatures_k(cl_command_queue command_queue)
    {
        super(command_queue, GPU.Program.scan_deletes.gpu.kernels().get(GPU.Kernel.compact_armatures), Args.values());
    }
}
