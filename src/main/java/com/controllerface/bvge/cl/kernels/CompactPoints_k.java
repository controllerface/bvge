package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CompactPoints_k extends GPUKernel
{
    public CompactPoints_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.compact_points), 5);
        int arg_index = 0;
        def_arg(0, Sizeof.cl_mem);  // __global int *point_shift
        def_arg(1, Sizeof.cl_mem);  // __global float4 *points
        def_arg(2, Sizeof.cl_mem);  // __global float *anti_gravity
        def_arg(3, Sizeof.cl_mem);  // __global int2 *vertex_tables
        def_arg(3, Sizeof.cl_mem);  // __global int4 *bone_tables
        System.out.printf("set %d args for %s\n", arg_index, this.getClass().getSimpleName());
    }

    public void set_point_shift(Pointer point_shift)
    {
        new_arg(0, Sizeof.cl_mem, point_shift);
    }

    public void set_points(Pointer points)
    {
        new_arg(1, Sizeof.cl_mem, points);
    }

    public void set_anti_gravity(Pointer anti_gravity)
    {
        new_arg(2, Sizeof.cl_mem, anti_gravity);
    }

    public void set_vertex_tables(Pointer vertex_tables)
    {
        new_arg(3, Sizeof.cl_mem, vertex_tables);
    }

    public void set_bone_tables(Pointer bone_tables)
    {
        new_arg(4, Sizeof.cl_mem, bone_tables);
    }
}
