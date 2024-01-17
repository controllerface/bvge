package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class AnimateHulls_k extends GPUKernel
{
    public AnimateHulls_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.animate_hulls), 8);
        int arg_index = 0;
        def_arg(arg_index++, Sizeof.cl_mem);  // __global float4 *points
        def_arg(arg_index++, Sizeof.cl_mem);  // __global float4 *hulls
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int4 *hull_flags
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int2 *vertex_tables
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int4 *bone_tables
        def_arg(arg_index++, Sizeof.cl_mem);  // __global float4 *armatures
        def_arg(arg_index++, Sizeof.cl_mem);  // __global float2 *vertex_references
        def_arg(arg_index++, Sizeof.cl_mem);  // __global float16 *bones
        System.out.printf("set %d args for %s\n", arg_index, this.getClass().getSimpleName());
    }

    public void set_points(Pointer points)
    {
        new_arg(0, Sizeof.cl_mem, points);
    }

    public void set_hulls(Pointer hulls)
    {
        new_arg(1, Sizeof.cl_mem, hulls);
    }

    public void set_hull_flags(Pointer hull_flags)
    {
        new_arg(2, Sizeof.cl_mem, hull_flags);
    }

    public void set_vertex_table(Pointer vertex_tables)
    {
        new_arg(3, Sizeof.cl_mem, vertex_tables);
    }

    public void set_bone_tables(Pointer bone_tables)
    {
        new_arg(4, Sizeof.cl_mem, bone_tables);
    }

    public void set_armatures(Pointer armatures)
    {
        new_arg(5, Sizeof.cl_mem, armatures);
    }

    public void set_vertex_refs(Pointer vertex_refs)
    {
        new_arg(6, Sizeof.cl_mem, vertex_refs);
    }

    public void set_bone_instances(Pointer bone_instances)
    {
        new_arg(7, Sizeof.cl_mem, bone_instances);
    }
}
