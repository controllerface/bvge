package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CompactArmatures_k extends GPUKernel
{
    public CompactArmatures_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.compact_armatures), 17);
        int arg_index = 0;
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int *buffer_in
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int4 *buffer_in_2
        def_arg(arg_index++, Sizeof.cl_mem);  // __global float4 *armatures
        def_arg(arg_index++, Sizeof.cl_mem);  // __global float2 *armature_accel
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int4 *armature_flags
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int2 *hull_tables
        def_arg(arg_index++, Sizeof.cl_mem);  // __global float4 *hulls
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int4 *hull_flags
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int4 *element_tables
        def_arg(arg_index++, Sizeof.cl_mem);  // __global float4 *points
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int2 *vertex_tables
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int4 *bone_tables
        def_arg(arg_index++, Sizeof.cl_mem);  // __global float4 *edges
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int *bone_shift
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int *point_shift
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int *edge_shift
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int *hull_shift
        System.out.printf("set %d args for %s\n", arg_index, this.getClass().getSimpleName());
    }

    public void set_armatures(Pointer armatures)
    {
        new_arg(2, Sizeof.cl_mem, armatures);
    }

    public void set_armature_accel(Pointer armature_accel)
    {
        new_arg(3, Sizeof.cl_mem, armature_accel);
    }

    public void set_armature_flags(Pointer armature_flags)
    {
        new_arg(4, Sizeof.cl_mem, armature_flags);
    }

    public void set_hull_tables(Pointer hull_tables)
    {
        new_arg(5, Sizeof.cl_mem, hull_tables);
    }

    public void set_hulls(Pointer hulls)
    {
        new_arg(6, Sizeof.cl_mem, hulls);
    }

    public void set_hull_flags(Pointer hull_flags)
    {
        new_arg(7, Sizeof.cl_mem, hull_flags);
    }

    public void set_element_tables(Pointer element_tables)
    {
        new_arg(8, Sizeof.cl_mem, element_tables);
    }

    public void set_points(Pointer points)
    {
        new_arg(9, Sizeof.cl_mem, points);
    }

    public void set_vertex_tables(Pointer vertex_tables)
    {
        new_arg(10, Sizeof.cl_mem, vertex_tables);
    }

    public void set_bone_tables(Pointer bone_tables)
    {
        new_arg(11, Sizeof.cl_mem, bone_tables);
    }

    public void set_edges(Pointer edges)
    {
        new_arg(12, Sizeof.cl_mem, edges);
    }

    public void set_bone_shift(Pointer bone_shift)
    {
        new_arg(13, Sizeof.cl_mem, bone_shift);
    }

    public void set_point_shift(Pointer point_shift)
    {
        new_arg(14, Sizeof.cl_mem, point_shift);
    }

    public void set_edge_shift(Pointer edge_shift)
    {
        new_arg(15, Sizeof.cl_mem, edge_shift);
    }

    public void set_hull_shift(Pointer hull_shift)
    {
        new_arg(16, Sizeof.cl_mem, hull_shift);
    }
}
