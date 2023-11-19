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
        super(command_queue, program.kernels().get(GPU.Kernel.compact_armatures), 16);
        def_arg(0, Sizeof.cl_mem);
        def_arg(1, Sizeof.cl_mem);
        def_arg(2, Sizeof.cl_mem);
        def_arg(3, Sizeof.cl_mem);
        def_arg(4, Sizeof.cl_mem);
        def_arg(5, Sizeof.cl_mem);
        def_arg(6, Sizeof.cl_mem);
        def_arg(7, Sizeof.cl_mem);
        def_arg(8, Sizeof.cl_mem);
        def_arg(9, Sizeof.cl_mem);
        def_arg(10, Sizeof.cl_mem);
        def_arg(11, Sizeof.cl_mem);
        def_arg(12, Sizeof.cl_mem);
        def_arg(13, Sizeof.cl_mem);
        def_arg(14, Sizeof.cl_mem);
        def_arg(15, Sizeof.cl_mem);
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

    public void set_edges(Pointer edges)
    {
        new_arg(11, Sizeof.cl_mem, edges);
    }

    public void set_bone_shift(Pointer bone_shift)
    {
        new_arg(12, Sizeof.cl_mem, bone_shift);
    }

    public void set_point_shift(Pointer point_shift)
    {
        new_arg(13, Sizeof.cl_mem, point_shift);
    }

    public void set_edge_shift(Pointer edge_shift)
    {
        new_arg(14, Sizeof.cl_mem, edge_shift);
    }

    public void set_hull_shift(Pointer hull_shift)
    {
        new_arg(15, Sizeof.cl_mem, hull_shift);
    }

}
