package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class Integrate_k extends GPUKernel
{
    public Integrate_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.integrate), 13);
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
    }

    public void set_hulls(Pointer hulls)
    {
        new_arg(0, Sizeof.cl_mem, hulls);
    }

    public void set_armatures(Pointer armatures)
    {
        new_arg(1, Sizeof.cl_mem, armatures);
    }

    public void set_armature_flags(Pointer armature_flags)
    {
        new_arg(2, Sizeof.cl_mem, armature_flags);
    }

    public void set_hull_element_table(Pointer hull_element_table)
    {
        new_arg(3, Sizeof.cl_mem, hull_element_table);
    }

    public void set_armature_accel(Pointer armature_accel)
    {
        new_arg(4, Sizeof.cl_mem, armature_accel);
    }

    public void set_hull_rotation(Pointer hull_rotation)
    {
        new_arg(5, Sizeof.cl_mem, hull_rotation);
    }

    public void set_points(Pointer points)
    {
        new_arg(6, Sizeof.cl_mem, points);
    }

    public void set_aabb(Pointer aabb)
    {
        new_arg(7, Sizeof.cl_mem, aabb);
    }

    public void set_aabb_index(Pointer aabb_index)
    {
        new_arg(8, Sizeof.cl_mem, aabb_index);
    }

    public void set_aabb_key_table(Pointer aabb_key_table)
    {
        new_arg(9, Sizeof.cl_mem, aabb_key_table);
    }

    public void set_hull_flags(Pointer hull_flags)
    {
        new_arg(10, Sizeof.cl_mem, hull_flags);
    }

    public void set_point_anti_gravity(Pointer point_anti_gravity)
    {
        new_arg(11, Sizeof.cl_mem, point_anti_gravity);
    }
}
