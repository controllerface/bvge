package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CompactHulls_k extends GPUKernel
{
    public CompactHulls_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.compact_hulls), 9);
        int arg_index = 0;
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int *hull_shift
        def_arg(arg_index++, Sizeof.cl_mem);  // __global float4 *hulls
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int *hull_mesh_ids
        def_arg(arg_index++, Sizeof.cl_mem);  // __global float2 *hull_rotations
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int4 *hull_flags
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int4 *element_tables
        def_arg(arg_index++, Sizeof.cl_mem);  // __global float4 *bounds
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int4 *bounds_index_data
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int2 *bounds_bank_data
        System.out.printf("set %d args for %s\n", arg_index, this.getClass().getSimpleName());
    }

    public void set_hull_shift(Pointer hull_shift)
    {
        new_arg(0, Sizeof.cl_mem, hull_shift);
    }

    public void set_hulls(Pointer hulls)
    {
        new_arg(1, Sizeof.cl_mem, hulls);
    }

    public void set_hull_mesh_ids(Pointer hulls)
    {
        new_arg(2, Sizeof.cl_mem, hulls);
    }

    public void set_hull_rotations(Pointer hull_rotations)
    {
        new_arg(3, Sizeof.cl_mem, hull_rotations);
    }

    public void set_hull_flags(Pointer hull_flags)
    {
        new_arg(4, Sizeof.cl_mem, hull_flags);
    }

    public void set_element_tables(Pointer element_tables)
    {
        new_arg(5, Sizeof.cl_mem, element_tables);
    }

    public void set_bounds(Pointer element_tables)
    {
        new_arg(6, Sizeof.cl_mem, element_tables);
    }

    public void set_bounds_index(Pointer bounds_index)
    {
        new_arg(7, Sizeof.cl_mem, bounds_index);
    }

    public void set_bounds_bank(Pointer bounds_bank)
    {
        new_arg(8, Sizeof.cl_mem, bounds_bank);
    }
}
