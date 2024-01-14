package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class WriteMeshData_k extends GPUKernel
{
    public WriteMeshData_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.write_mesh_data), 7);
        int arg_index = 0;
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int *hull_mesh_ids,
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int4 *mesh_references,
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int *counters,
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int *query,
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int *offsets,
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int4 *count_data,
        def_arg(arg_index++, Sizeof.cl_int);  // int count
        System.out.printf("set %d args for %s\n", arg_index, this.getClass().getSimpleName());
    }

    public void set_mesh_ids(Pointer mesh_ids)
    {
        new_arg(0, Sizeof.cl_mem, mesh_ids);
    }

    public void set_mesh_refs(Pointer mesh_refs)
    {
        new_arg(1, Sizeof.cl_mem, mesh_refs);
    }

}
