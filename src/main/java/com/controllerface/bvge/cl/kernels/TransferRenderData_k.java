package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class TransferRenderData_k extends GPUKernel
{
    public TransferRenderData_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.transfer_render_data), 15);
        int arg_index = 0;
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int4 *hull_element_tables
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int *hull_mesh_ids
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int4 *mesh_references
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int4 *mesh_faces
        def_arg(arg_index++, Sizeof.cl_mem);  // __global float4 *points
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int4 *vertex_tables
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int2 *uv_tables
        def_arg(arg_index++, Sizeof.cl_mem);  // __global float2 *texture_uvs
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int *command_buffer
        def_arg(arg_index++, Sizeof.cl_mem);  // __global float2 *vertex_buffer
        def_arg(arg_index++, Sizeof.cl_mem);  // __global float2 *uv_buffer
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int *element_buffer
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int4 *mesh_details
        def_arg(arg_index++, Sizeof.cl_mem);  // __global int2 *mesh_transfer
        def_arg(arg_index++, Sizeof.cl_int);  // int offset
        System.out.printf("set %d args for %s\n", arg_index, this.getClass().getSimpleName());
    }

    public void set_element_tables(Pointer element_tables)
    {
        new_arg(0, Sizeof.cl_mem, element_tables);
    }

    public void set_mesh_ids(Pointer mesh_ids)
    {
        new_arg(1, Sizeof.cl_mem, mesh_ids);
    }

    public void set_mesh_refs(Pointer mesh_refs)
    {
        new_arg(2, Sizeof.cl_mem, mesh_refs);
    }

    public void set_mesh_faces(Pointer mesh_faces)
    {
        new_arg(3, Sizeof.cl_mem, mesh_faces);
    }

    public void set_points(Pointer points)
    {
        new_arg(4, Sizeof.cl_mem, points);
    }

    public void set_vertex_tables(Pointer vertex_tables)
    {
        new_arg(5, Sizeof.cl_mem, vertex_tables);
    }

    public void set_uv_tables(Pointer uv_tables)
    {
        new_arg(6, Sizeof.cl_mem, uv_tables);
    }

    public void set_texture_uvs(Pointer texture_uvs)
    {
        new_arg(7, Sizeof.cl_mem, texture_uvs);
    }

}
