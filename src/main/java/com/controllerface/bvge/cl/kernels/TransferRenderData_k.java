package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class TransferRenderData_k extends GPUKernel<TransferRenderData_k.Args>
{
    private static final GPU.Program program = GPU.Program.mesh_query;
    private static final GPU.Kernel kernel = GPU.Kernel.transfer_render_data;

    public enum Args implements GPUKernelArg
    {
        hull_element_tables(Sizeof.cl_mem),
        hull_mesh_ids(Sizeof.cl_mem),
        mesh_references(Sizeof.cl_mem),
        mesh_faces(Sizeof.cl_mem),
        points(Sizeof.cl_mem),
        vertex_tables(Sizeof.cl_mem),
        uv_tables(Sizeof.cl_mem),
        texture_uvs(Sizeof.cl_mem),
        command_buffer(Sizeof.cl_mem),
        vertex_buffer(Sizeof.cl_mem),
        uv_buffer(Sizeof.cl_mem),
        element_buffer(Sizeof.cl_mem),
        mesh_details(Sizeof.cl_mem),
        mesh_transfer(Sizeof.cl_mem),
        offset(Sizeof.cl_int);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public TransferRenderData_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel), Args.values());
    }
}
