package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelArg;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.groups.ReferenceBufferGroup;

import static com.controllerface.bvge.memory.types.ReferenceBufferType.MESH_FACE_TABLE;
import static com.controllerface.bvge.memory.types.ReferenceBufferType.MESH_VERTEX_TABLE;

public class CreateMeshReference_k extends GPUKernel
{
    public static final String kernel_source = GPU.CL.crud_create_k_src(KernelType.create_mesh_reference, Args.class);

    public enum Args implements KernelArg
    {
        mesh_vertex_tables    (CL_DataTypes.cl_int2.buffer_name()),
        mesh_face_tables      (CL_DataTypes.cl_int2.buffer_name()),
        target                (CL_DataTypes.cl_int.name()),
        new_mesh_vertex_table (CL_DataTypes.cl_int2.name()),
        new_mesh_face_table   (CL_DataTypes.cl_int2.name()),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CreateMeshReference_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.create_mesh_reference));
    }

    public GPUKernel init(ReferenceBufferGroup reference_buffers)
    {
        return this.buf_arg(Args.mesh_vertex_tables, reference_buffers.buffer(MESH_VERTEX_TABLE))
            .buf_arg(Args.mesh_face_tables, reference_buffers.buffer(MESH_FACE_TABLE));
    }
}
