package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GpuKernel;

import static com.controllerface.bvge.cl.OpenCL.*;
import static com.controllerface.bvge.cl.OpenCLUtils.read_src;

public class GpuCrud extends GpuKernel
{
    @Override
    protected void init()
    {
        var source = read_src("kernels/gpu_crud.cl");
        this.program = cl_p(func_rotate_point, source);
        this.kernels.put(kn_update_accel,            cl_k(program, kn_update_accel));
        this.kernels.put(kn_rotate_hull,             cl_k(program, kn_rotate_hull));
        this.kernels.put(kn_read_position,           cl_k(program, kn_read_position));
        this.kernels.put(kn_create_hull,             cl_k(program, kn_create_hull));
        this.kernels.put(kn_create_point,            cl_k(program, kn_create_point));
        this.kernels.put(kn_create_edge,             cl_k(program, kn_create_edge));
        this.kernels.put(kn_create_bone_reference,   cl_k(program, kn_create_bone_reference));
        this.kernels.put(kn_create_vertex_reference, cl_k(program, kn_create_vertex_reference));
        this.kernels.put(kn_create_bone,             cl_k(program, kn_create_bone));
        this.kernels.put(kn_create_armature,         cl_k(program, kn_create_armature));
    }
}
