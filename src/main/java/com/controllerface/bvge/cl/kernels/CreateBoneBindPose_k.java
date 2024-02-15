package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CreateBoneBindPose_k extends GPUKernel<CreateBoneBindPose_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        bone_bind_poses(Sizeof.cl_mem),
        bone_bind_parents(Sizeof.cl_mem),
        target(Sizeof.cl_int),
        new_bone_bind_pose(Sizeof.cl_float16),
        bone_bind_parent(Sizeof.cl_int);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CreateBoneBindPose_k(long command_queue_ptr)
    {
        super(command_queue_ptr, GPU.Program.gpu_crud.gpu.kernels().get(GPU.Kernel.create_bone_bind_pose), Args.values());
    }
}
