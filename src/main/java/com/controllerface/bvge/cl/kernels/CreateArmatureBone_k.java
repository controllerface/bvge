package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CreateArmatureBone_k extends GPUKernel<CreateArmatureBone_k.Args>
{
    private static final GPU.Program program = GPU.Program.gpu_crud;
    private static final GPU.Kernel kernel = GPU.Kernel.create_armature_bone;

    public enum Args implements GPUKernelArg
    {
        armature_bones(Sizeof.cl_mem),
        bone_bind_tables(Sizeof.cl_mem),
        target(Sizeof.cl_int),
        new_armature_bone(Sizeof.cl_float16),
        new_bone_bind_table(Sizeof.cl_int2);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CreateArmatureBone_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel), Args.values());
    }
}
