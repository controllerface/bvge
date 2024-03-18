package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.GPUProgram;
import com.controllerface.bvge.cl.Kernel;

public class BufferTransfer extends GPUProgram
{
    @Override
    public void init()
    {
        src.add(CLUtils.read_src("programs/buffer_transfer.cl"));

        make_program();

        load_kernel(Kernel.buffer_transfer);
        load_kernel(Kernel.verify_buffer_transfer);
    }
}
