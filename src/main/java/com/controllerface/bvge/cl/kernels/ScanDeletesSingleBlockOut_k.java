package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class ScanDeletesSingleBlockOut_k extends GPUKernel
{
    public ScanDeletesSingleBlockOut_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.scan_deletes_single_block_out), 9);
        def_arg(0, Sizeof.cl_mem); // armature flags
        def_arg(1, Sizeof.cl_mem); // hull tables
        def_arg(2, Sizeof.cl_mem); // element tables
        def_arg(3, Sizeof.cl_mem); // output buffer
        def_arg(4, Sizeof.cl_mem); // output buffer 2
        def_arg(5, Sizeof.cl_mem); // counter
        def_arg(6, -1); // local buffer 1
        def_arg(7, -1); // local buffer 2
        def_arg(8, Sizeof.cl_int); // total count
    }

    public void set_armature_flags(Pointer armature_flags)
    {
        new_arg(0, Sizeof.cl_mem, armature_flags);
    }

    public void set_hull_tables(Pointer hull_tables)
    {
        new_arg(1, Sizeof.cl_mem, hull_tables);
    }

    public void set_element_tables(Pointer element_tables)
    {
        new_arg(2, Sizeof.cl_mem, element_tables);
    }
}
