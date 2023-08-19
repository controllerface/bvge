package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.GPU.*;
import static com.controllerface.bvge.cl.CLUtils.read_src;

public class ScanIntArray extends GPUProgram
{
    @Override
    protected void init()
    {
        add_src(func_exclusive_scan);
        add_src(read_src("kernels/scan_int_array.cl"));

        make_program();

        make_kernel(Kernel.scan_int_single_block);
        make_kernel(Kernel.scan_int_multi_block);
        make_kernel(Kernel.complete_int_multi_block);
    }
}
