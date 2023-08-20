package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.GPU.*;
import static com.controllerface.bvge.cl.CLUtils.read_src;

public class ScanKeyBank extends GPUProgram
{
    @Override
    protected void init()
    {
        src.add(func_exclusive_scan);
        src.add(read_src("kernels/scan_key_bank.cl"));

        make_program();

        load_kernel(Kernel.scan_bounds_single_block);
        load_kernel(Kernel.scan_bounds_multi_block);
        load_kernel(Kernel.complete_bounds_multi_block);
    }
}
