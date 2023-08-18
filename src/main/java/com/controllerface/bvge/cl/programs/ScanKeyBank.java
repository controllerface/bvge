package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.GPU.*;
import static com.controllerface.bvge.cl.CLUtils.read_src;

public class ScanKeyBank extends GPUProgram
{
    @Override
    protected void init()
    {
        add_src(func_exclusive_scan);
        add_src(read_src("kernels/scan_key_bank.cl"));

        make_program();

        make_kernel(kn_scan_bounds_single_block);
        make_kernel(kn_scan_bounds_multi_block);
        make_kernel(kn_complete_bounds_multi_block);
    }
}
