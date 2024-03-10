package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.GPUProgram;
import com.controllerface.bvge.cl.Kernel;

public class ScanKeyBank extends GPUProgram
{
    @Override
    public void init()
    {
        src.add(func_exclusive_scan);
        src.add(CLUtils.read_src("programs/scan_key_bank.cl"));

        make_program();

        load_kernel(Kernel.scan_bounds_single_block);
        load_kernel(Kernel.scan_bounds_multi_block);
        load_kernel(Kernel.complete_bounds_multi_block);
    }
}
