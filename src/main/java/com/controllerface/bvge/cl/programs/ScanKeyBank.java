package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.CLUtils.read_src;
import static com.controllerface.bvge.cl.GPU.Kernel;

public class ScanKeyBank extends GPUProgram
{
    @Override
    public void init()
    {
        src.add(func_exclusive_scan);
        src.add(read_src("programs/scan_key_bank.cl"));

        make_program();

        load_kernel(Kernel.scan_bounds_single_block);
        load_kernel(Kernel.scan_bounds_multi_block);
        load_kernel(Kernel.complete_bounds_multi_block);
    }
}
