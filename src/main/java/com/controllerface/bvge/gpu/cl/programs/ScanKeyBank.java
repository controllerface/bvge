package com.controllerface.bvge.gpu.cl.programs;

import com.controllerface.bvge.gpu.cl.CLUtils;
import com.controllerface.bvge.gpu.cl.kernels.Kernel;

public class ScanKeyBank extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(func_exclusive_scan);
        src.add(CLUtils.read_src("programs/scan_key_bank.cl"));

        make_program();

        load_kernel(Kernel.scan_bounds_single_block);
        load_kernel(Kernel.scan_bounds_multi_block);
        load_kernel(Kernel.complete_bounds_multi_block);

        return this;
    }
}
