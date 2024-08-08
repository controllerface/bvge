package com.controllerface.bvge.gpu.cl.programs;

import com.controllerface.bvge.gpu.cl.CLUtils;
import com.controllerface.bvge.gpu.cl.kernels.Kernel;

public class ScanIntArrayOut extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(func_exclusive_scan);
        src.add(CLUtils.read_src("programs/scan_int_array_out.cl"));

        make_program();

        load_kernel(Kernel.scan_int_single_block_out);
        load_kernel(Kernel.scan_int_multi_block_out);
        load_kernel(Kernel.complete_int_multi_block_out);

        return this;
    }
}
