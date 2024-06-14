package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.Kernel;

public class ScanInt4Array extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(func_exclusive_scan);
        src.add(CLUtils.read_src("programs/scan_int4_array.cl"));

        make_program();

        load_kernel(Kernel.scan_int4_single_block);
        load_kernel(Kernel.scan_int4_multi_block);
        load_kernel(Kernel.complete_int4_multi_block);

        return this;
    }
}
