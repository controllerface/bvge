package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.CLUtils.read_src;
import static com.controllerface.bvge.cl.GPU.Kernel;

public class ScanInt4Array extends GPUProgram
{
    @Override
    protected void init()
    {
        src.add(func_exclusive_scan);
        src.add(read_src("kernels/scan_int4_array.cl"));

        make_program();

        load_kernel(Kernel.scan_int4_single_block);
        load_kernel(Kernel.scan_int4_multi_block);
        load_kernel(Kernel.complete_int4_multi_block);
    }
}
