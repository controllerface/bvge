package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.CLUtils.read_src;
import static com.controllerface.bvge.cl.GPU.Kernel;

public class ScanInt2Array extends GPUProgram
{
    @Override
    public void init()
    {
        src.add(func_exclusive_scan);
        src.add(read_src("programs/scan_int2_array.cl"));

        make_program();

        load_kernel(Kernel.scan_int2_single_block);
        load_kernel(Kernel.scan_int2_multi_block);
        load_kernel(Kernel.complete_int2_multi_block);
    }
}
