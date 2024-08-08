package com.controllerface.bvge.gpu.cl.programs;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.CLUtils;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;

public class ScanIntArrayOut extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(func_exclusive_scan);
        src.add(GPU.CL.read_src("programs/scan_int_array_out.cl"));

        make_program();

        load_kernel(KernelType.scan_int_single_block_out);
        load_kernel(KernelType.scan_int_multi_block_out);
        load_kernel(KernelType.complete_int_multi_block_out);

        return this;
    }
}
