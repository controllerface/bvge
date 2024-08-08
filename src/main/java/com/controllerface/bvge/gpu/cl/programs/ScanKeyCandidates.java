package com.controllerface.bvge.gpu.cl.programs;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.CLUtils;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;

public class ScanKeyCandidates extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(func_exclusive_scan);
        src.add(GPU.CL.read_src("programs/scan_key_candidates.cl"));

        make_program();

        load_kernel(KernelType.scan_candidates_single_block_out);
        load_kernel(KernelType.scan_candidates_multi_block_out);
        load_kernel(KernelType.complete_candidates_multi_block_out);

        return this;
    }
}
