package com.controllerface.bvge.gpu.cl.programs.scan;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.kernels.compact.*;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class ScanDeletes extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(const_entity_flags);
        src.add(const_hull_flags);
        src.add(func_exclusive_scan);
        src.add(CompactPoints_k.kernel_source);
        src.add(CompactEdges_k.kernel_source);
        src.add(CompactHulls_k.kernel_source);
        src.add(CompactHullBones_k.kernel_source);
        src.add(CompactEntityBones_k.kernel_source);
        src.add(GPU.CL.read_src("programs/scan_deletes.cl"));

        make_program();

        load_kernel(KernelType.scan_deletes_single_block_out);
        load_kernel(KernelType.scan_deletes_multi_block_out);
        load_kernel(KernelType.complete_deletes_multi_block_out);
        load_kernel(KernelType.compact_entities);
        load_kernel(KernelType.compact_hulls);
        load_kernel(KernelType.compact_edges);
        load_kernel(KernelType.compact_points);
        load_kernel(KernelType.compact_hull_bones);
        load_kernel(KernelType.compact_entity_bones);

        return this;
    }
}
