package com.controllerface.bvge.cl.kernels.compact;

import com.controllerface.bvge.gpu.cl.kernels.compact.CompactHullBones_k;
import org.junit.jupiter.api.Test;

class CompactHullBones_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CompactHullBones_k.kernel_source);
    }
}