package com.controllerface.bvge.cl.kernels.compact;

import com.controllerface.bvge.gpu.cl.kernels.compact.CompactEntityBones_k;
import org.junit.jupiter.api.Test;

class CompactEntityBones_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CompactEntityBones_k.kernel_source);
    }
}