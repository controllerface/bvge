package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.kernels.crud.SetBoneChannelTable_k;
import org.junit.jupiter.api.Test;

class SetBoneChannelTable_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(SetBoneChannelTable_k.kernel_source);
    }
}