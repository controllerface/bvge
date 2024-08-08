package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.kernels.crud.CreateTextureUV_k;
import org.junit.jupiter.api.Test;

class CreateTextureUV_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CreateTextureUV_k.kernel_source);
    }
}