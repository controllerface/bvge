package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.kernels.crud.CreateKeyFrame_k;
import org.junit.jupiter.api.Test;

class CreateKeyFrame_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CreateKeyFrame_k.kernel_source);
    }
}