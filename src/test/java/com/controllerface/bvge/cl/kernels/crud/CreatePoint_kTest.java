package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.kernels.crud.CreatePoint_k;
import org.junit.jupiter.api.Test;

class CreatePoint_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CreatePoint_k.kernel_source);
    }
}