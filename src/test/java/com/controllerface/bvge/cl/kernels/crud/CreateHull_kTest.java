package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.kernels.crud.CreateHull_k;
import org.junit.jupiter.api.Test;

class CreateHull_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CreateHull_k.kernel_source);
    }
}