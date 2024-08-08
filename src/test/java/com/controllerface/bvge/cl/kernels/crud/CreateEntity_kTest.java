package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.kernels.crud.CreateEntity_k;
import org.junit.jupiter.api.Test;


class CreateEntity_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CreateEntity_k.kernel_source);
    }
}