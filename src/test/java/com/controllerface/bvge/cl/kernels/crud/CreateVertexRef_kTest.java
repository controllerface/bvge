package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.kernels.crud.CreateVertexRef_k;
import org.junit.jupiter.api.Test;

class CreateVertexRef_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CreateVertexRef_k.kernel_source);
    }
}