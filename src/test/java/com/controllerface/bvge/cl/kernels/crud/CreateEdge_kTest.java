package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.kernels.crud.CreateEdge_k;
import org.junit.jupiter.api.Test;

class CreateEdge_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CreateEdge_k.kernel_source);
    }
}