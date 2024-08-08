package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.kernels.crud.CreateMeshReference_k;
import org.junit.jupiter.api.Test;

class CreateMeshReference_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CreateMeshReference_k.kernel_source);
    }
}