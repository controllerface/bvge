package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.kernels.crud.CreateMeshFace_k;
import org.junit.jupiter.api.Test;

class CreateMeshFace_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CreateMeshFace_k.kernel_source);
    }
}