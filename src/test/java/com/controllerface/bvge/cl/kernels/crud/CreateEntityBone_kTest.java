package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.kernels.crud.CreateEntityBone_k;
import org.junit.jupiter.api.Test;

class CreateEntityBone_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CreateEntityBone_k.kernel_source);
    }
}