package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.kernels.crud.CreateModelTransform_k;
import org.junit.jupiter.api.Test;

class CreateModelTransform_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CreateModelTransform_k.kernel_source);
    }
}