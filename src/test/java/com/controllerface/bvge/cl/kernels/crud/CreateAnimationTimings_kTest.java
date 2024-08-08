package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.kernels.crud.CreateAnimationTimings_k;
import org.junit.jupiter.api.Test;

class CreateAnimationTimings_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CreateAnimationTimings_k.kernel_source);
    }
}