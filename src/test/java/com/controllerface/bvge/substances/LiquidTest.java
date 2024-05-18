package com.controllerface.bvge.substances;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class LiquidTest
{
    @Test
    public void lookup_table()
    {
        System.out.println(Liquid.cl_lookup_table());
    }
}