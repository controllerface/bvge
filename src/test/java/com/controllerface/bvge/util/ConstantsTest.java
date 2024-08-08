package com.controllerface.bvge.util;

import com.controllerface.bvge.game.Constants;
import org.junit.jupiter.api.Test;

class ConstantsTest
{
    @Test
    public void generate_flag_constants()
    {
        System.out.println(Constants.hull_flags_src());
        System.out.println("-");
        System.out.println(Constants.entity_flags_src());
        System.out.println("-");
        System.out.println(Constants.edge_flags_src());
        System.out.println("-");
        System.out.println(Constants.point_flags_src());
    }
}