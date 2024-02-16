package com.controllerface.bvge;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.window.Window;


public class Main
{
    private static final int MAX_HULLS  = 100_000;
    private static final int MAX_POINTS = 1_000_000;

    public static void main(String[] args)
    {
        String bits = System.getProperty("sun.arch.data.model");

        System.out.println("debug: bits=" + bits);

        //Configuration.DISABLE_CHECKS.set(true);
        Window window = Window.get();
        window.init();

        // todo: the maximum number of certain objects should be collapsed into a single "limits"
        //  object and passed in, this will make it cleaner to add more limits, which is needed
        GPU.init(MAX_HULLS, MAX_POINTS);

        window.initGameMode();
        window.run();

        GPU.destroy();
    }
}

