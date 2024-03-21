package com.controllerface.bvge;

import com.controllerface.bvge.cl.GPGPU;
import com.controllerface.bvge.window.Window;


public class Main
{
    public static void main(String[] args)
    {
        System.out.println("Java Version: " + System.getProperty("java.version"));

        //Configuration.DISABLE_CHECKS.set(true);
        Window window = Window.get();
        window.init();

        GPGPU.init();

        window.initGameMode();
        window.run();

        GPGPU.destroy();
    }
}

