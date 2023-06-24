package com.controllerface.bvge;

import com.controllerface.bvge.cl.OpenCL;
import com.controllerface.bvge.window.Window;
import org.jocl.*;
import org.lwjgl.system.FunctionProviderLocal;

import java.util.Arrays;

import static com.controllerface.bvge.util.InfoUtil.getDeviceInfoStringUTF8;
import static com.controllerface.bvge.util.InfoUtil.getPlatformInfoStringUTF8;
import static org.jocl.CL.*;


public class Main
{
    public static class Memory
    {

    }

    public static void main(String[] args)
    {
        OpenCL.init();
        Window window = Window.get();
        window.run();
        OpenCL.destroy();
    }
}

