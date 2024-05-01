package com.controllerface.bvge;

import com.controllerface.bvge.cl.GPGPU;
import com.controllerface.bvge.editor.Editor;
import com.controllerface.bvge.window.Window;
import org.lwjgl.system.Configuration;

public class Main
{
    public static void main(String[] args)
    {
        System.out.println("Java Version: " + System.getProperty("java.version"));

        //Configuration.DISABLE_CHECKS.set(true);
        Window window = Window.get();
        window.init();

        GPGPU.init();
        Editor.init();

        window.init_game_mode();
        try
        {
            window.run();
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
        finally
        {
            Editor.destroy();
            GPGPU.destroy();
        }
    }
}

