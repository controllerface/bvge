package com.controllerface.bvge;

import com.controllerface.bvge.cl.GPGPU;
import com.controllerface.bvge.editor.Editor;
import com.controllerface.bvge.window.Window;
import org.lwjgl.system.Configuration;

import java.io.File;
import java.io.IOException;
import java.util.logging.FileHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

public class Main
{
    public static void main(String[] args)
    {
        System.out.println("Java Version: " + System.getProperty("java.version"));
        System.out.println("Working Directory: " + System.getProperty("user.dir"));
//        try
//        {
//            var root = Logger.getLogger("");
//            var file_handler = new FileHandler(System.getProperty("user.dir") + File.separator + "test.log");
//            var formatter = new SimpleFormatter();
//            file_handler.setFormatter(formatter);
//            root.addHandler(file_handler);
//            root.setLevel(Level.INFO);
//        }
//        catch (SecurityException | IOException e)
//        {
//            e.printStackTrace();
//        }

        Configuration.DEBUG.set(true);
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

