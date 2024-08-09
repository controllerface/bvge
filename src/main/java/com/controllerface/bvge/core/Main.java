package com.controllerface.bvge.core;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.editor.Editor;
import com.controllerface.bvge.gpu.cl.GPGPU;
import org.lwjgl.Version;
import org.lwjgl.system.Configuration;
import org.lwjgl.util.freetype.FreeType;

import java.io.File;
import java.io.IOException;
import java.util.logging.FileHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Main
{
    private static final Logger LOGGER = Logger.getLogger(Main.class.getName());

    public static void main(String[] args)
    {
        Thread.setDefaultUncaughtExceptionHandler(new CrashHandler());
        configure_logger();
        configure_libraries();
        version_info();
        run();
    }

    private static void version_info()
    {
        LOGGER.log(Level.INFO, "Java Version: " + System.getProperty("java.version"));
        LOGGER.log(Level.INFO, "LWJGL version: " + Version.getVersion());
    }

    private static void configure_logger()
    {
        try
        {
            var root = Logger.getLogger("");
            var file_handler = new FileHandler(System.getProperty("user.dir") + File.separator + "out.log");
            var formatter = new LogFormatter();
            file_handler.setFormatter(formatter);
            for (var handler : root.getHandlers())
            {
                root.removeHandler(handler);
            }
            root.addHandler(file_handler);
            root.setLevel(Level.INFO);
        }
        catch (SecurityException | IOException e)
        {
            throw new RuntimeException("Could not initialize logger", e);
        }
    }

    private static void configure_libraries()
    {
        Configuration.HARFBUZZ_LIBRARY_NAME.set(FreeType.getLibrary());
    }

    private static void run()
    {

        ECS ecs = new ECS();
        Window window = Window.get();
        window.init(ecs);

        GPGPU.init(ecs);
        Editor.init();

        window.init_game_mode();
        try { window.run(); }
        catch (Exception e) { throw new RuntimeException("Unexpected error", e); }
        finally
        {
            Editor.destroy();
            GPGPU.destroy();
            ecs.shutdown();
        }
    }
}

