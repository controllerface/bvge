package com.controllerface.bvge.core;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.editor.Editor;
import com.controllerface.bvge.events.EventBus;
import com.controllerface.bvge.game.InputSystem;
import com.controllerface.bvge.gpu.GPU;
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
    private static final String WINDOW_TITLE = "BVGE Prototype";

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
        Window window = Window.get();
        Editor.init();

        var ecs             = new ECS();
        var event_bus       = new EventBus();
        var input_system    = new InputSystem(ecs);

        GPU.startup(ecs, event_bus, input_system, WINDOW_TITLE);
        window.init(ecs, event_bus, input_system);
        window.init_game_mode();

        try { window.show(); }
        catch (Exception e) { throw new RuntimeException("Unexpected error", e); }
        finally
        {
            GPU.shutdown();
            input_system.shutdown();
            event_bus.clear();
            ecs.shutdown();
            Editor.destroy();
        }
    }
}

