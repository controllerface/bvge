package com.controllerface.bvge.editor;

import com.controllerface.bvge.substances.Solid;

import java.util.Collections;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.Map;

public class Editor
{
    private static final int DEFAULT_PORT = 9000;
    private static EditorServer server;

    public static final boolean ACTIVE = true;



    public static void init()
    {
        if (ACTIVE)
        {
            server = new EditorServer(DEFAULT_PORT);
            server.start();
        }
    }

    public static void destroy()
    {
        if (ACTIVE)
        {
            server.stop();
        }
    }

    public static void queue_event(String name, String data)
    {
        if (ACTIVE)
        {
            server.queue_stat_event(name, data);
        }
    }

    public static void inventory(int id, int qty)
    {
        if (ACTIVE)
        {
            server.inventory(id, qty);
        }
    }
}
