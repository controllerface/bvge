package com.controllerface.bvge.editor;

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
}
