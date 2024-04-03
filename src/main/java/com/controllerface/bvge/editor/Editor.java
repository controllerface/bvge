package com.controllerface.bvge.editor;

public class Editor
{
    public static final boolean ACTIVE = true;
    private static EditorServer editorServer;

    public static void init()
    {
        if (ACTIVE)
        {
            // start editor server
            editorServer = new EditorServer();
            editorServer.start();
        }
    }

    public static void destroy()
    {
        if (ACTIVE)
        {
            // stop editor server
            editorServer.stop();
        }
    }
}
