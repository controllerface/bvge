package com.controllerface.bvge.core;

import java.util.logging.Level;
import java.util.logging.Logger;

public class CrashHandler implements Thread.UncaughtExceptionHandler
{
    private static final Logger LOGGER = Logger.getLogger(CrashHandler.class.getName());

    @Override
    public void uncaughtException(Thread t, Throwable e)
    {
        LOGGER.log(Level.SEVERE, "Crashed on thread: " + t.getName());
        LOGGER.log(Level.SEVERE, "Stacktrace: ", e);
    }
}
