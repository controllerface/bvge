package com.controllerface.bvge.gpu.gl;

import com.controllerface.bvge.gpu.GPUResource;

import static org.lwjgl.glfw.Callbacks.glfwFreeCallbacks;
import static org.lwjgl.glfw.GLFW.*;

public class GL_GraphicsController implements GPUResource
{
    public final long window_handle;

    public GL_GraphicsController(long windowHandle)
    {
        window_handle = windowHandle;
    }

    public void swap_buffers()
    {
        glfwSwapBuffers(window_handle);
    }

    public boolean should_close()
    {
        return glfwWindowShouldClose(window_handle);
    }

    public void show_window()
    {
        glfwShowWindow(window_handle);
    }

    @Override
    public void release()
    {
        glfwFreeCallbacks(window_handle);
        glfwDestroyWindow(window_handle);
    }
}
