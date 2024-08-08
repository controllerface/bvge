package com.controllerface.bvge.gpu.gl.buffers;

import com.controllerface.bvge.gpu.GPUResource;

import static org.lwjgl.opengl.GL15C.glDeleteBuffers;

public record GL_ElementBuffer(int id) implements GPUResource
{
    @Override
    public void release()
    {
        glDeleteBuffers(id);
    }
}
