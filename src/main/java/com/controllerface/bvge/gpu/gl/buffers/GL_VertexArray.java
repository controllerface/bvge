package com.controllerface.bvge.gpu.gl.buffers;

import com.controllerface.bvge.gpu.GPUResource;

import static org.lwjgl.opengl.GL30C.glBindVertexArray;
import static org.lwjgl.opengl.GL30C.glDeleteVertexArrays;
import static org.lwjgl.opengl.GL45C.glEnableVertexArrayAttrib;
import static org.lwjgl.opengl.GL45C.glVertexArrayBindingDivisor;

public record GL_VertexArray(int gl_id) implements GPUResource
{
    public void bind()
    {
        glBindVertexArray(gl_id);
    }

    public void unbind()
    {
        glBindVertexArray(0);
    }

    public void enable_attribute(int attr)
    {
        glEnableVertexArrayAttrib(gl_id, attr);
    }

    public void instance_attribute(int attr, int divisor)
    {
        glVertexArrayBindingDivisor(gl_id, attr, divisor);
    }

    @Override
    public void release()
    {
        glDeleteVertexArrays(gl_id);
    }
}
