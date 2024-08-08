package com.controllerface.bvge.gpu.gl.buffers;

import com.controllerface.bvge.gpu.GPUResource;

import java.nio.FloatBuffer;
import java.util.Objects;

import static org.lwjgl.opengl.GL15C.*;
import static org.lwjgl.opengl.GL45C.*;

public record GL_VertexBuffer(int id) implements GPUResource
{
    @Override
    public void release()
    {
        glDeleteBuffers(id);
    }

    public void load_float_data(float[] data)
    {
        glNamedBufferData(id, data, GL_DYNAMIC_DRAW);
    }

    public void load_float_sub_data(float[] data, int offset)
    {
        glNamedBufferSubData(id, offset, data);
    }

    public FloatBuffer map_as_float_buffer()
    {
        return Objects.requireNonNull(glMapNamedBuffer(id, GL_WRITE_ONLY)).asFloatBuffer();
    }

    public void unmap_buffer()
    {
        glUnmapNamedBuffer(id);
    }
}
