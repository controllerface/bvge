package com.controllerface.bvge.gpu.gl.buffers;

import com.controllerface.bvge.gpu.GPUResource;

import static org.lwjgl.opengl.GL15C.glBindBuffer;
import static org.lwjgl.opengl.GL15C.glDeleteBuffers;
import static org.lwjgl.opengl.GL40C.GL_DRAW_INDIRECT_BUFFER;
import static org.lwjgl.opengl.GL45C.glNamedBufferSubData;

public record GL_CommandBuffer(int id) implements GPUResource
{
    @Override
    public void release()
    {
        glDeleteBuffers(id);
    }

    public void bind()
    {
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, id);
    }

    public void load_int_sub_data(int[] data, int offset)
    {
        glNamedBufferSubData(id, offset, data);
    }

}
