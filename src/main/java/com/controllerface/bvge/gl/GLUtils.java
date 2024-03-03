package com.controllerface.bvge.gl;

import static com.controllerface.bvge.util.Constants.Rendering.*;
import static org.lwjgl.opengl.GL45C.*;

public class GLUtils
{
    private static final int DEFAULT_OFFSET = 0;
    private static final int DEFAULT_STRIDE = 0;
    private static final int DEFAULT_TYPE = GL_FLOAT;
    private static final int DEFAULT_FLAG = GL_DYNAMIC_DRAW;

    public static int create_buffer_float(int vao,
                                          int bind_index,
                                          int buffer_size)
    {
        return create_single_buffer(vao, bind_index, buffer_size, SCALAR_LENGTH, SCALAR_FLOAT_SIZE);
    }

    public static int create_buffer_vec2(int vao,
                                         int bind_index,
                                         int buffer_size)
    {
        return create_single_buffer(vao, bind_index, buffer_size, VECTOR_2D_LENGTH, VECTOR_FLOAT_2D_SIZE);
    }

    public static int create_buffer_vec4(int vao,
                                         int bind_index,
                                         int buffer_size)
    {
        return create_single_buffer(vao, bind_index, buffer_size, VECTOR_4D_LENGTH, VECTOR_FLOAT_4D_SIZE);
    }

    public static int create_single_buffer(int vao,
                                           int bind_index,
                                           int buffer_size,
                                           int data_count,
                                           int data_size)
    {
        return create_vertex_buffer(vao,
            DEFAULT_OFFSET,
            bind_index,
            bind_index,
            buffer_size,
            DEFAULT_TYPE,
            data_count,
            data_size,
            DEFAULT_STRIDE,
            DEFAULT_FLAG);
    }

    public static int create_vertex_buffer(int vao,
                                           int buffer_offset,
                                           int bind_index,
                                           int attribute_index,
                                           int buffer_size,
                                           int data_type,
                                           int data_count,
                                           int data_size,
                                           int data_stride,
                                           int flags)
    {
        int buffer = glCreateBuffers();
        glNamedBufferData(buffer, buffer_size, flags);
        glVertexArrayVertexBuffer(vao, bind_index, buffer, buffer_offset, data_size);
        glEnableVertexArrayAttrib(vao, bind_index);
        glVertexArrayAttribFormat(vao, bind_index, data_count, data_type, false, data_stride);
        glVertexArrayAttribBinding(vao, attribute_index, bind_index);
        return buffer;
    }
}
