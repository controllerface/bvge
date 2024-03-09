package com.controllerface.bvge.gl;

import static com.controllerface.bvge.util.Constants.Rendering.*;
import static org.lwjgl.opengl.GL45C.*;

public class GLUtils
{
    private static final int DEFAULT_OFFSET = 0;
    private static final int DEFAULT_STRIDE = 0;

    public static int new_buffer_float(int vao,
                                       int bind_index,
                                       int buffer_size)
    {
        return dynamic_float_buffer(vao, bind_index, buffer_size, SCALAR_LENGTH, SCALAR_FLOAT_SIZE);
    }

    public static int new_buffer_vec2(int vao,
                                      int bind_index,
                                      int buffer_size)
    {
        return dynamic_float_buffer(vao, bind_index, buffer_size, VECTOR_2D_LENGTH, VECTOR_FLOAT_2D_SIZE);
    }

    public static int fill_buffer_vec2(int vao,
                                       int bind_index,
                                       float[] buffer_data)
    {
        return static_float_buffer(vao, bind_index, VECTOR_2D_LENGTH, VECTOR_FLOAT_2D_SIZE, buffer_data);
    }

    public static int new_buffer_vec4(int vao,
                                      int bind_index,
                                      int buffer_size)
    {
        return dynamic_float_buffer(vao, bind_index, buffer_size, VECTOR_4D_LENGTH, VECTOR_FLOAT_4D_SIZE);
    }

    private static int dynamic_float_buffer(int vao,
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
            GL_FLOAT,
            data_count,
            data_size,
            DEFAULT_STRIDE,
            GL_DYNAMIC_DRAW);
    }

    private static int static_float_buffer(int vao,
                                           int bind_index,
                                           int data_count,
                                           int data_size,
                                           float[] buffer_data)
    {
        return create_vertex_buffer(vao,
            DEFAULT_OFFSET,
            bind_index,
            bind_index,
            GL_FLOAT,
            data_count,
            data_size,
            DEFAULT_STRIDE,
            GL_STATIC_DRAW,
            buffer_data);
    }

    private static int create_vertex_buffer(int vao,
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
        glVertexArrayAttribFormat(vao, bind_index, data_count, data_type, false, data_stride);
        glVertexArrayAttribBinding(vao, attribute_index, bind_index);
        return buffer;
    }

    private static int create_vertex_buffer(int vao,
                                            int buffer_offset,
                                            int bind_index,
                                            int attribute_index,
                                            int data_type,
                                            int data_count,
                                            int data_size,
                                            int data_stride,
                                            int flags,
                                            float[] buffer_data)
    {
        int buffer = glCreateBuffers();
        glNamedBufferData(buffer, buffer_data, flags);
        glVertexArrayVertexBuffer(vao, bind_index, buffer, buffer_offset, data_size);
        glVertexArrayAttribFormat(vao, bind_index, data_count, data_type, false, data_stride);
        glVertexArrayAttribBinding(vao, attribute_index, bind_index);
        return buffer;
    }
}
