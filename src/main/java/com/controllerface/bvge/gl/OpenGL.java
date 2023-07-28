package com.controllerface.bvge.gl;

import com.controllerface.bvge.cl.OpenCL;

import static org.lwjgl.opengl.GL11.GL_FLOAT;
import static org.lwjgl.opengl.GL15.*;
import static org.lwjgl.opengl.GL15.GL_DYNAMIC_DRAW;
import static org.lwjgl.opengl.GL20C.glVertexAttribPointer;
import static org.lwjgl.opengl.GL30.glBindVertexArray;
import static org.lwjgl.opengl.GL30.glGenVertexArrays;

public class OpenGL
{
    private static final int VERTEX_SIZE = 4; // a vertex is 4 floats (x, y, x_prv, y_prv)
    private static final int VERTEX_SIZE_BYTES = VERTEX_SIZE * Float.BYTES;

    private static int vaoID, vboID;

    public static void init(int buffer_size)
    {
        // Generate and bind a Vertex Array Object
        vaoID = glGenVertexArrays();
        glBindVertexArray(vaoID); // this sets this VAO as being active

        // generate and bind a Vertex Buffer Object, note that because our vao is currently active,
        // the new vbo is "attached" to it implicitly
        vboID = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, vboID);
        // allocates enough space for the vertices we can handle in this batch, but doesn't transfer any data
        // into the buffer just yet

        // todo: figure out buffer size
        glBufferData(GL_ARRAY_BUFFER, buffer_size, GL_DYNAMIC_DRAW);

        // todo: bind the entire vertex buffer here instead
        // bind the buffer to the CL context
        OpenCL.bindvertexVBO(vboID);

        // define the buffer attribute pointers
        glVertexAttribPointer(0, VERTEX_SIZE, GL_FLOAT, false, VERTEX_SIZE_BYTES, 0);

        // unbind the vao since we're done defining things for now
        glBindVertexArray(0);
    }
}
