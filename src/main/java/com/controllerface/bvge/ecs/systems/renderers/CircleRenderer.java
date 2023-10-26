package com.controllerface.bvge.ecs.systems.renderers;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.geometry.Models;
import com.controllerface.bvge.gl.AbstractShader;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;

import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_4D_LENGTH;
import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_FLOAT_4D_SIZE;
import static org.jocl.CL.clReleaseMemObject;
import static org.lwjgl.opengl.GL11C.GL_FLOAT;
import static org.lwjgl.opengl.GL11C.glDrawArrays;
import static org.lwjgl.opengl.GL15.GL_POINTS;
import static org.lwjgl.opengl.GL15C.*;
import static org.lwjgl.opengl.GL20.glDisableVertexAttribArray;
import static org.lwjgl.opengl.GL20.glEnableVertexAttribArray;
import static org.lwjgl.opengl.GL20C.glVertexAttribPointer;
import static org.lwjgl.opengl.GL30.glBindVertexArray;
import static org.lwjgl.opengl.GL30.glGenVertexArrays;

public class CircleRenderer extends GameSystem
{
    public static final int CIRCLES_VERTEX_COUNT = Constants.Rendering.MAX_BATCH_SIZE * VECTOR_4D_LENGTH;
    public static final int CIRCLES_BUFFER_SIZE = CIRCLES_VERTEX_COUNT * Float.BYTES;
    private final AbstractShader shader;
    private int vao_id;
    private int circles_vbo;

    public CircleRenderer(ECS ecs)
    {
        super(ecs);
        this.shader = Assets.shader("circle_shader.glsl");
        init();
    }

    public void init()
    {
        // Generate and bind a Vertex Array Object
        vao_id = glGenVertexArrays();
        glBindVertexArray(vao_id);

        // create buffer for transforms, batches will use this during the rendering process
        circles_vbo = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, circles_vbo); // this attribute comes from a different vertex buffer
        glBufferData(GL_ARRAY_BUFFER, CIRCLES_BUFFER_SIZE, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, VECTOR_4D_LENGTH, GL_FLOAT, false, VECTOR_FLOAT_4D_SIZE, 0);

        // share the buffer with the CL context
        GPU.share_memory(circles_vbo);

        // unbind
        glBindBuffer(GL_ARRAY_BUFFER, 0);

    }

    @Override
    public void tick(float dt)
    {
        glBindVertexArray(vao_id);

        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());

        var data = GPU.GL_hull_filter(Models.CIRCLE_PARTICLE);

        glEnableVertexAttribArray(0);

        int offset = 0;
        for (int circles = data.hull_count(); circles > 0; circles -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, circles);
            GPU.GL_circles(circles_vbo, data.hulls_out(), offset, count);
            glDrawArrays(GL_POINTS, 0, count);
            offset += count;
        }

        glDisableVertexAttribArray(0);
        glBindVertexArray(0);
        shader.detach();

        clReleaseMemObject(data.hulls_out());
    }
}
