package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.HullIndexData;
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
import static org.lwjgl.opengl.GL30C.GL_FLOAT;
import static org.lwjgl.opengl.GL11C.glDrawArrays;
import static org.lwjgl.opengl.GL15C.GL_ARRAY_BUFFER;
import static org.lwjgl.opengl.GL15C.GL_DYNAMIC_DRAW;
import static org.lwjgl.opengl.GL15C.GL_POINTS;
import static org.lwjgl.opengl.GL15C.glBindBuffer;
import static org.lwjgl.opengl.GL15C.glBufferData;
import static org.lwjgl.opengl.GL15C.glGenBuffers;
import static org.lwjgl.opengl.GL20C.glDisableVertexAttribArray;
import static org.lwjgl.opengl.GL20C.glEnableVertexAttribArray;
import static org.lwjgl.opengl.GL20C.glVertexAttribPointer;
import static org.lwjgl.opengl.GL30C.glBindVertexArray;
import static org.lwjgl.opengl.GL30C.glGenVertexArrays;

public class CircleRenderer extends GameSystem
{
    public static final int CIRCLES_VERTEX_COUNT = Constants.Rendering.MAX_BATCH_SIZE * VECTOR_4D_LENGTH;
    public static final int CIRCLES_BUFFER_SIZE = CIRCLES_VERTEX_COUNT * Float.BYTES;

    private final AbstractShader shader;
    private int vao_id;
    private int circles_vbo;
    private HullIndexData circle_hulls;

    public CircleRenderer(ECS ecs)
    {
        super(ecs);
        this.shader = Assets.load_shader("circle_shader.glsl");
        init();
    }

    public void init()
    {
        vao_id = glGenVertexArrays();
        glBindVertexArray(vao_id);

        circles_vbo = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, circles_vbo);
        glBufferData(GL_ARRAY_BUFFER, CIRCLES_BUFFER_SIZE, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, VECTOR_4D_LENGTH, GL_FLOAT, false, VECTOR_FLOAT_4D_SIZE, 0);
        GPU.share_memory(circles_vbo);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    @Override
    public void tick(float dt)
    {
        if (circle_hulls != null && circle_hulls.indices() != null)
        {
            clReleaseMemObject(circle_hulls.indices());
        }
        circle_hulls = GPU.GL_hull_filter(Models.CIRCLE_PARTICLE);

        if (circle_hulls.count() == 0) return;

        glBindVertexArray(vao_id);

        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());

        glEnableVertexAttribArray(0);

        int offset = 0;
        for (int remaining = circle_hulls.count(); remaining > 0; remaining -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, remaining);
            GPU.GL_circles(circles_vbo, circle_hulls.indices(), offset, count);
            glDrawArrays(GL_POINTS, 0, count);
            offset += count;
        }

        glDisableVertexAttribArray(0);
        glBindVertexArray(0);
        shader.detach();
    }
}
