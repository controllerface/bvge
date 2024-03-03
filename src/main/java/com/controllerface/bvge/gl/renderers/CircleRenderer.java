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
import org.lwjgl.opencl.CL12;

import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_4D_LENGTH;
import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_FLOAT_4D_SIZE;
import static org.lwjgl.opencl.CL12.*;
import static org.lwjgl.opengl.ARBDirectStateAccess.glCreateVertexArrays;
import static org.lwjgl.opengl.GL11C.glDrawArrays;
import static org.lwjgl.opengl.GL15C.*;
import static org.lwjgl.opengl.GL20C.*;
import static org.lwjgl.opengl.GL30C.GL_FLOAT;
import static org.lwjgl.opengl.GL30C.*;
import static org.lwjgl.opengl.GL45C.*;

public class CircleRenderer extends GameSystem
{
    public static final int CIRCLES_BUFFER_SIZE = Constants.Rendering.MAX_BATCH_SIZE * VECTOR_FLOAT_4D_SIZE;

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
        vao_id = glCreateVertexArrays();

        circles_vbo = glCreateBuffers();
        glNamedBufferData(circles_vbo, CIRCLES_BUFFER_SIZE, GL_DYNAMIC_DRAW);
        glVertexArrayVertexBuffer(vao_id, 0, circles_vbo, 0, VECTOR_FLOAT_4D_SIZE);
        glEnableVertexArrayAttrib(vao_id, 0);
        glVertexArrayAttribFormat(vao_id, 0, VECTOR_4D_LENGTH, GL_FLOAT, false, 0);
        glVertexArrayAttribBinding(vao_id, 0, 0);

        GPU.share_memory(circles_vbo);
    }

    @Override
    public void tick(float dt)
    {
        if (circle_hulls != null && circle_hulls.indices() != -1)
        {
            clReleaseMemObject(circle_hulls.indices());
        }
        circle_hulls = GPU.GL_hull_filter(Models.CIRCLE_PARTICLE);

        if (circle_hulls.count() == 0) return;

        glBindVertexArray(vao_id);

        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());

        glEnableVertexArrayAttrib(vao_id, 0);

        int offset = 0;
        for (int remaining = circle_hulls.count(); remaining > 0; remaining -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, remaining);
            GPU.GL_circles(circles_vbo, circle_hulls.indices(), offset, count);
            glDrawArrays(GL_POINTS, 0, count);
            offset += count;
        }

        glBindVertexArray(0);
        shader.detach();
    }
}
