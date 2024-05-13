package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.cl.*;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.Component;
import com.controllerface.bvge.ecs.components.ControlPoints;
import com.controllerface.bvge.ecs.components.GameComponent;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.gl.GLUtils;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.window.Window;

import java.util.Map;
import java.util.Objects;

import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_FLOAT_4D_SIZE;
import static org.lwjgl.opengl.ARBDirectStateAccess.glCreateVertexArrays;
import static org.lwjgl.opengl.GL11C.glDrawArrays;
import static org.lwjgl.opengl.GL15C.*;
import static org.lwjgl.opengl.GL30C.glBindVertexArray;
import static org.lwjgl.opengl.GL30C.glDeleteVertexArrays;
import static org.lwjgl.opengl.GL45C.glEnableVertexArrayAttrib;
import static org.lwjgl.opengl.GL45C.glNamedBufferData;

public class MouseRenderer extends GameSystem
{
    private static final int TRANSFORM_ATTRIBUTE = 0;

    private int vao;
    private int vbo;
    private long vbo_ptr;

    private Shader shader;

    public MouseRenderer(ECS ecs)
    {
        super(ecs);
        init_GL();
        init_CL();
    }

    public void init_GL()
    {
        shader = Assets.load_shader("mouse_shader.glsl");
        vao = glCreateVertexArrays();
        vbo = GLUtils.new_buffer_vec4(vao, TRANSFORM_ATTRIBUTE, VECTOR_FLOAT_4D_SIZE);
        glEnableVertexArrayAttrib(vao, TRANSFORM_ATTRIBUTE);
    }

    private void init_CL()
    {
        vbo_ptr = GPGPU.share_memory(vbo);
    }

    @Override
    public void tick(float dt)
    {
        var control_components = ecs.getComponents(Component.ControlPoints);
        ControlPoints control_points = null;
        for (Map.Entry<String, GameComponent> entry : control_components.entrySet())
        {
            GameComponent component = entry.getValue();
            control_points = Component.ControlPoints.coerce(component);
        }

        assert control_points != null : "Component was null";
        Objects.requireNonNull(control_points);

        var camera = Window.get().camera();
        float world_x = control_points.get_screen_target().x * camera.get_zoom() + camera.position.x;
        float world_y = (Window.get().height() - control_points.get_screen_target().y) * camera.get_zoom() + camera.position.y;
        control_points.get_world_target().set(world_x, world_y);
        float[] loc = { world_x, world_y, 25.0f, 25.0f };

        glNamedBufferData(vbo, loc, GL_DYNAMIC_DRAW);
        glBindVertexArray(vao);

        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());

        glDrawArrays(GL_POINTS, 0, 1);

        glBindVertexArray(0);
        shader.detach();
    }

    @Override
    public void shutdown()
    {
        glDeleteVertexArrays(vao);
        glDeleteBuffers(vbo);
        shader.destroy();
        GPGPU.cl_release_buffer(vbo_ptr);
    }
}
