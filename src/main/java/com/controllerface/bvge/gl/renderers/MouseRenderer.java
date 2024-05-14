package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.cl.*;
import com.controllerface.bvge.cl.kernels.PrepareTransforms_k;
import com.controllerface.bvge.cl.programs.PrepareTransforms;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.Component;
import com.controllerface.bvge.ecs.components.ControlPoints;
import com.controllerface.bvge.ecs.components.GameComponent;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.geometry.ModelRegistry;
import com.controllerface.bvge.gl.GLUtils;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.window.Window;

import java.util.Map;
import java.util.Objects;

import static com.controllerface.bvge.cl.CLUtils.arg_long;
import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_FLOAT_4D_SIZE;
import static org.lwjgl.opengl.ARBDirectStateAccess.glCreateVertexArrays;
import static org.lwjgl.opengl.GL11C.glDrawArrays;
import static org.lwjgl.opengl.GL15C.*;
import static org.lwjgl.opengl.GL30C.glBindVertexArray;
import static org.lwjgl.opengl.GL30C.glDeleteVertexArrays;
import static org.lwjgl.opengl.GL45C.*;

public class MouseRenderer extends GameSystem
{
    private static final int TRANSFORM_ATTRIBUTE = 0;

    private int vao;
    private int vbo;
    private long vbo_ptr;

    private final GPUProgram prepare_transforms = new PrepareTransforms();

    private HullIndexData cursor_hulls;

    private Shader shader;
    private GPUKernel prepare_transforms_k;


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
        vbo = GLUtils.new_buffer_vec4(vao, TRANSFORM_ATTRIBUTE, VECTOR_FLOAT_4D_SIZE * 2);
        glEnableVertexArrayAttrib(vao, TRANSFORM_ATTRIBUTE);
    }

    private void init_CL()
    {
        vbo_ptr = GPGPU.share_memory(vbo);

        prepare_transforms.init();

        long ptr = prepare_transforms.kernel_ptr(Kernel.prepare_transforms);
        prepare_transforms_k = (new PrepareTransforms_k(GPGPU.cl_cmd_queue_ptr, ptr))
            .ptr_arg(PrepareTransforms_k.Args.transforms_out, vbo_ptr)
            .buf_arg(PrepareTransforms_k.Args.hull_positions, GPGPU.core_memory.buffer(BufferType.MIRROR_HULL))
            .buf_arg(PrepareTransforms_k.Args.hull_scales, GPGPU.core_memory.buffer(BufferType.MIRROR_HULL_SCALE))
            .buf_arg(PrepareTransforms_k.Args.hull_rotations, GPGPU.core_memory.buffer(BufferType.MIRROR_HULL_ROTATION));
    }

    @Override
    public void tick(float dt)
    {
        if (cursor_hulls != null && cursor_hulls.indices() != -1)
        {
            GPGPU.cl_release_buffer(cursor_hulls.indices());
        }
        cursor_hulls = GPGPU.GL_hull_filter(GPGPU.cl_cmd_queue_ptr, ModelRegistry.CURSOR);

        if (cursor_hulls.count() == 0) return;

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
        float[] loc = { world_x, world_y, -1.0f, 10.0f };

        glBindVertexArray(vao);

        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());

        prepare_transforms_k
                .share_mem(vbo_ptr)
                .ptr_arg(PrepareTransforms_k.Args.indices, cursor_hulls.indices())
                .set_arg(PrepareTransforms_k.Args.offset, 0)
                .call(arg_long(1));

        glNamedBufferSubData(vbo, VECTOR_FLOAT_4D_SIZE, loc);
        glDrawArrays(GL_POINTS, 0, 2);

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
