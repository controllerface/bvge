package com.controllerface.bvge.rendering.renderers;

import com.controllerface.bvge.core.Window;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.ComponentType;
import com.controllerface.bvge.editor.Editor;
import com.controllerface.bvge.game.Constants;
import com.controllerface.bvge.game.PlayerInput;
import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.rendering.PrepareTransforms_k;
import com.controllerface.bvge.gpu.cl.kernels.rendering.RootHullCount_k;
import com.controllerface.bvge.gpu.cl.kernels.rendering.RootHullFilter_k;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.gpu.cl.programs.rendering.PrepareTransforms;
import com.controllerface.bvge.gpu.cl.programs.rendering.RootHullFilter;
import com.controllerface.bvge.gpu.gl.buffers.GL_VertexArray;
import com.controllerface.bvge.gpu.gl.buffers.GL_VertexBuffer;
import com.controllerface.bvge.gpu.gl.shaders.GL_Shader;
import com.controllerface.bvge.gpu.gl.shaders.GL_ShaderType;
import com.controllerface.bvge.models.geometry.ModelRegistry;
import com.controllerface.bvge.physics.UniformGrid;
import com.controllerface.bvge.rendering.HullIndexData;
import com.controllerface.bvge.rendering.Renderer;

import java.util.Objects;

import static com.controllerface.bvge.game.Constants.Rendering.VECTOR_FLOAT_4D_SIZE;
import static com.controllerface.bvge.gpu.GPU.CL.arg_long;
import static org.lwjgl.opengl.GL11C.glDrawArrays;
import static org.lwjgl.opengl.GL15C.GL_POINTS;

public class MouseRenderer implements Renderer
{
    private static final int TRANSFORM_ATTRIBUTE = 0;

    private final GPUProgram prepare_transforms = new PrepareTransforms();
    private final GPUProgram root_hull_filter = new RootHullFilter();
    private GPUKernel k_prepare_transforms;
    private GPUKernel k_root_hull_filter;
    private GPUKernel k_root_hull_count;
    private GL_Shader shader;

    private GL_VertexArray vao;
    private GL_VertexBuffer vbo_transforms;
    private CL_Buffer transforms_buf;
    private CL_Buffer atomic_counter;

    private HullIndexData cursor_hulls;
    private final ECS ecs;

    public MouseRenderer(ECS ecs)
    {
        this.ecs = ecs;
        init_GL();
        init_CL();
    }

    public void init_GL()
    {
        shader = GPU.GL.new_shader("mouse_shader.glsl", GL_ShaderType.THREE_STAGE);
        vao = GPU.GL.new_vao();
        vbo_transforms = GPU.GL.new_buffer_vec4(vao, TRANSFORM_ATTRIBUTE, VECTOR_FLOAT_4D_SIZE * 2);
        vao.enable_attribute(TRANSFORM_ATTRIBUTE);
    }

    private void init_CL()
    {
        prepare_transforms.init();
        root_hull_filter.init();
        transforms_buf = GPU.CL.gl_share_memory(GPU.compute.context, vbo_transforms);
        atomic_counter = GPU.CL.new_pinned_int(GPU.compute.context);

        k_root_hull_filter = new RootHullFilter_k(GPU.compute.render_queue, root_hull_filter).init();
        k_root_hull_count  = new RootHullCount_k(GPU.compute.render_queue, root_hull_filter).init();

        k_prepare_transforms = new PrepareTransforms_k(GPU.compute.render_queue, prepare_transforms)
            .init(transforms_buf)
            .set_arg(PrepareTransforms_k.Args.max_hull, 1)
            .set_arg(PrepareTransforms_k.Args.offset, 0);
    }

    public HullIndexData hull_filter(CL_CommandQueue cmd_queue, int model_id)
    {
        GPU.CL.zero_buffer(cmd_queue, atomic_counter, CL_DataTypes.cl_int.size());

        int entity_count = GPU.memory.sector_container().next_entity();
        int entity_size  = GPU.compute.calculate_preferred_global_size(entity_count);

        k_root_hull_count
            .buf_arg(RootHullCount_k.Args.counter, atomic_counter)
            .set_arg(RootHullCount_k.Args.model_id, model_id)
            .set_arg(RootHullCount_k.Args.max_entity, entity_count)
            .call(arg_long(entity_size), GPU.compute.preferred_work_size);

        int final_count =  GPU.CL.read_pinned_int(cmd_queue, atomic_counter);

        if (final_count == 0)
        {
            return new HullIndexData(null, final_count);
        }

        long final_buffer_size = (long) CL_DataTypes.cl_int.size() * final_count;
        var hulls_out = GPU.CL.new_buffer(GPU.compute.context, final_buffer_size);

        GPU.CL.zero_buffer(cmd_queue, atomic_counter, CL_DataTypes.cl_int.size());

        k_root_hull_filter
            .buf_arg(RootHullFilter_k.Args.hulls_out, hulls_out)
            .buf_arg(RootHullFilter_k.Args.counter, atomic_counter)
            .set_arg(RootHullFilter_k.Args.model_id, model_id)
            .set_arg(RootHullFilter_k.Args.max_entity, entity_count)
            .call(arg_long(entity_size), GPU.compute.preferred_work_size);

        return new HullIndexData(hulls_out, final_count);
    }

    @Override
    public void render()
    {
        if (cursor_hulls != null && cursor_hulls.indices() != null)
        {
            cursor_hulls.indices().release();
        }
        cursor_hulls = hull_filter(GPU.compute.render_queue, ModelRegistry.CURSOR);

        if (cursor_hulls.count() == 0) return;

        PlayerInput player_input = ComponentType.InputState.forEntity(ecs, Constants.PLAYER_ID);
        assert player_input != null : "Component was null";
        Objects.requireNonNull(player_input);

        var camera = Window.get().camera();
        float world_x = player_input.get_screen_target().x * camera.get_zoom() + camera.position().x;
        float world_y = (Window.get().height() - player_input.get_screen_target().y) * camera.get_zoom() + camera.position().y;
        player_input.get_world_target().set(world_x, world_y);
        float[] mouse_loc = { world_x, world_y, -1.0f, 15.0f };

        if (Editor.ACTIVE)
        {
            var sector = UniformGrid.get_sector_for_point(world_x, world_y);
            Editor.queue_event("mouse_sector", sector[0] + ":" + sector[1]);
        }

        vao.bind();

        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());

        k_prepare_transforms
            .share_mem(transforms_buf)
            .buf_arg(PrepareTransforms_k.Args.indices, cursor_hulls.indices())
            .call_task();

        vbo_transforms.load_float_sub_data(mouse_loc, VECTOR_FLOAT_4D_SIZE);
        glDrawArrays(GL_POINTS, 0, 2);

        shader.detach();
        vao.unbind();
    }

    @Override
    public void destroy()
    {
        vao.release();
        vbo_transforms.release();
        shader.release();
        transforms_buf.release();
        prepare_transforms.release();
        root_hull_filter.release();
        atomic_counter.release();
    }
}