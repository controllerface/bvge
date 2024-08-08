package com.controllerface.bvge.rendering.renderers;

import com.controllerface.bvge.core.Window;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.GameSystem;
import com.controllerface.bvge.ecs.components.ComponentType;
import com.controllerface.bvge.editor.Editor;
import com.controllerface.bvge.game.Constants;
import com.controllerface.bvge.game.PlayerInput;
import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.GPGPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.kernels.rendering.PrepareLiquids_k;
import com.controllerface.bvge.gpu.cl.kernels.rendering.RootHullCount_k;
import com.controllerface.bvge.gpu.cl.kernels.rendering.RootHullFilter_k;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.gpu.cl.programs.rendering.PrepareLiquids;
import com.controllerface.bvge.gpu.cl.programs.rendering.RootHullFilter;
import com.controllerface.bvge.gpu.gl.buffers.GL_VertexArray;
import com.controllerface.bvge.gpu.gl.buffers.GL_VertexBuffer;
import com.controllerface.bvge.gpu.gl.shaders.GL_Shader;
import com.controllerface.bvge.gpu.gl.shaders.GL_ShaderType;
import com.controllerface.bvge.memory.types.RenderBufferType;
import com.controllerface.bvge.models.geometry.ModelRegistry;
import com.controllerface.bvge.physics.UniformGrid;
import com.controllerface.bvge.rendering.HullIndexData;

import java.util.Objects;

import static com.controllerface.bvge.game.Constants.Rendering.MAX_BATCH_SIZE;
import static com.controllerface.bvge.game.Constants.Rendering.VECTOR_FLOAT_4D_SIZE;
import static com.controllerface.bvge.gpu.GPU.CL.arg_long;
import static com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes.cl_int;
import static org.lwjgl.opengl.GL11C.glDrawArrays;
import static org.lwjgl.opengl.GL15C.GL_POINTS;

public class LiquidRenderer extends GameSystem
{
    private final UniformGrid uniformGrid;

    public static final int CIRCLES_BUFFER_SIZE = MAX_BATCH_SIZE * VECTOR_FLOAT_4D_SIZE;
    private static final int COLOR_BUFFER_SIZE  = MAX_BATCH_SIZE * VECTOR_FLOAT_4D_SIZE;
    private static final int TRANSFORM_ATTRIBUTE = 0;
    private static final int COLOR_ATTRIBUTE     = 1;

    private final GPUProgram p_prepare_liquids = new PrepareLiquids();
    private final GPUProgram p_root_hull_filter = new RootHullFilter();
    private GPUKernel k_prepare_liquids;
    private GPUKernel k_root_hull_count;
    private GPUKernel k_root_hull_filter;
    private GL_Shader shader;

    private GL_VertexArray vao;
    private GL_VertexBuffer vbo_transform;
    private GL_VertexBuffer vbo_color;
    private CL_Buffer ptr_vbo_transform;
    private CL_Buffer ptr_vbo_color;
    private CL_Buffer atomic_counter;

    private HullIndexData circle_hulls;

    public LiquidRenderer(ECS ecs, UniformGrid uniformGrid)
    {
        super(ecs);
        this.uniformGrid = uniformGrid;

        init_GL();
        init_CL();
    }

    public void init_GL()
    {
        shader = GPU.GL.new_shader("water_shader.glsl", GL_ShaderType.THREE_STAGE);
        vao = GPU.GL.new_vao();
        vbo_transform = GPU.GL.new_buffer_vec4(vao, TRANSFORM_ATTRIBUTE, CIRCLES_BUFFER_SIZE);
        vbo_color = GPU.GL.new_buffer_vec4(vao, COLOR_ATTRIBUTE, COLOR_BUFFER_SIZE);
        vao.enable_attribute(TRANSFORM_ATTRIBUTE);
        vao.enable_attribute(COLOR_ATTRIBUTE);
    }

    private void init_CL()
    {
        p_prepare_liquids.init();
        p_root_hull_filter.init();
        atomic_counter = GPU.CL.new_pinned_int(GPGPU.compute.context);
        ptr_vbo_transform = GPU.CL.gl_share_memory(GPGPU.compute.context, vbo_transform);
        ptr_vbo_color = GPU.CL.gl_share_memory(GPGPU.compute.context, vbo_color);

        long k_ptr_prepare_liquids = p_prepare_liquids.kernel_ptr(KernelType.prepare_liquids);
        k_prepare_liquids = (new PrepareLiquids_k(GPGPU.compute.render_queue, k_ptr_prepare_liquids))
            .buf_arg(PrepareLiquids_k.Args.transforms_out, ptr_vbo_transform)
            .buf_arg(PrepareLiquids_k.Args.colors_out, ptr_vbo_color)
            .buf_arg(PrepareLiquids_k.Args.hull_positions, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL))
            .buf_arg(PrepareLiquids_k.Args.hull_scales, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL_SCALE))
            .buf_arg(PrepareLiquids_k.Args.hull_rotations, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL_ROTATION))
            .buf_arg(PrepareLiquids_k.Args.hull_point_tables, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL_POINT_TABLE))
            .buf_arg(PrepareLiquids_k.Args.hull_uv_offsets, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL_UV_OFFSET))
            .buf_arg(PrepareLiquids_k.Args.point_hit_counts, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_POINT_HIT_COUNT));

        long k_ptr_root_hull_filter = p_root_hull_filter.kernel_ptr(KernelType.root_hull_filter);
        k_root_hull_filter = new RootHullFilter_k(GPGPU.compute.render_queue, k_ptr_root_hull_filter)
            .buf_arg(RootHullFilter_k.Args.entity_root_hulls, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_ENTITY_ROOT_HULL))
            .buf_arg(RootHullFilter_k.Args.entity_model_indices, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_ENTITY_MODEL_ID));

        long k_ptr_root_hull_count = p_root_hull_filter.kernel_ptr(KernelType.root_hull_count);
        k_root_hull_count = new RootHullCount_k(GPGPU.compute.render_queue, k_ptr_root_hull_count)
            .buf_arg(RootHullCount_k.Args.entity_model_indices, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_ENTITY_MODEL_ID));
    }

    @Override
    public void tick(float dt)
    {
        if (circle_hulls != null && circle_hulls.indices() != null)
        {
            circle_hulls.indices().release();
        }
        circle_hulls = GL_hull_filter(GPGPU.compute.render_queue, ModelRegistry.CIRCLE_PARTICLE);


        if (Editor.ACTIVE)
        {
            Editor.queue_event("render_liquid_count", String.valueOf(circle_hulls.count()));
        }

        if (circle_hulls.count() == 0) return;

        vao.bind();
        shader.use();

        PlayerInput player_input = ComponentType.InputState.forEntity(ecs, Constants.PLAYER_ID);
        assert player_input != null : "Component was null";
        Objects.requireNonNull(player_input);

        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());
        shader.uploadvec2f("uMouse", player_input.get_world_target());
        shader.uploadvec2f("uCamera", uniformGrid.getWorld_position());

        int offset = 0;
        for (int remaining = circle_hulls.count(); remaining > 0; remaining -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, remaining);
            int count_size = GPGPU.compute.calculate_preferred_global_size(count);

            k_prepare_liquids
                .share_mem(ptr_vbo_transform)
                .share_mem(ptr_vbo_color)
                .buf_arg(PrepareLiquids_k.Args.indices, circle_hulls.indices())
                .set_arg(PrepareLiquids_k.Args.offset, offset)
                .set_arg(PrepareLiquids_k.Args.max_hull, count)
                .call(arg_long(count_size), GPGPU.compute.preferred_work_size);

            glDrawArrays(GL_POINTS, 0, count);
            offset += count;
        }

        shader.detach();
        vao.unbind();
    }

    private HullIndexData GL_hull_filter(CL_CommandQueue cmd_queue, int model_id)
    {
        GPU.CL.zero_buffer(cmd_queue, atomic_counter, cl_int.size());

        int entity_count = GPGPU.core_memory.sector_container().next_entity();
        int entity_size  = GPGPU.compute.calculate_preferred_global_size(entity_count);

        k_root_hull_count
            .buf_arg(RootHullCount_k.Args.counter, atomic_counter)
            .set_arg(RootHullCount_k.Args.model_id, model_id)
            .set_arg(RootHullCount_k.Args.max_entity, entity_count)
            .call(arg_long(entity_size), GPGPU.compute.preferred_work_size);

        int final_count = GPGPU.cl_read_pinned_int(cmd_queue.ptr(), atomic_counter.ptr());

        if (final_count == 0)
        {
            return new HullIndexData(null, final_count);
        }

        long final_buffer_size = (long) cl_int.size() * final_count;
        var hulls_out = GPU.CL.new_buffer(GPGPU.compute.context, final_buffer_size);

        GPU.CL.zero_buffer(cmd_queue, atomic_counter, cl_int.size());

        k_root_hull_filter
            .buf_arg(RootHullFilter_k.Args.hulls_out, hulls_out)
            .buf_arg(RootHullFilter_k.Args.counter, atomic_counter)
            .set_arg(RootHullFilter_k.Args.model_id, model_id)
            .set_arg(RootHullFilter_k.Args.max_entity, entity_count)
            .call(arg_long(entity_size), GPGPU.compute.preferred_work_size);

        return new HullIndexData(hulls_out, final_count);
    }

    @Override
    public void shutdown()
    {
        vao.release();
        vbo_transform.release();
        vbo_color.release();
        shader.release();
        p_prepare_liquids.release();
        ptr_vbo_transform.release();
        ptr_vbo_color.release();
        atomic_counter.release();
    }
}
