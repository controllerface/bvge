package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.cl.*;
import com.controllerface.bvge.cl.buffers.RenderBufferType;
import com.controllerface.bvge.cl.kernels.*;
import com.controllerface.bvge.cl.programs.GPUProgram;
import com.controllerface.bvge.cl.programs.PrepareLiquids;
import com.controllerface.bvge.cl.programs.RootHullFilter;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.ComponentType;
import com.controllerface.bvge.ecs.components.PlayerInput;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.editor.Editor;
import com.controllerface.bvge.geometry.ModelRegistry;
import com.controllerface.bvge.gl.GLUtils;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.physics.UniformGrid;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;

import java.util.Objects;

import static com.controllerface.bvge.cl.CLData.*;
import static com.controllerface.bvge.cl.CLUtils.arg_long;
import static com.controllerface.bvge.util.Constants.Rendering.MAX_BATCH_SIZE;
import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_FLOAT_4D_SIZE;
import static org.lwjgl.opengl.ARBDirectStateAccess.glCreateVertexArrays;
import static org.lwjgl.opengl.GL11C.*;
import static org.lwjgl.opengl.GL15C.GL_POINTS;
import static org.lwjgl.opengl.GL15C.glDeleteBuffers;
import static org.lwjgl.opengl.GL30C.glBindVertexArray;
import static org.lwjgl.opengl.GL30C.glDeleteVertexArrays;
import static org.lwjgl.opengl.GL45C.glEnableVertexArrayAttrib;

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
    private Shader shader;

    private int vao;
    private int vbo_transform;
    private int vbo_color;
    private long ptr_vbo_transform;
    private long ptr_vbo_color;
    private long svm_atomic_counter;

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
        shader = Assets.load_shader("water_shader.glsl");
        vao = glCreateVertexArrays();
        vbo_transform = GLUtils.new_buffer_vec4(vao, TRANSFORM_ATTRIBUTE, CIRCLES_BUFFER_SIZE);
        vbo_color = GLUtils.new_buffer_vec4(vao, COLOR_ATTRIBUTE, COLOR_BUFFER_SIZE);
        glEnableVertexArrayAttrib(vao, TRANSFORM_ATTRIBUTE);
        glEnableVertexArrayAttrib(vao, COLOR_ATTRIBUTE);
    }

    private void init_CL()
    {
        p_prepare_liquids.init();
        p_root_hull_filter.init();
        svm_atomic_counter = GPGPU.cl_new_pinned_int();
        ptr_vbo_transform = GPGPU.share_memory(vbo_transform);
        ptr_vbo_color = GPGPU.share_memory(vbo_color);

        long k_ptr_prepare_liquids = p_prepare_liquids.kernel_ptr(Kernel.prepare_liquids);
        k_prepare_liquids = (new PrepareLiquids_k(GPGPU.ptr_render_queue, k_ptr_prepare_liquids))
            .ptr_arg(PrepareLiquids_k.Args.transforms_out, ptr_vbo_transform)
            .ptr_arg(PrepareLiquids_k.Args.colors_out, ptr_vbo_color)
            .buf_arg(PrepareLiquids_k.Args.hull_positions, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL))
            .buf_arg(PrepareLiquids_k.Args.hull_scales, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL_SCALE))
            .buf_arg(PrepareLiquids_k.Args.hull_rotations, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL_ROTATION))
            .buf_arg(PrepareLiquids_k.Args.hull_point_tables, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL_POINT_TABLE))
            .buf_arg(PrepareLiquids_k.Args.hull_uv_offsets, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL_UV_OFFSET))
            .buf_arg(PrepareLiquids_k.Args.point_hit_counts, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_POINT_HIT_COUNT));

        long k_ptr_root_hull_filter = p_root_hull_filter.kernel_ptr(Kernel.root_hull_filter);
        k_root_hull_filter = new RootHullFilter_k(GPGPU.ptr_render_queue, k_ptr_root_hull_filter)
            .buf_arg(RootHullFilter_k.Args.entity_root_hulls, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_ENTITY_ROOT_HULL))
            .buf_arg(RootHullFilter_k.Args.entity_model_indices, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_ENTITY_MODEL_ID));

        long k_ptr_root_hull_count = p_root_hull_filter.kernel_ptr(Kernel.root_hull_count);
        k_root_hull_count = new RootHullCount_k(GPGPU.ptr_render_queue, k_ptr_root_hull_count)
            .buf_arg(RootHullCount_k.Args.entity_model_indices, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_ENTITY_MODEL_ID));
    }

    @Override
    public void tick(float dt)
    {
        if (circle_hulls != null && circle_hulls.indices() != -1)
        {
            GPGPU.cl_release_buffer(circle_hulls.indices());
        }
        circle_hulls = GL_hull_filter(GPGPU.ptr_render_queue, ModelRegistry.CIRCLE_PARTICLE);


        if (Editor.ACTIVE)
        {
            Editor.queue_event("render_liquid_count", String.valueOf(circle_hulls.count()));
        }

        if (circle_hulls.count() == 0) return;

        glBindVertexArray(vao);
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
            int count_size = GPGPU.calculate_preferred_global_size(count);

            k_prepare_liquids
                .share_mem(ptr_vbo_transform)
                .share_mem(ptr_vbo_color)
                .ptr_arg(PrepareLiquids_k.Args.indices, circle_hulls.indices())
                .set_arg(PrepareLiquids_k.Args.offset, offset)
                .set_arg(PrepareLiquids_k.Args.max_hull, count)
                .call(arg_long(count_size), GPGPU.preferred_work_size);

            glDrawArrays(GL_POINTS, 0, count);
            offset += count;
        }

        shader.detach();
        glBindVertexArray(0);
    }

    private HullIndexData GL_hull_filter(long queue_ptr, int model_id)
    {
        GPGPU.cl_zero_buffer(queue_ptr, svm_atomic_counter, cl_int.size());

        int entity_count = GPGPU.core_memory.sector_container().next_entity();
        int entity_size  = GPGPU.calculate_preferred_global_size(entity_count);

        k_root_hull_count
            .ptr_arg(RootHullCount_k.Args.counter, svm_atomic_counter)
            .set_arg(RootHullCount_k.Args.model_id, model_id)
            .set_arg(RootHullCount_k.Args.max_entity, entity_count)
            .call(arg_long(entity_size), GPGPU.preferred_work_size);

        int final_count = GPGPU.cl_read_pinned_int(queue_ptr, svm_atomic_counter);

        if (final_count == 0)
        {
            return new HullIndexData(-1, final_count);
        }

        long final_buffer_size = (long) cl_int.size() * final_count;
        var hulls_out = GPGPU.cl_new_buffer(final_buffer_size);

        GPGPU.cl_zero_buffer(queue_ptr, svm_atomic_counter, cl_int.size());

        k_root_hull_filter
            .ptr_arg(RootHullFilter_k.Args.hulls_out, hulls_out)
            .ptr_arg(RootHullFilter_k.Args.counter, svm_atomic_counter)
            .set_arg(RootHullFilter_k.Args.model_id, model_id)
            .set_arg(RootHullFilter_k.Args.max_entity, entity_count)
            .call(arg_long(entity_size), GPGPU.preferred_work_size);

        return new HullIndexData(hulls_out, final_count);
    }

    @Override
    public void shutdown()
    {
        glDeleteVertexArrays(vao);
        glDeleteBuffers(vbo_transform);
        shader.destroy();
        p_prepare_liquids.destroy();
        GPGPU.cl_release_buffer(ptr_vbo_transform);
    }
}
