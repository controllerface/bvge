package com.controllerface.bvge.gpu.gl.renderers;

import com.controllerface.bvge.core.Window;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.GameSystem;
import com.controllerface.bvge.ecs.components.ComponentType;
import com.controllerface.bvge.editor.Editor;
import com.controllerface.bvge.game.Constants;
import com.controllerface.bvge.game.PlayerInput;
import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.CL_DataTypes;
import com.controllerface.bvge.gpu.cl.GPGPU;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.Kernel;
import com.controllerface.bvge.gpu.cl.kernels.rendering.PrepareTransforms_k;
import com.controllerface.bvge.gpu.cl.kernels.rendering.RootHullCount_k;
import com.controllerface.bvge.gpu.cl.kernels.rendering.RootHullFilter_k;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.gpu.cl.programs.PrepareTransforms;
import com.controllerface.bvge.gpu.cl.programs.RootHullFilter;
import com.controllerface.bvge.gpu.gl.GLUtils;
import com.controllerface.bvge.gpu.gl.shaders.GL_Shader;
import com.controllerface.bvge.gpu.gl.shaders.GL_ShaderType;
import com.controllerface.bvge.memory.types.RenderBufferType;
import com.controllerface.bvge.models.geometry.ModelRegistry;
import com.controllerface.bvge.physics.UniformGrid;
import com.controllerface.bvge.rendering.HullIndexData;

import java.util.Objects;

import static com.controllerface.bvge.game.Constants.Rendering.VECTOR_FLOAT_4D_SIZE;
import static com.controllerface.bvge.gpu.cl.CLUtils.arg_long;
import static org.lwjgl.opengl.ARBDirectStateAccess.glCreateVertexArrays;
import static org.lwjgl.opengl.GL11C.glDrawArrays;
import static org.lwjgl.opengl.GL15C.GL_POINTS;
import static org.lwjgl.opengl.GL15C.glDeleteBuffers;
import static org.lwjgl.opengl.GL30C.glBindVertexArray;
import static org.lwjgl.opengl.GL30C.glDeleteVertexArrays;
import static org.lwjgl.opengl.GL45C.glEnableVertexArrayAttrib;
import static org.lwjgl.opengl.GL45C.glNamedBufferSubData;

public class MouseRenderer extends GameSystem
{
    private static final int TRANSFORM_ATTRIBUTE = 0;

    private final GPUProgram prepare_transforms = new PrepareTransforms();
    private final GPUProgram root_hull_filter = new RootHullFilter();
    private GPUKernel k_prepare_transforms;
    private GPUKernel k_root_hull_filter;
    private GPUKernel k_root_hull_count;
    private GL_Shader shader;

    private int vao;
    private int vbo_transforms;
    private long ptr_vbo_transforms;
    private long svm_atomic_counter;

    private HullIndexData cursor_hulls;

    public MouseRenderer(ECS ecs)
    {
        super(ecs);
        init_GL();
        init_CL();
    }

    public void init_GL()
    {
        shader = GPU.GL.new_shader("mouse_shader.glsl", GL_ShaderType.THREE_STAGE);
        vao = glCreateVertexArrays();
        vbo_transforms = GLUtils.new_buffer_vec4(vao, TRANSFORM_ATTRIBUTE, VECTOR_FLOAT_4D_SIZE * 2);
        glEnableVertexArrayAttrib(vao, TRANSFORM_ATTRIBUTE);
    }


    private void init_CL()
    {
        prepare_transforms.init();
        root_hull_filter.init();
        ptr_vbo_transforms = GPGPU.share_memory(vbo_transforms);
        svm_atomic_counter = GPGPU.cl_new_pinned_int();

        long ptr = prepare_transforms.kernel_ptr(Kernel.prepare_transforms);
        k_prepare_transforms = (new PrepareTransforms_k(GPGPU.ptr_render_queue, ptr))
            .ptr_arg(PrepareTransforms_k.Args.transforms_out, ptr_vbo_transforms)
            .set_arg(PrepareTransforms_k.Args.max_hull, 1)
            .set_arg(PrepareTransforms_k.Args.offset, 0)
            .buf_arg(PrepareTransforms_k.Args.hull_positions, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL))
            .buf_arg(PrepareTransforms_k.Args.hull_scales, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL_SCALE))
            .buf_arg(PrepareTransforms_k.Args.hull_rotations, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL_ROTATION));

        long root_hull_filter_ptr = root_hull_filter.kernel_ptr(Kernel.root_hull_filter);
        k_root_hull_filter = new RootHullFilter_k(GPGPU.ptr_render_queue, root_hull_filter_ptr)
            .buf_arg(RootHullFilter_k.Args.entity_root_hulls, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_ENTITY_ROOT_HULL))
            .buf_arg(RootHullFilter_k.Args.entity_model_indices, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_ENTITY_MODEL_ID));

        long root_hull_count_ptr =  root_hull_filter.kernel_ptr(Kernel.root_hull_count);
        k_root_hull_count = new RootHullCount_k(GPGPU.ptr_render_queue, root_hull_count_ptr)
            .buf_arg(RootHullCount_k.Args.entity_model_indices, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_ENTITY_MODEL_ID));
    }

    public HullIndexData hull_filter(long queue_ptr, int model_id)
    {
        GPGPU.cl_zero_buffer(queue_ptr, svm_atomic_counter, CL_DataTypes.cl_int.size());

        int entity_count = GPGPU.core_memory.sector_container().next_entity();
        int entity_size  = GPGPU.calculate_preferred_global_size(entity_count);

        k_root_hull_count
            .ptr_arg(RootHullCount_k.Args.counter, svm_atomic_counter)
            .set_arg(RootHullCount_k.Args.model_id, model_id)
            .set_arg(RootHullCount_k.Args.max_entity, entity_count)
            .call(arg_long(entity_size), GPGPU.preferred_work_size);

        int final_count =  GPGPU.cl_read_pinned_int(queue_ptr, svm_atomic_counter);

        if (final_count == 0)
        {
            return new HullIndexData(-1, final_count);
        }

        long final_buffer_size = (long) CL_DataTypes.cl_int.size() * final_count;
        var hulls_out =  GPGPU.cl_new_buffer(final_buffer_size);

        GPGPU.cl_zero_buffer(queue_ptr, svm_atomic_counter, CL_DataTypes.cl_int.size());

        k_root_hull_filter
            .ptr_arg(RootHullFilter_k.Args.hulls_out, hulls_out)
            .ptr_arg(RootHullFilter_k.Args.counter, svm_atomic_counter)
            .set_arg(RootHullFilter_k.Args.model_id, model_id)
            .set_arg(RootHullFilter_k.Args.max_entity, entity_count)
            .call(arg_long(entity_size), GPGPU.preferred_work_size);

        return new HullIndexData(hulls_out, final_count);
    }

    @Override
    public void tick(float dt)
    {
        if (cursor_hulls != null && cursor_hulls.indices() != -1)
        {
            GPGPU.cl_release_buffer(cursor_hulls.indices());
        }
        cursor_hulls = hull_filter(GPGPU.ptr_render_queue, ModelRegistry.CURSOR);

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

        glBindVertexArray(vao);

        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());

        k_prepare_transforms
            .share_mem(ptr_vbo_transforms)
            .ptr_arg(PrepareTransforms_k.Args.indices, cursor_hulls.indices())
            .call_task();

        glNamedBufferSubData(vbo_transforms, VECTOR_FLOAT_4D_SIZE, mouse_loc);
        glDrawArrays(GL_POINTS, 0, 2);

        shader.detach();
        glBindVertexArray(0);
    }

    @Override
    public void shutdown()
    {
        glDeleteVertexArrays(vao);
        glDeleteBuffers(vbo_transforms);
        shader.release();
        GPGPU.cl_release_buffer(ptr_vbo_transforms);
        prepare_transforms.release();
        root_hull_filter.release();
        GPGPU.cl_release_buffer(svm_atomic_counter);
    }
}
