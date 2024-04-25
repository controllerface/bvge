package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.cl.*;
import com.controllerface.bvge.cl.kernels.PrepareTransforms_k;
import com.controllerface.bvge.cl.programs.PrepareTransforms;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.geometry.ModelRegistry;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.gl.GLUtils;
import com.controllerface.bvge.gl.Texture;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;

import static com.controllerface.bvge.cl.CLUtils.arg_long;
import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_4D_LENGTH;
import static org.lwjgl.opengl.GL30.glBindVertexArray;
import static org.lwjgl.opengl.GL31.glDrawElementsInstanced;
import static org.lwjgl.opengl.GL45C.*;

public class CrateRenderer extends GameSystem
{
    private static final int TRANSFORM_VERTEX_COUNT = Constants.Rendering.MAX_BATCH_SIZE * VECTOR_4D_LENGTH;
    private static final int TRANSFORM_BUFFER_SIZE = TRANSFORM_VERTEX_COUNT * Float.BYTES;
    private static final int POSITION_ATTRIBUTE = 0;
    private static final int UV_COORD_ATTRIBUTE = 1;
    private static final int TRANSFORM_ATTRIBUTE = 2;

    private final int[] texture_slots = {0};

    private final GPUProgram prepare_transforms = new PrepareTransforms();

    private int vao;
    private int ebo;
    private int transform_vbo;
    private int position_vbo;
    private int uv_vbo;

    private long vbo_ptr;

    private Texture texture;
    private HullIndexData crate_hulls;
    private Shader shader;
    private GPUKernel prepare_transforms_k;

    public CrateRenderer(ECS ecs)
    {
        super(ecs);
        init_GL();
        init_CL();
    }

    private void init_GL()
    {
        var model = ModelRegistry.get_model_by_index(ModelRegistry.TEST_SQUARE_INDEX);
        var base_mesh = model.meshes()[0];
        var raw_mesh = base_mesh.raw_copy();

        texture = model.textures().getFirst();
        shader = Assets.load_shader("box_model.glsl");
        vao = glCreateVertexArrays();
        ebo = GLUtils.static_element_buffer(vao, raw_mesh.faces());
        position_vbo = GLUtils.fill_buffer_vec2(vao, POSITION_ATTRIBUTE, raw_mesh.vertices());
        uv_vbo = GLUtils.fill_buffer_vec2(vao, UV_COORD_ATTRIBUTE, raw_mesh.uvs());
        transform_vbo = GLUtils.new_buffer_vec4(vao, TRANSFORM_ATTRIBUTE, TRANSFORM_BUFFER_SIZE);

        glVertexArrayBindingDivisor(vao, TRANSFORM_ATTRIBUTE, 1);
        glEnableVertexArrayAttrib(vao, POSITION_ATTRIBUTE);
        glEnableVertexArrayAttrib(vao, UV_COORD_ATTRIBUTE);
        glEnableVertexArrayAttrib(vao, TRANSFORM_ATTRIBUTE);
    }

    private void init_CL()
    {
        vbo_ptr = GPGPU.share_memory(transform_vbo);

        prepare_transforms.init();

        long ptr = prepare_transforms.kernel_ptr(Kernel.prepare_transforms);
        prepare_transforms_k = (new PrepareTransforms_k(GPGPU.command_queue_ptr, ptr))
            .ptr_arg(PrepareTransforms_k.Args.transforms_out, vbo_ptr)
            .buf_arg(PrepareTransforms_k.Args.hull_positions, GPGPU.core_memory.buffer(BufferType.HULL))
            .buf_arg(PrepareTransforms_k.Args.hull_scales, GPGPU.core_memory.buffer(BufferType.HULL_SCALE))
            .buf_arg(PrepareTransforms_k.Args.hull_rotations, GPGPU.core_memory.buffer(BufferType.HULL_ROTATION));
    }

    @Override
    public void tick(float dt)
    {
        if (crate_hulls != null && crate_hulls.indices() != -1)
        {
            GPGPU.cl_release_buffer(crate_hulls.indices());
        }
        crate_hulls = GPGPU.GL_hull_filter(ModelRegistry.TEST_SQUARE_INDEX);

        if (crate_hulls.count() == 0)
        {
            return;
        }

        glBindVertexArray(vao);

        shader.use();
        texture.bind(0);

        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());
        shader.uploadIntArray("uTextures", texture_slots);

        int offset = 0;
        for (int remaining = crate_hulls.count(); remaining > 0; remaining -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, remaining);

            prepare_transforms_k
                .share_mem(vbo_ptr)
                .ptr_arg(PrepareTransforms_k.Args.indices, crate_hulls.indices())
                .set_arg(PrepareTransforms_k.Args.offset, offset)
                .call(arg_long(count));

            glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, count);
            offset += count;
        }

        glBindVertexArray(0);

        shader.detach();
    }

    @Override
    public void shutdown()
    {
        glDeleteVertexArrays(vao);
        glDeleteBuffers(ebo);
        glDeleteBuffers(transform_vbo);
        glDeleteBuffers(position_vbo);
        glDeleteBuffers(uv_vbo);
        shader.destroy();
        texture.destroy();
        prepare_transforms.destroy();
        GPGPU.cl_release_buffer(vbo_ptr);
    }
}