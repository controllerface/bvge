package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.cl.CLSize;
import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUProgram;
import com.controllerface.bvge.cl.programs.MeshQuery;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.geometry.Models;
import com.controllerface.bvge.gl.AbstractShader;
import com.controllerface.bvge.gl.GLUtils;
import com.controllerface.bvge.gl.Texture;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;

import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_FLOAT_2D_SIZE;
import static org.lwjgl.opengl.GL15C.*;
import static org.lwjgl.opengl.GL30C.glBindVertexArray;
import static org.lwjgl.opengl.GL40C.GL_DRAW_INDIRECT_BUFFER;
import static org.lwjgl.opengl.GL43C.glMultiDrawElementsIndirect;
import static org.lwjgl.opengl.GL45C.*;

public class HumanoidRenderer extends GameSystem
{
    private static final int ELEMENT_BUFFER_SIZE = Constants.Rendering.MAX_BATCH_SIZE * Integer.BYTES;
    private static final int VERTEX_BUFFER_SIZE = Constants.Rendering.MAX_BATCH_SIZE * VECTOR_FLOAT_2D_SIZE;
    private static final int COMMAND_BUFFER_SIZE = Constants.Rendering.MAX_BATCH_SIZE * Integer.BYTES * 5;

    private static final int POSITION_ATTRIBUTE = 0;
    private static final int UV_COORD_ATTRIBUTE = 1;

    private Texture texture;
    private final AbstractShader shader;
    private final int[] texture_slots = {0};

    private int vao_id;
    private int vertex_b;
    private int texture_uv_b;
    private int element_b;
    private int command_b;

    private int mesh_count;
    private long mesh_size;

    // todo: these memory buffers are set as arguments into the relevant kernels on each call, but
    //  they could be set once and kept as-is, if the kernel being used was local to this class. At
    //  the moment, this isn't possible due to how the memory objects, programs, and kernels are
    //  are tied directly to the GPU class, but in the future it will probably be a good idea to
    //  consider allowing individual classes to have private kernels. It will increase efficiency,
    //  especially if multiple classes end up needing the same kernel calls a lot.
    private long query_ptr;
    private long counters;
    private long total;
    private long offsets;
    private long mesh_transfer;

    private GPUProgram mesh_query;

    public HumanoidRenderer(ECS ecs)
    {
        super(ecs);
        this.shader = Assets.load_shader("poly_model.glsl");
        init();
        init_kernels();
    }

    private void init()
    {
        var model = Models.get_model_by_index(Models.TEST_MODEL_INDEX);
        this.texture = model.textures().get(0);

        mesh_count = model.meshes().length;
        mesh_size = (long)mesh_count * CLSize.cl_int;
        int[] raw_query = new int[mesh_count];
        for (int i = 0; i < model.meshes().length; i++)
        {
            var m = model.meshes()[i];
            raw_query[i] = m.mesh_id();
        }

        total = GPU.cl_new_pinned_int();
        query_ptr = GPU.new_mutable_buffer(raw_query);
        counters = GPU.new_empty_buffer(mesh_size);
        offsets = GPU.new_empty_buffer(mesh_size);
        mesh_transfer = GPU.new_empty_buffer(ELEMENT_BUFFER_SIZE * 2);

        vao_id = glCreateVertexArrays();

        element_b = glCreateBuffers();
        glNamedBufferData(element_b, ELEMENT_BUFFER_SIZE, GL_STATIC_DRAW);
        glVertexArrayElementBuffer(vao_id, element_b);

        vertex_b = GLUtils.new_buffer_vec2(vao_id, POSITION_ATTRIBUTE, VERTEX_BUFFER_SIZE);
        texture_uv_b = GLUtils.new_buffer_vec2(vao_id, UV_COORD_ATTRIBUTE, VERTEX_BUFFER_SIZE);

        glBindVertexArray(vao_id);
        command_b = glCreateBuffers();
        glNamedBufferData(command_b, COMMAND_BUFFER_SIZE, GL_STATIC_DRAW);
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, command_b);
        glBindVertexArray(0);

        GPU.share_memory(element_b);
        GPU.share_memory(vertex_b);
        GPU.share_memory(command_b);
        GPU.share_memory(texture_uv_b);
    }

    private void init_kernels()
    {
        mesh_query = new MeshQuery();
        mesh_query.init();
    }

    @Override
    public void tick(float dt)
    {
        GPU.clear_buffer(counters, mesh_size);
        GPU.clear_buffer(offsets, mesh_size);
        GPU.clear_buffer(total, CLSize.cl_int);
        GPU.clear_buffer(mesh_transfer, ELEMENT_BUFFER_SIZE * 2);

        GPU.GL_count_mesh_instances(query_ptr, counters, total, mesh_count);
        GPU.GL_scan_mesh_offsets(counters, offsets, mesh_count);

        int total_instances = GPU.cl_read_pinned_int(total);
        if (total_instances == 0) // highly unlikely, but just in case
        {
            return;
        }

        long data_size = (long)total_instances * CLSize.cl_int4;
        var details_b = GPU.new_empty_buffer(data_size);

        GPU.GL_write_mesh_details(query_ptr, counters, offsets, details_b, mesh_count);
        GPU.GL_count_mesh_batches(details_b, total, total_instances, Constants.Rendering.MAX_BATCH_SIZE);

        int total_batches = GPU.cl_read_pinned_int(total);
        long batch_index_size = (long) total_batches * CLSize.cl_int;

        var batch_offset_b = GPU.new_empty_buffer(batch_index_size);

        GPU.GL_calculate_batch_offsets(batch_offset_b, details_b, total_instances);

        int[] raw_offsets = new int[total_batches];
        GPU.cl_read_buffer(batch_offset_b, raw_offsets);

        glBindVertexArray(vao_id);

        shader.use();
        texture.bind(0);

        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());
        shader.uploadIntArray("uTextures", texture_slots);

        glEnableVertexArrayAttrib(vao_id, POSITION_ATTRIBUTE);
        glEnableVertexArrayAttrib(vao_id, UV_COORD_ATTRIBUTE);

        for (int current_batch = 0; current_batch < raw_offsets.length; current_batch++)
        {
            int next_batch = current_batch + 1;
            int start = raw_offsets[current_batch];
            int count = next_batch == raw_offsets.length
                ? total_instances - start
                : raw_offsets[next_batch] - start;

            GPU.GL_transfer_detail_data(details_b, mesh_transfer, count, start);
            GPU.GL_transfer_render_data(element_b, vertex_b, command_b, texture_uv_b, details_b, mesh_transfer, count, start);
            glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, 0, count, 0);
        }

        glDisableVertexArrayAttrib(vao_id, POSITION_ATTRIBUTE);
        glDisableVertexArrayAttrib(vao_id, UV_COORD_ATTRIBUTE);

        glBindVertexArray(0);

        shader.detach();

        GPU.release_buffer(details_b);
        GPU.release_buffer(batch_offset_b);
    }

    @Override
    public void shutdown()
    {
        GPU.release_buffer(total);
        GPU.release_buffer(query_ptr);
        GPU.release_buffer(counters);
        GPU.release_buffer(offsets);
        GPU.release_buffer(mesh_transfer);
    }
}
