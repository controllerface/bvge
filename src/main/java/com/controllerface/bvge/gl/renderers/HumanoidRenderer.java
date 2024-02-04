package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.geometry.Models;
import com.controllerface.bvge.gl.AbstractShader;
import com.controllerface.bvge.gl.Texture;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_mem;

import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_2D_LENGTH;
import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_FLOAT_2D_SIZE;
import static org.lwjgl.opengl.GL11C.GL_FLOAT;
import static org.lwjgl.opengl.GL13.GL_TEXTURE0;
import static org.lwjgl.opengl.GL15C.*;
import static org.lwjgl.opengl.GL20.glDisableVertexAttribArray;
import static org.lwjgl.opengl.GL20.glEnableVertexAttribArray;
import static org.lwjgl.opengl.GL20C.glVertexAttribPointer;
import static org.lwjgl.opengl.GL30C.glBindVertexArray;
import static org.lwjgl.opengl.GL30C.glGenVertexArrays;
import static org.lwjgl.opengl.GL40C.GL_DRAW_INDIRECT_BUFFER;
import static org.lwjgl.opengl.GL43C.glMultiDrawElementsIndirect;

public class HumanoidRenderer extends GameSystem
{
    private static final int ELEMENT_BUFFER_SIZE = Constants.Rendering.MAX_BATCH_SIZE * Integer.BYTES;
    private static final int VERTEX_BUFFER_SIZE = Constants.Rendering.MAX_BATCH_SIZE * VECTOR_FLOAT_2D_SIZE;
    private static final int COMMAND_BUFFER_SIZE = Constants.Rendering.MAX_BATCH_SIZE * Integer.BYTES * 5;

    // todo: determine the sizes required for a single render batch and calculate them
    private Texture texture;
    private final AbstractShader shader;
    private final int[] texture_slots = {0};

    private int vao;
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
    private cl_mem query;
    private cl_mem counters;
    private cl_mem total;
    private cl_mem offsets;
    private cl_mem mesh_transfer;

    public HumanoidRenderer(ECS ecs)
    {
        super(ecs);
        this.shader = Assets.load_shader("poly_model.glsl");
        init();
    }

    private void init()
    {
        var model = Models.get_model_by_index(Models.TEST_MODEL_INDEX);
        this.texture = model.textures().get(0);

        mesh_count = model.meshes().length;
        mesh_size = (long)mesh_count * Sizeof.cl_int;
        int[] raw_query = new int[mesh_count];
        for (int i = 0; i < model.meshes().length; i++)
        {
            var m = model.meshes()[i];
            raw_query[i] = m.mesh_id();
        }

        total = GPU.cl_new_pinned_int();
        query = GPU.new_mutable_buffer(mesh_size, Pointer.to(raw_query));
        counters = GPU.new_empty_buffer(mesh_size);
        offsets = GPU.new_empty_buffer(mesh_size);
        mesh_transfer = GPU.new_empty_buffer(ELEMENT_BUFFER_SIZE * 2);

        vao = glGenVertexArrays();
        glBindVertexArray(vao);

        element_b = glGenBuffers();
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_b);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, ELEMENT_BUFFER_SIZE, GL_DYNAMIC_DRAW);

        vertex_b = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, vertex_b);
        glBufferData(GL_ARRAY_BUFFER, VERTEX_BUFFER_SIZE, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, VECTOR_2D_LENGTH, GL_FLOAT, false, VECTOR_FLOAT_2D_SIZE, 0);

        texture_uv_b = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, texture_uv_b);
        glBufferData(GL_ARRAY_BUFFER, VERTEX_BUFFER_SIZE, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(1, VECTOR_2D_LENGTH, GL_FLOAT, false, VECTOR_FLOAT_2D_SIZE, 0);

        command_b = glGenBuffers();
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, command_b);
        glBufferData(GL_DRAW_INDIRECT_BUFFER, COMMAND_BUFFER_SIZE, GL_DYNAMIC_DRAW);

        GPU.share_memory(element_b);
        GPU.share_memory(vertex_b);
        GPU.share_memory(command_b);
        GPU.share_memory(texture_uv_b);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    @Override
    public void tick(float dt)
    {
        GPU.clear_buffer(counters, mesh_size);
        GPU.clear_buffer(offsets, mesh_size);
        GPU.clear_buffer(total, Sizeof.cl_int);
        GPU.clear_buffer(mesh_transfer, ELEMENT_BUFFER_SIZE * 2);

        GPU.GL_count_mesh_instances(query, counters, total, mesh_count);
        GPU.GL_scan_mesh_offsets(counters, offsets, mesh_count);

        int total_instances = GPU.cl_read_pinned_int(total);
        if (total_instances == 0) // highly unlikely, but just in case
        {
            return;
        }

        long data_size = (long)total_instances * Sizeof.cl_int4;
        var details_b = GPU.new_empty_buffer(data_size);

        GPU.GL_write_mesh_details(query, counters, offsets, details_b, mesh_count);
        GPU.GL_count_mesh_batches(details_b, total, total_instances, Constants.Rendering.MAX_BATCH_SIZE);

        int total_batches = GPU.cl_read_pinned_int(total);
        long batch_index_size = (long) total_batches * Sizeof.cl_int;

        var batch_offset_b = GPU.new_empty_buffer(batch_index_size);

        GPU.GL_calculate_batch_offsets(batch_offset_b, details_b, total_instances);

        int[] raw_offsets = new int[total_batches];
        GPU.cl_read_buffer(batch_offset_b, batch_index_size, Pointer.to(raw_offsets));

        glBindVertexArray(vao);
        shader.use();
        texture.bind(GL_TEXTURE0);

        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());
        shader.uploadIntArray("uTextures", texture_slots);

        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);

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

        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        glBindVertexArray(0);

        shader.detach();
        texture.unbind();

        GPU.release_buffer(details_b);
        GPU.release_buffer(batch_offset_b);
    }

    @Override
    public void shutdown()
    {
        GPU.release_buffer(total);
        GPU.release_buffer(query);
        GPU.release_buffer(counters);
        GPU.release_buffer(offsets);
        GPU.release_buffer(mesh_transfer);
    }
}
