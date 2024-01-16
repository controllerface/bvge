package com.controllerface.bvge.ecs.systems.renderers;

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
import org.lwjgl.BufferUtils;
import org.lwjgl.opengl.GL30;
import org.lwjgl.opengl.GL43C;

import java.util.Arrays;

import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_2D_LENGTH;
import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_FLOAT_2D_SIZE;
import static org.lwjgl.opengl.GL11C.GL_FLOAT;
import static org.lwjgl.opengl.GL15C.GL_ARRAY_BUFFER;
import static org.lwjgl.opengl.GL15C.glBindBuffer;
import static org.lwjgl.opengl.GL15C.*;
import static org.lwjgl.opengl.GL20.glDisableVertexAttribArray;
import static org.lwjgl.opengl.GL20.glEnableVertexAttribArray;
import static org.lwjgl.opengl.GL20C.glVertexAttribPointer;
import static org.lwjgl.opengl.GL30C.glBindVertexArray;
import static org.lwjgl.opengl.GL30C.glGenVertexArrays;
import static org.lwjgl.opengl.GL32C.GL_SYNC_FENCE;
import static org.lwjgl.opengl.GL40C.GL_DRAW_INDIRECT_BUFFER;
import static org.lwjgl.opengl.GL42C.GL_ALL_BARRIER_BITS;
import static org.lwjgl.opengl.GL42C.glMemoryBarrier;
import static org.lwjgl.opengl.GL43C.glMultiDrawElementsIndirect;

public class HumanoidRenderer extends GameSystem
{
    private static final int ELEMENT_BUFFER_SIZE = Constants.Rendering.MAX_BATCH_SIZE * Integer.BYTES;
    private static final int VERTEX_BUFFER_SIZE = Constants.Rendering.MAX_BATCH_SIZE * VECTOR_FLOAT_2D_SIZE;
    private static final int COMMAND_BUFFER_SIZE = Constants.Rendering.MAX_BATCH_SIZE * Integer.BYTES * 5;

    // todo: determine the sizes required for a single render batch and calculate them
    private Texture texture;
    private final AbstractShader shader;
    private int vao;
    private int vbo;
    private int ebo;
    private int cbo;

    private int mesh_count;
    private long mesh_size;
    private cl_mem query;
    private cl_mem counters;
    private cl_mem total;
    private cl_mem offsets;
    private cl_mem vertex_transfer;
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
        int[] raw_counters = new int[mesh_count];
        for (int i = 0; i < model.meshes().length; i++)
        {
            var m = model.meshes()[i];
            raw_query[i] = m.mesh_id();
        }

        total = GPU.cl_new_pinned_int();
        query = GPU.new_mutable_buffer(mesh_size, Pointer.to(raw_query));
        counters = GPU.new_mutable_buffer(mesh_size, Pointer.to(raw_counters));
        offsets = GPU.new_empty_buffer(mesh_size);
        vertex_transfer = GPU.new_empty_buffer(ELEMENT_BUFFER_SIZE);
        mesh_transfer = GPU.new_empty_buffer(ELEMENT_BUFFER_SIZE * 2);

        vao = glGenVertexArrays();
        glBindVertexArray(vao);

        ebo = glGenBuffers();
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, ELEMENT_BUFFER_SIZE, GL_DYNAMIC_DRAW);

        vbo = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, VERTEX_BUFFER_SIZE, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, VECTOR_2D_LENGTH, GL_FLOAT, false, VECTOR_FLOAT_2D_SIZE, 0);

        cbo = glGenBuffers();
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, cbo);
        glBufferData(GL_DRAW_INDIRECT_BUFFER, COMMAND_BUFFER_SIZE, GL_DYNAMIC_DRAW);

        GPU.share_memory(ebo);
        GPU.share_memory(vbo);
        GPU.share_memory(cbo);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    @Override
    public void tick(float dt)
    {
        GPU.clear_buffer(counters, mesh_size);
        GPU.clear_buffer(offsets, mesh_size);
        GPU.clear_buffer(total, Sizeof.cl_int);

        GPU.count_mesh_instances(query, counters, total, mesh_count);
        GPU.scan_mesh_offsets(counters, offsets, mesh_count);

        // todo: bail if none? even though unlikely?
        int total_instances = GPU.cl_read_pinned_int(total);
        long data_size = (long)total_instances * Sizeof.cl_int4;
        var details_buffer = GPU.new_empty_buffer(data_size);
        GPU.clear_buffer(counters, mesh_size);

        GPU.write_mesh_details(query, counters, offsets, details_buffer, mesh_count);
        GPU.count_mesh_batches(details_buffer, total, total_instances);

        int total_batches = GPU.cl_read_pinned_int(total);
        long batch_index_size = (long) total_batches * Sizeof.cl_int;

        var batch_offsets = GPU.new_empty_buffer(batch_index_size);

        GPU.calculate_batch_offsets(batch_offsets, details_buffer, total_instances);

        int[] raw_offsets = new int[total_batches];
        GPU.cl_read_buffer(batch_offsets, batch_index_size, Pointer.to(raw_offsets));



        glBindVertexArray(vao);
        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());
        glEnableVertexAttribArray(0);

        for (int current_batch = 0; current_batch < raw_offsets.length; current_batch++)
        {
            int next_batch = current_batch + 1;
            int start = raw_offsets[current_batch];
            int count = next_batch == raw_offsets.length
                ? total_instances - start
                : raw_offsets[next_batch] - start;

            // calculate mesh data offsets for this batch
            GPU.transfer_detail_data(details_buffer, mesh_transfer, count, start);
            GPU.transfer_render_data(ebo, vbo, cbo, details_buffer, mesh_transfer, count, start);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);

            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);

            //glDrawElements(GL_TRIANGLES, 42, GL_UNSIGNED_INT, 0);


            glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, 0, count, 0);

            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
        }

        glDisableVertexAttribArray(0);
        GL30.glBindVertexArray(0);

        GPU.release_buffer(details_buffer);
        GPU.release_buffer(batch_offsets);
    }

    @Override
    public void shutdown()
    {
        GPU.release_buffer(total);
        GPU.release_buffer(query);
        GPU.release_buffer(counters);
        GPU.release_buffer(offsets);
        GPU.release_buffer(vertex_transfer);
        GPU.release_buffer(mesh_transfer);
    }
}
