package com.controllerface.bvge.ecs.systems.renderers;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.geometry.Models;
import com.controllerface.bvge.gl.Texture;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_mem;

import static org.lwjgl.opengl.GL30.glGenVertexArrays;

public class HumanoidRenderer extends GameSystem
{
    // todo: determine the sizes required for a single render batch and calculate them
    private Texture texture;
    private int vao;
    private int mesh_count;
    private long mesh_size;
    private cl_mem query_buffer;
    private cl_mem counter_buffer;
    private cl_mem total_buffer;
    private cl_mem offset_buffer;

    public HumanoidRenderer(ECS ecs)
    {
        super(ecs);
        // todo: load a shader to render the meshes. The crate shader can be used a base,
        //  but the color data should be removed.
        init();
    }

    private void init()
    {
        var model = Models.get_model_by_index(Models.TEST_MODEL_INDEX);
        this.texture = model.textures().get(0);

        mesh_count = model.meshes().length;
        mesh_size = (long)mesh_count * Sizeof.cl_int;
        int[] query = new int[mesh_count];
        int[] counters = new int[mesh_count];
        for (int i = 0; i < model.meshes().length; i++)
        {
            var m = model.meshes()[i];
            query[i] = m.mesh_id();
        }

        total_buffer = GPU.cl_new_pinned_int();
        query_buffer = GPU.new_mutable_buffer(mesh_size, Pointer.to(query));
        counter_buffer = GPU.new_mutable_buffer(mesh_size, Pointer.to(counters));
        offset_buffer = GPU.new_empty_buffer(mesh_size);

//        vao = glGenVertexArrays();
//        glBindVertexArray(vao);

    }

    @Override
    public void tick(float dt)
    {
        GPU.clear_buffer(counter_buffer, (long)mesh_count * Sizeof.cl_int);
        GPU.clear_buffer(offset_buffer, (long)mesh_count * Sizeof.cl_int);
        GPU.clear_buffer(total_buffer, Sizeof.cl_int);

        GPU.count_mesh_instances(query_buffer, counter_buffer, total_buffer, mesh_count);

        GPU.test_query_2(counter_buffer, offset_buffer, mesh_count);

        int total_instances = GPU.cl_read_pinned_int(total_buffer);
        System.out.println("INSTANCES: " + total_instances);

        long data_size = (long)total_instances * Sizeof.cl_int4;
        var details_buffer = GPU.new_empty_buffer(data_size);

        GPU.write_mesh_data(query_buffer, counter_buffer, offset_buffer, details_buffer, mesh_count);

        GPU.count_mesh_batches(details_buffer, total_buffer, total_instances);

        int total_batches = GPU.cl_read_pinned_int(total_buffer);
        System.out.println("BATCHES: " + total_batches);

        var batch_offsets = GPU.new_empty_buffer(total_batches);

        GPU.calculate_batch_offsets(batch_offsets, details_buffer, total_instances);






        GPU.release_buffer(details_buffer);
        GPU.release_buffer(batch_offsets);
    }

    @Override
    public void shutdown()
    {
        GPU.release_buffer(total_buffer);
        GPU.release_buffer(query_buffer);
        GPU.release_buffer(counter_buffer);
        GPU.release_buffer(offset_buffer);
    }
}
