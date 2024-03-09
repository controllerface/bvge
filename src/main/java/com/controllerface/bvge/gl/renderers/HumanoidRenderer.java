package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.cl.CLSize;
import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import com.controllerface.bvge.cl.kernels.*;
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

import static com.controllerface.bvge.cl.CLUtils.arg_long;
import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_FLOAT_2D_SIZE;
import static org.lwjgl.opencl.CL10.clReleaseMemObject;
import static org.lwjgl.opengl.GL45C.*;

public class HumanoidRenderer extends GameSystem
{
    private static final int ELEMENT_BUFFER_SIZE = Constants.Rendering.MAX_BATCH_SIZE * Integer.BYTES;
    private static final int VERTEX_BUFFER_SIZE = Constants.Rendering.MAX_BATCH_SIZE * VECTOR_FLOAT_2D_SIZE;
    private static final int COMMAND_BUFFER_SIZE = Constants.Rendering.MAX_BATCH_SIZE * Integer.BYTES * 5;
    private static final int POSITION_ATTRIBUTE = 0;
    private static final int UV_COORD_ATTRIBUTE = 1;

    private final Texture texture;
    private final AbstractShader shader;
    private final int[] texture_slots = {0};
    private final int[] raw_query;
    private final int mesh_count;
    private final long mesh_size;

    private int vao_id;

    private long element_b;
    private long vertex_b;
    private long command_b;
    private long texture_uv_b;

    private long query_ptr;
    private long counters_ptr;
    private long total_ptr;
    private long offsets_ptr;
    private long mesh_transfer_ptr;

    private final GPUProgram mesh_query_p = new MeshQuery();
    private GPUKernel count_instances_k;
    private GPUKernel write_details_k;
    private GPUKernel count_batches_k;
    private GPUKernel calc_offsets_k;
    private GPUKernel transfer_detail_k;
    private GPUKernel transfer_render_k;

    public HumanoidRenderer(ECS ecs)
    {
        super(ecs);
        var model = Models.get_model_by_index(Models.TEST_MODEL_INDEX);
        this.shader = Assets.load_shader("poly_model.glsl");
        this.texture = model.textures().get(0);
        this.mesh_count = model.meshes().length;
        this.mesh_size = (long)mesh_count * CLSize.cl_int;
        this.raw_query = new int[mesh_count];
        for (int i = 0; i < model.meshes().length; i++)
        {
            var m = model.meshes()[i];
            raw_query[i] = m.mesh_id();
        }

        init_GL();
        init_CL();
    }

    private void init_GL()
    {
        this.vao_id = glCreateVertexArrays();

        int element_b = glCreateBuffers();
        glNamedBufferData(element_b, ELEMENT_BUFFER_SIZE, GL_STATIC_DRAW);
        glVertexArrayElementBuffer(vao_id, element_b);

        int vertex_b = GLUtils.new_buffer_vec2(vao_id, POSITION_ATTRIBUTE, VERTEX_BUFFER_SIZE);
        int texture_uv_b = GLUtils.new_buffer_vec2(vao_id, UV_COORD_ATTRIBUTE, VERTEX_BUFFER_SIZE);

        glBindVertexArray(vao_id);
        int command_b = glCreateBuffers();
        glNamedBufferData(command_b, COMMAND_BUFFER_SIZE, GL_STATIC_DRAW);
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, command_b);
        glBindVertexArray(0);

        this.element_b = GPU.share_memory_ex(element_b);
        this.vertex_b = GPU.share_memory_ex(vertex_b);
        this.command_b = GPU.share_memory_ex(command_b);
        this.texture_uv_b = GPU.share_memory_ex(texture_uv_b);
    }

    private void init_CL()
    {
        total_ptr = GPU.cl_new_pinned_int();
        query_ptr = GPU.new_mutable_buffer(raw_query);
        counters_ptr = GPU.new_empty_buffer(mesh_size);
        offsets_ptr = GPU.new_empty_buffer(mesh_size);
        mesh_transfer_ptr = GPU.new_empty_buffer(ELEMENT_BUFFER_SIZE * 2);

        mesh_query_p.init();

        long count_instances_k_ptr = mesh_query_p.kernel_ptr(GPU.Kernel.count_mesh_instances);
        count_instances_k = new CountMeshInstances_k(GPU.command_queue_ptr, count_instances_k_ptr)
            .ptr_arg(CountMeshInstances_k.Args.counters, counters_ptr)
            .ptr_arg(CountMeshInstances_k.Args.query, query_ptr)
            .ptr_arg(CountMeshInstances_k.Args.total, total_ptr)
            .set_arg(CountMeshInstances_k.Args.count, mesh_count)
            .mem_arg(CountMeshInstances_k.Args.hull_mesh_ids, GPU.Buffer.hull_mesh_ids.memory);

        long write_details_k_ptr = mesh_query_p.kernel_ptr(GPU.Kernel.write_mesh_details);
        write_details_k = new WriteMeshDetails_k(GPU.command_queue_ptr, write_details_k_ptr)
            .ptr_arg(WriteMeshDetails_k.Args.counters, counters_ptr)
            .ptr_arg(WriteMeshDetails_k.Args.query, query_ptr)
            .ptr_arg(WriteMeshDetails_k.Args.offsets, offsets_ptr)
            .set_arg(WriteMeshDetails_k.Args.count, mesh_count)
            .mem_arg(WriteMeshDetails_k.Args.hull_mesh_ids, GPU.Buffer.hull_mesh_ids.memory)
            .mem_arg(WriteMeshDetails_k.Args.mesh_references, GPU.Buffer.mesh_references.memory);

        long count_batches_k_ptr = mesh_query_p.kernel_ptr(GPU.Kernel.count_mesh_batches);
        count_batches_k = new CountMeshBatches_k(GPU.command_queue_ptr, count_batches_k_ptr)
            .ptr_arg(CountMeshBatches_k.Args.total, total_ptr)
            .set_arg(CountMeshBatches_k.Args.max_per_batch, Constants.Rendering.MAX_BATCH_SIZE);

        long calc_offsets_k_ptr = mesh_query_p.kernel_ptr(GPU.Kernel.calculate_batch_offsets);
        calc_offsets_k = new CalculateBatchOffsets_k(GPU.command_queue_ptr, calc_offsets_k_ptr);

        long transfer_detail_k_ptr = mesh_query_p.kernel_ptr(GPU.Kernel.transfer_detail_data);
        transfer_detail_k = new TransferDetailData_k(GPU.command_queue_ptr, transfer_detail_k_ptr)
            .ptr_arg(TransferDetailData_k.Args.mesh_transfer, mesh_transfer_ptr);

        long transfer_render_k_ptr = mesh_query_p.kernel_ptr(GPU.Kernel.transfer_render_data);
        transfer_render_k = new TransferRenderData_k(GPU.command_queue_ptr, transfer_render_k_ptr)
            .ptr_arg(TransferRenderData_k.Args.mesh_transfer, mesh_transfer_ptr)
            .mem_arg(TransferRenderData_k.Args.hull_element_tables, GPU.Buffer.hull_element_tables.memory)
            .mem_arg(TransferRenderData_k.Args.hull_mesh_ids, GPU.Buffer.hull_mesh_ids.memory)
            .mem_arg(TransferRenderData_k.Args.mesh_references, GPU.Buffer.mesh_references.memory)
            .mem_arg(TransferRenderData_k.Args.mesh_faces, GPU.Buffer.mesh_faces.memory)
            .mem_arg(TransferRenderData_k.Args.points, GPU.Buffer.points.memory)
            .mem_arg(TransferRenderData_k.Args.vertex_tables, GPU.Buffer.point_vertex_tables.memory)
            .mem_arg(TransferRenderData_k.Args.uv_tables, GPU.Buffer.uv_tables.memory)
            .mem_arg(TransferRenderData_k.Args.texture_uvs, GPU.Buffer.texture_uvs.memory);
    }

    @Override
    public void tick(float dt)
    {
        GPU.clear_buffer(counters_ptr, mesh_size);
        GPU.clear_buffer(offsets_ptr, mesh_size);
        GPU.clear_buffer(total_ptr, CLSize.cl_int);
        GPU.clear_buffer(mesh_transfer_ptr, ELEMENT_BUFFER_SIZE * 2);

        long[] hull_count = arg_long(GPU.ref_memory.next_hull());

        count_instances_k.call(hull_count);

        GPU.scan_int_out(counters_ptr, offsets_ptr, mesh_count);

        int total_instances = GPU.cl_read_pinned_int(total_ptr);
        if (total_instances == 0)
        {
            return;
        }

        long data_size = (long)total_instances * CLSize.cl_int4;
        var mesh_details_ptr = GPU.new_empty_buffer(data_size);

        write_details_k
            .ptr_arg(WriteMeshDetails_k.Args.mesh_details, mesh_details_ptr)
            .call(hull_count);

        count_batches_k
            .ptr_arg(CountMeshBatches_k.Args.mesh_details, mesh_details_ptr)
            .set_arg(CountMeshBatches_k.Args.count, total_instances)
            .call(GPU.global_single_size);

        int total_batches = GPU.cl_read_pinned_int(total_ptr);
        long batch_index_size = (long) total_batches * CLSize.cl_int;

        var mesh_offset_ptr = GPU.new_empty_buffer(batch_index_size);

        calc_offsets_k
            .ptr_arg(CalculateBatchOffsets_k.Args.mesh_offsets, mesh_offset_ptr)
            .ptr_arg(CalculateBatchOffsets_k.Args.mesh_details, mesh_details_ptr)
            .set_arg(CalculateBatchOffsets_k.Args.count, total_instances)
            .call(GPU.global_single_size);

        int[] raw_offsets = new int[total_batches];
        GPU.cl_read_buffer(mesh_offset_ptr, raw_offsets);

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
            int offset = raw_offsets[current_batch];
            int count = next_batch == raw_offsets.length
                ? total_instances - offset
                : raw_offsets[next_batch] - offset;

            transfer_detail_k
                .ptr_arg(TransferDetailData_k.Args.mesh_details, mesh_details_ptr)
                .set_arg(TransferDetailData_k.Args.offset, offset)
                .call(arg_long(count));

            GPU.scan_int2(mesh_transfer_ptr, count);

            transfer_render_k
                .share_mem(command_b)
                .share_mem(vertex_b)
                .share_mem(texture_uv_b)
                .share_mem(element_b)
                .ptr_arg(TransferRenderData_k.Args.command_buffer, command_b)
                .ptr_arg(TransferRenderData_k.Args.vertex_buffer, vertex_b)
                .ptr_arg(TransferRenderData_k.Args.uv_buffer, texture_uv_b)
                .ptr_arg(TransferRenderData_k.Args.element_buffer, element_b)
                .ptr_arg(TransferRenderData_k.Args.mesh_details, mesh_details_ptr)
                .set_arg(TransferRenderData_k.Args.offset, offset)
                .call(arg_long(count));

            glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, 0, count, 0);
        }

        glDisableVertexArrayAttrib(vao_id, POSITION_ATTRIBUTE);
        glDisableVertexArrayAttrib(vao_id, UV_COORD_ATTRIBUTE);

        glBindVertexArray(0);

        shader.detach();

        GPU.release_buffer(mesh_details_ptr);
        GPU.release_buffer(mesh_offset_ptr);
    }

    @Override
    public void shutdown()
    {
        mesh_query_p.destroy();
        GPU.release_buffer(this.element_b);
        GPU.release_buffer(this.vertex_b);
        GPU.release_buffer(this.command_b);
        GPU.release_buffer(this.texture_uv_b);
        GPU.release_buffer(total_ptr);
        GPU.release_buffer(query_ptr);
        GPU.release_buffer(counters_ptr);
        GPU.release_buffer(offsets_ptr);
        GPU.release_buffer(mesh_transfer_ptr);
    }
}
