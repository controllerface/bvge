package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.cl.*;
import com.controllerface.bvge.cl.kernels.*;
import com.controllerface.bvge.cl.programs.MeshQuery;
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
import static com.controllerface.bvge.util.Constants.Rendering.*;
import static org.lwjgl.opengl.GL45C.*;

public class ModelRenderer extends GameSystem
{
    private static final int ELEMENT_BUFFER_SIZE = MAX_BATCH_SIZE * Integer.BYTES;
    private static final int COMMAND_BUFFER_SIZE = MAX_BATCH_SIZE * Integer.BYTES * 5;
    private static final int VERTEX_BUFFER_SIZE  = MAX_BATCH_SIZE * VECTOR_FLOAT_4D_SIZE;
    private static final int UV_BUFFER_SIZE  = MAX_BATCH_SIZE * VECTOR_FLOAT_2D_SIZE;
    private static final int COLOR_BUFFER_SIZE   = MAX_BATCH_SIZE * VECTOR_FLOAT_4D_SIZE;
    private static final int POSITION_ATTRIBUTE  = 0;
    private static final int UV_COORD_ATTRIBUTE  = 1;
    private static final int COLOR_ATTRIBUTE     = 2;


    private final int[] texture_slots = { 0 };

    private final GPUProgram mesh_query_p = new MeshQuery();

    private int vao;
    private int ebo;
    private int cbo;
    private int vbo;
    private int uvo;
    private int vcb;

    private long command_buffer_ptr;
    private long element_buffer_ptr;
    private long vertex_buffer_ptr;
    private long uv_buffer_ptr;
    private long color_buffer_ptr;
    private long query_ptr;
    private long counters_ptr;
    private long total_ptr;
    private long offsets_ptr;
    private long mesh_transfer_ptr;

    private int[] raw_query;
    private int mesh_count;
    private long mesh_size;

    private Texture texture;
    private Shader shader;

    private GPUKernel count_mesh_instances_k;
    private GPUKernel write_mesh_details_k;
    private GPUKernel count_mesh_batches_k;
    private GPUKernel calculate_batch_offsets_k;
    private GPUKernel transfer_detail_data_k;
    private GPUKernel transfer_render_data_k;

    private final int model_id;
    private final String shader_file;

    public ModelRenderer(ECS ecs, String shader_file, int model_id)
    {
        super(ecs);
        this.shader_file = shader_file;
        this.model_id = model_id;
        init_GL();
        init_CL();
    }

    private void init_GL()
    {
        var model = ModelRegistry.get_model_by_index(model_id);
        shader = Assets.load_shader(shader_file);
        texture = model.textures().getFirst();
        mesh_count = model.meshes().length;
        mesh_size = (long)mesh_count * CLSize.cl_int;
        raw_query = new int[mesh_count];
        for (int i = 0; i < model.meshes().length; i++)
        {
            var m = model.meshes()[i];
            raw_query[i] = m.mesh_id();
        }

        vao = glCreateVertexArrays();
        ebo = GLUtils.dynamic_element_buffer(vao, ELEMENT_BUFFER_SIZE);
        cbo = GLUtils.dynamic_command_buffer(vao, COMMAND_BUFFER_SIZE);

        vbo = GLUtils.new_buffer_vec4(vao, POSITION_ATTRIBUTE, VERTEX_BUFFER_SIZE);
        uvo = GLUtils.new_buffer_vec2(vao, UV_COORD_ATTRIBUTE, UV_BUFFER_SIZE);
        vcb = GLUtils.new_buffer_vec4(vao, COLOR_ATTRIBUTE, COLOR_BUFFER_SIZE);

        glEnableVertexArrayAttrib(vao, POSITION_ATTRIBUTE);
        glEnableVertexArrayAttrib(vao, UV_COORD_ATTRIBUTE);
        glEnableVertexArrayAttrib(vao, COLOR_ATTRIBUTE);
    }

    private void init_CL()
    {
        command_buffer_ptr = GPGPU.share_memory(cbo);
        element_buffer_ptr = GPGPU.share_memory(ebo);
        vertex_buffer_ptr = GPGPU.share_memory(vbo);
        uv_buffer_ptr = GPGPU.share_memory(uvo);
        color_buffer_ptr = GPGPU.share_memory(vcb);

        total_ptr = GPGPU.cl_new_pinned_int();
        query_ptr = GPGPU.new_mutable_buffer(raw_query);
        counters_ptr = GPGPU.new_empty_buffer(GPGPU.command_queue_ptr, mesh_size);
        offsets_ptr = GPGPU.new_empty_buffer(GPGPU.command_queue_ptr, mesh_size);
        mesh_transfer_ptr = GPGPU.new_empty_buffer(GPGPU.command_queue_ptr, ELEMENT_BUFFER_SIZE * 2);

        mesh_query_p.init();

        long count_instances_k_ptr = mesh_query_p.kernel_ptr(Kernel.count_mesh_instances);
        count_mesh_instances_k = new CountMeshInstances_k(GPGPU.command_queue_ptr, count_instances_k_ptr)
            .ptr_arg(CountMeshInstances_k.Args.counters, counters_ptr)
            .ptr_arg(CountMeshInstances_k.Args.query, query_ptr)
            .ptr_arg(CountMeshInstances_k.Args.total, total_ptr)
            .set_arg(CountMeshInstances_k.Args.count, mesh_count)
            .buf_arg(CountMeshInstances_k.Args.hull_mesh_ids, GPGPU.core_memory.buffer(BufferType.MIRROR_HULL_MESH_ID));

        long write_details_k_ptr = mesh_query_p.kernel_ptr(Kernel.write_mesh_details);
        write_mesh_details_k = new WriteMeshDetails_k(GPGPU.command_queue_ptr, write_details_k_ptr)
            .ptr_arg(WriteMeshDetails_k.Args.counters, counters_ptr)
            .ptr_arg(WriteMeshDetails_k.Args.query, query_ptr)
            .ptr_arg(WriteMeshDetails_k.Args.offsets, offsets_ptr)
            .set_arg(WriteMeshDetails_k.Args.count, mesh_count)
            .buf_arg(WriteMeshDetails_k.Args.hull_mesh_ids, GPGPU.core_memory.buffer(BufferType.MIRROR_HULL_MESH_ID))
            .buf_arg(WriteMeshDetails_k.Args.mesh_vertex_tables, GPGPU.core_memory.buffer(BufferType.MESH_VERTEX_TABLE))
            .buf_arg(WriteMeshDetails_k.Args.mesh_face_tables, GPGPU.core_memory.buffer(BufferType.MESH_FACE_TABLE));

        long count_batches_k_ptr = mesh_query_p.kernel_ptr(Kernel.count_mesh_batches);
        count_mesh_batches_k = new CountMeshBatches_k(GPGPU.command_queue_ptr, count_batches_k_ptr)
            .ptr_arg(CountMeshBatches_k.Args.total, total_ptr)
            .set_arg(CountMeshBatches_k.Args.max_per_batch, Constants.Rendering.MAX_BATCH_SIZE);

        long calc_offsets_k_ptr = mesh_query_p.kernel_ptr(Kernel.calculate_batch_offsets);
        calculate_batch_offsets_k = new CalculateBatchOffsets_k(GPGPU.command_queue_ptr, calc_offsets_k_ptr);

        long transfer_detail_k_ptr = mesh_query_p.kernel_ptr(Kernel.transfer_detail_data);
        transfer_detail_data_k = new TransferDetailData_k(GPGPU.command_queue_ptr, transfer_detail_k_ptr)
            .ptr_arg(TransferDetailData_k.Args.mesh_transfer, mesh_transfer_ptr);

        long transfer_render_k_ptr = mesh_query_p.kernel_ptr(Kernel.transfer_render_data);
        transfer_render_data_k = new TransferRenderData_k(GPGPU.command_queue_ptr, transfer_render_k_ptr)
            .ptr_arg(TransferRenderData_k.Args.command_buffer, command_buffer_ptr)
            .ptr_arg(TransferRenderData_k.Args.element_buffer, element_buffer_ptr)
            .ptr_arg(TransferRenderData_k.Args.vertex_buffer, vertex_buffer_ptr)
            .ptr_arg(TransferRenderData_k.Args.uv_buffer, uv_buffer_ptr)
            .ptr_arg(TransferRenderData_k.Args.color_buffer, color_buffer_ptr)
            .ptr_arg(TransferRenderData_k.Args.mesh_transfer, mesh_transfer_ptr)
            .buf_arg(TransferRenderData_k.Args.hull_point_tables, GPGPU.core_memory.buffer(BufferType.MIRROR_HULL_POINT_TABLE))
            .buf_arg(TransferRenderData_k.Args.hull_mesh_ids, GPGPU.core_memory.buffer(BufferType.MIRROR_HULL_MESH_ID))
            .buf_arg(TransferRenderData_k.Args.hull_flags, GPGPU.core_memory.buffer(BufferType.MIRROR_HULL_FLAG))
            .buf_arg(TransferRenderData_k.Args.mesh_vertex_tables, GPGPU.core_memory.buffer(BufferType.MESH_VERTEX_TABLE))
            .buf_arg(TransferRenderData_k.Args.mesh_face_tables, GPGPU.core_memory.buffer(BufferType.MESH_FACE_TABLE))
            .buf_arg(TransferRenderData_k.Args.mesh_faces, GPGPU.core_memory.buffer(BufferType.MESH_FACE))
            .buf_arg(TransferRenderData_k.Args.points, GPGPU.core_memory.buffer(BufferType.MIRROR_POINT))
            .buf_arg(TransferRenderData_k.Args.point_hit_counts, GPGPU.core_memory.buffer(BufferType.MIRROR_POINT_HIT_COUNT))
            .buf_arg(TransferRenderData_k.Args.point_vertex_references, GPGPU.core_memory.buffer(BufferType.MIRROR_POINT_VERTEX_REFERENCE))
            .buf_arg(TransferRenderData_k.Args.uv_tables, GPGPU.core_memory.buffer(BufferType.VERTEX_UV_TABLE))
            .buf_arg(TransferRenderData_k.Args.texture_uvs, GPGPU.core_memory.buffer(BufferType.VERTEX_TEXTURE_UV));
    }

    @Override
    public void tick(float dt)
    {
        GPGPU.cl_zero_buffer(GPGPU.command_queue_ptr, counters_ptr, mesh_size);
        GPGPU.cl_zero_buffer(GPGPU.command_queue_ptr, offsets_ptr, mesh_size);
        GPGPU.cl_zero_buffer(GPGPU.command_queue_ptr, total_ptr, CLSize.cl_int);
        GPGPU.cl_zero_buffer(GPGPU.command_queue_ptr, mesh_transfer_ptr, ELEMENT_BUFFER_SIZE * 2);

        long[] hull_count = arg_long(GPGPU.core_memory.next_hull());

        count_mesh_instances_k.call(hull_count);

        GPGPU.scan_int_out(counters_ptr, offsets_ptr, mesh_count);

        int total_instances = GPGPU.cl_read_pinned_int(GPGPU.command_queue_ptr, total_ptr);
        if (total_instances == 0)
        {
            return;
        }

        long data_size = (long)total_instances * CLSize.cl_int4;
        var mesh_details_ptr = GPGPU.new_empty_buffer(GPGPU.command_queue_ptr, data_size);

        write_mesh_details_k
            .ptr_arg(WriteMeshDetails_k.Args.mesh_details, mesh_details_ptr)
            .call(hull_count);

        count_mesh_batches_k
            .ptr_arg(CountMeshBatches_k.Args.mesh_details, mesh_details_ptr)
            .set_arg(CountMeshBatches_k.Args.count, total_instances)
            .call(GPGPU.global_single_size);

        int total_batches = GPGPU.cl_read_pinned_int(GPGPU.command_queue_ptr, total_ptr);
        long batch_index_size = (long) total_batches * CLSize.cl_int;

        var mesh_offset_ptr = GPGPU.new_empty_buffer(GPGPU.command_queue_ptr, batch_index_size);

        calculate_batch_offsets_k
            .ptr_arg(CalculateBatchOffsets_k.Args.mesh_offsets, mesh_offset_ptr)
            .ptr_arg(CalculateBatchOffsets_k.Args.mesh_details, mesh_details_ptr)
            .set_arg(CalculateBatchOffsets_k.Args.count, total_instances)
            .call(GPGPU.global_single_size);

        int[] raw_offsets = new int[total_batches];
        GPGPU.cl_read_buffer(GPGPU.command_queue_ptr, mesh_offset_ptr, raw_offsets);

        glBindVertexArray(vao);

        glEnable(GL_DEPTH_TEST);
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, cbo);

        shader.use();
        texture.bind(0);

        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());
        shader.uploadIntArray("uTextures", texture_slots);

        for (int current_batch = 0; current_batch < raw_offsets.length; current_batch++)
        {
            int next_batch = current_batch + 1;
            int offset = raw_offsets[current_batch];
            int count = next_batch == raw_offsets.length
                ? total_instances - offset
                : raw_offsets[next_batch] - offset;

            transfer_detail_data_k
                .ptr_arg(TransferDetailData_k.Args.mesh_details, mesh_details_ptr)
                .set_arg(TransferDetailData_k.Args.offset, offset)
                .call(arg_long(count));

            GPGPU.scan_int2(mesh_transfer_ptr, count);

            transfer_render_data_k
                .share_mem(command_buffer_ptr)
                .share_mem(element_buffer_ptr)
                .share_mem(vertex_buffer_ptr)
                .share_mem(uv_buffer_ptr)
                .share_mem(color_buffer_ptr)
                .ptr_arg(TransferRenderData_k.Args.mesh_details, mesh_details_ptr)
                .set_arg(TransferRenderData_k.Args.offset, offset)
                .call(arg_long(count));

            glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, 0, count, 0);
        }

        glBindVertexArray(0);

        shader.detach();

        GPGPU.cl_release_buffer(mesh_details_ptr);
        GPGPU.cl_release_buffer(mesh_offset_ptr);
    }

    @Override
    public void shutdown()
    {
        glDeleteVertexArrays(vao);
        glDeleteBuffers(cbo);
        glDeleteBuffers(ebo);
        glDeleteBuffers(vbo);
        glDeleteBuffers(uvo);
        shader.destroy();
        texture.destroy();
        mesh_query_p.destroy();
        GPGPU.cl_release_buffer(element_buffer_ptr);
        GPGPU.cl_release_buffer(vertex_buffer_ptr);
        GPGPU.cl_release_buffer(command_buffer_ptr);
        GPGPU.cl_release_buffer(uv_buffer_ptr);
        GPGPU.cl_release_buffer(total_ptr);
        GPGPU.cl_release_buffer(query_ptr);
        GPGPU.cl_release_buffer(counters_ptr);
        GPGPU.cl_release_buffer(offsets_ptr);
        GPGPU.cl_release_buffer(mesh_transfer_ptr);
    }
}
