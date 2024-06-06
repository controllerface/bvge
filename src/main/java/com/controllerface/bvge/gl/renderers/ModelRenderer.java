package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.cl.*;
import com.controllerface.bvge.cl.kernels.*;
import com.controllerface.bvge.cl.programs.*;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.Component;
import com.controllerface.bvge.ecs.components.ControlPoints;
import com.controllerface.bvge.ecs.components.GameComponent;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.editor.Editor;
import com.controllerface.bvge.geometry.Model;
import com.controllerface.bvge.geometry.ModelRegistry;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.gl.GLUtils;
import com.controllerface.bvge.gl.Texture;
import com.controllerface.bvge.physics.UniformGrid;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;

import java.nio.ByteBuffer;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Objects;

import static com.controllerface.bvge.cl.CLUtils.arg_long;
import static com.controllerface.bvge.util.Constants.Rendering.*;
import static org.lwjgl.opengl.GL45C.*;

public class ModelRenderer extends GameSystem
{
    private final UniformGrid uniformGrid;

    private static final int ELEMENT_BUFFER_SIZE = MAX_BATCH_SIZE * Integer.BYTES;
    private static final int TEXTURE_BUFFER_SIZE = MAX_BATCH_SIZE * SCALAR_FLOAT_SIZE;
    private static final int COMMAND_BUFFER_SIZE = MAX_BATCH_SIZE * Integer.BYTES * 5;
    private static final int VERTEX_BUFFER_SIZE  = MAX_BATCH_SIZE * VECTOR_FLOAT_4D_SIZE;
    private static final int UV_BUFFER_SIZE  = MAX_BATCH_SIZE * VECTOR_FLOAT_2D_SIZE;
    private static final int COLOR_BUFFER_SIZE   = MAX_BATCH_SIZE * VECTOR_FLOAT_4D_SIZE;
    private static final int POSITION_ATTRIBUTE  = 0;
    private static final int UV_COORD_ATTRIBUTE  = 1;
    private static final int COLOR_ATTRIBUTE     = 2;
    private static final int TEXTURE_ATTRIBUTE   = 3;

    // todo: base this off of number of texture units
    private final int[] texture_slots = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    private final GPUProgram p_mesh_query = new MeshQuery();
    private final GPUProgram p_scan_int2_array = new ScanInt2Array();
    private final GPUProgram p_scan_int_array = new ScanIntArray();
    private final GPUProgram p_scan_int_array_out = new ScanIntArrayOut();

    private int vao;
    private int ebo;
    private int cbo;
    private int vbo_position;
    private int vbo_texture_uv;
    private int vbo_color;
    private int vbo_texture_slot;

    private long ptr_command_buffer;
    private long ptr_element_buffer;
    private long ptr_vertex_buffer;
    private long ptr_uv_buffer;
    private long ptr_color_buffer;
    private long ptr_slot_buffer;
    private long ptr_query;
    private long ptr_counters;
    private long ptr_offsets;
    private long ptr_mesh_transfer;
    private ByteBuffer svm_total;

    private int[] raw_query;
    private int mesh_count;
    private long mesh_size;

    private Texture[] textures;
    private Shader shader;

    private GPUKernel k_count_mesh_instances;
    private GPUKernel k_write_mesh_details;
    private GPUKernel k_count_mesh_batches;
    private GPUKernel k_calculate_batch_offsets;
    private GPUKernel k_transfer_detail_data;
    private GPUKernel k_transfer_render_data;
    private GPUKernel k_scan_int_single_block;
    private GPUKernel k_scan_int_multi_block;
    private GPUKernel k_complete_int_multi_block;
    private GPUKernel k_scan_int2_single_block;
    private GPUKernel k_scan_int2_multi_block;
    private GPUKernel k_complete_int2_multi_block;
    private GPUKernel k_scan_int_single_block_out;
    private GPUKernel k_scan_int_multi_block_out;
    private GPUKernel k_complete_int_multi_block_out;

    private final int[] model_ids;
    private final String shader_file;

    public ModelRenderer(ECS ecs, UniformGrid uniformGrid, int ... model_ids)
    {
        super(ecs);
        this.uniformGrid = uniformGrid;
        this.shader_file = "block_model.glsl";
        this.model_ids = model_ids;
        init_GL();
        init_CL();
    }

    private void init_GL()
    {
        shader = Assets.load_shader(shader_file);

        Model[] models = new Model[model_ids.length];

        // todo: sum and validate texture count here
        for (int i = 0; i < model_ids.length; i++)
        {
            var model = ModelRegistry.get_model_by_index(model_ids[i]);
            models[i] = model;
            mesh_count += model.meshes().length;
        }

        mesh_size = (long)mesh_count * CLSize.cl_int;
        raw_query = new int[mesh_count * 2]; // int2

        var texture_buffer = new LinkedHashSet<Texture>();

        int query_index = 0;
        for (var model : models)
        {
            var model_texture = model.textures().getFirst();
            texture_buffer.add(model_texture);

            int tex_slot = 0;
            for (var texture : texture_buffer)
            {
                if (model_texture.equals(texture)) break;
                tex_slot++;
            }

            for (var mesh : model.meshes())
            {
                raw_query[query_index++] = mesh.mesh_id();
                raw_query[query_index++] = tex_slot;
            }
        }

        textures = new Texture[texture_buffer.size()];
        int texture_index = 0;
        for (var texture : texture_buffer)
        {
            textures[texture_index++] = texture;
        }

        vao = glCreateVertexArrays();
        ebo = GLUtils.dynamic_element_buffer(vao, ELEMENT_BUFFER_SIZE);
        cbo = GLUtils.dynamic_command_buffer(vao, COMMAND_BUFFER_SIZE);

        vbo_position     = GLUtils.new_buffer_vec4(vao, POSITION_ATTRIBUTE, VERTEX_BUFFER_SIZE);
        vbo_texture_uv   = GLUtils.new_buffer_vec2(vao, UV_COORD_ATTRIBUTE, UV_BUFFER_SIZE);
        vbo_color        = GLUtils.new_buffer_vec4(vao, COLOR_ATTRIBUTE, COLOR_BUFFER_SIZE);
        vbo_texture_slot = GLUtils.new_buffer_float(vao, TEXTURE_ATTRIBUTE, TEXTURE_BUFFER_SIZE);

        glEnableVertexArrayAttrib(vao, POSITION_ATTRIBUTE);
        glEnableVertexArrayAttrib(vao, UV_COORD_ATTRIBUTE);
        glEnableVertexArrayAttrib(vao, COLOR_ATTRIBUTE);
        glEnableVertexArrayAttrib(vao, TEXTURE_ATTRIBUTE);
    }

    private void init_CL()
    {
        ptr_command_buffer = GPGPU.share_memory(cbo);
        ptr_element_buffer = GPGPU.share_memory(ebo);
        ptr_vertex_buffer = GPGPU.share_memory(vbo_position);
        ptr_uv_buffer = GPGPU.share_memory(vbo_texture_uv);
        ptr_color_buffer = GPGPU.share_memory(vbo_color);
        ptr_slot_buffer = GPGPU.share_memory(vbo_texture_slot);
        svm_total = GPGPU.cl_new_svm_int();
        ptr_query = GPGPU.new_mutable_buffer(raw_query);
        ptr_counters = GPGPU.new_empty_buffer(GPGPU.ptr_render_queue, mesh_size);
        ptr_offsets = GPGPU.new_empty_buffer(GPGPU.ptr_render_queue, mesh_size);
        ptr_mesh_transfer = GPGPU.new_empty_buffer(GPGPU.ptr_render_queue, ELEMENT_BUFFER_SIZE * 2);

        p_mesh_query.init();
        p_scan_int_array.init();
        p_scan_int_array_out.init();
        p_scan_int2_array.init();

        long k_ptr_count_instances = p_mesh_query.kernel_ptr(Kernel.count_mesh_instances);
        k_count_mesh_instances = new CountMeshInstances_k(GPGPU.ptr_render_queue, k_ptr_count_instances)
            .ptr_arg(CountMeshInstances_k.Args.counters, ptr_counters)
            .ptr_arg(CountMeshInstances_k.Args.query, ptr_query)
            .ptr_arg(CountMeshInstances_k.Args.total, svm_total)
            .set_arg(CountMeshInstances_k.Args.count, mesh_count)
            .buf_arg(CountMeshInstances_k.Args.hull_mesh_ids, GPGPU.core_memory.buffer(BufferType.MIRROR_HULL_MESH_ID))
            .buf_arg(CountMeshInstances_k.Args.hull_flags, GPGPU.core_memory.buffer(BufferType.MIRROR_HULL_FLAG));

        long k_ptr_write_details = p_mesh_query.kernel_ptr(Kernel.write_mesh_details);
        k_write_mesh_details = new WriteMeshDetails_k(GPGPU.ptr_render_queue, k_ptr_write_details)
            .ptr_arg(WriteMeshDetails_k.Args.counters, ptr_counters)
            .ptr_arg(WriteMeshDetails_k.Args.query, ptr_query)
            .ptr_arg(WriteMeshDetails_k.Args.offsets, ptr_offsets)
            .set_arg(WriteMeshDetails_k.Args.count, mesh_count)
            .buf_arg(WriteMeshDetails_k.Args.hull_mesh_ids, GPGPU.core_memory.buffer(BufferType.MIRROR_HULL_MESH_ID))
            .buf_arg(WriteMeshDetails_k.Args.hull_flags, GPGPU.core_memory.buffer(BufferType.MIRROR_HULL_FLAG))
            .buf_arg(WriteMeshDetails_k.Args.mesh_vertex_tables, GPGPU.core_memory.buffer(BufferType.MESH_VERTEX_TABLE))
            .buf_arg(WriteMeshDetails_k.Args.mesh_face_tables, GPGPU.core_memory.buffer(BufferType.MESH_FACE_TABLE));

        long k_ptr_count_batches = p_mesh_query.kernel_ptr(Kernel.count_mesh_batches);
        k_count_mesh_batches = new CountMeshBatches_k(GPGPU.ptr_render_queue, k_ptr_count_batches)
            .ptr_arg(CountMeshBatches_k.Args.total, svm_total)
            .set_arg(CountMeshBatches_k.Args.max_per_batch, Constants.Rendering.MAX_BATCH_SIZE);

        long k_ptr_calc_offsets = p_mesh_query.kernel_ptr(Kernel.calculate_batch_offsets);
        k_calculate_batch_offsets = new CalculateBatchOffsets_k(GPGPU.ptr_render_queue, k_ptr_calc_offsets);

        long k_ptr_transfer_detail = p_mesh_query.kernel_ptr(Kernel.transfer_detail_data);
        k_transfer_detail_data = new TransferDetailData_k(GPGPU.ptr_render_queue, k_ptr_transfer_detail)
            .ptr_arg(TransferDetailData_k.Args.mesh_transfer, ptr_mesh_transfer);

        long k_ptr_transfer_render = p_mesh_query.kernel_ptr(Kernel.transfer_render_data);
        k_transfer_render_data = new TransferRenderData_k(GPGPU.ptr_render_queue, k_ptr_transfer_render)
            .ptr_arg(TransferRenderData_k.Args.command_buffer, ptr_command_buffer)
            .ptr_arg(TransferRenderData_k.Args.element_buffer, ptr_element_buffer)
            .ptr_arg(TransferRenderData_k.Args.vertex_buffer, ptr_vertex_buffer)
            .ptr_arg(TransferRenderData_k.Args.uv_buffer, ptr_uv_buffer)
            .ptr_arg(TransferRenderData_k.Args.color_buffer, ptr_color_buffer)
            .ptr_arg(TransferRenderData_k.Args.slot_buffer, ptr_slot_buffer)
            .ptr_arg(TransferRenderData_k.Args.mesh_transfer, ptr_mesh_transfer)
            .buf_arg(TransferRenderData_k.Args.hull_point_tables, GPGPU.core_memory.buffer(BufferType.MIRROR_HULL_POINT_TABLE))
            .buf_arg(TransferRenderData_k.Args.hull_mesh_ids, GPGPU.core_memory.buffer(BufferType.MIRROR_HULL_MESH_ID))
            .buf_arg(TransferRenderData_k.Args.hull_entity_ids, GPGPU.core_memory.buffer(BufferType.MIRROR_HULL_ENTITY_ID))
            .buf_arg(TransferRenderData_k.Args.hull_flags, GPGPU.core_memory.buffer(BufferType.MIRROR_HULL_FLAG))
            .buf_arg(TransferRenderData_k.Args.hull_uv_offsets, GPGPU.core_memory.buffer(BufferType.MIRROR_HULL_UV_OFFSET))
            .buf_arg(TransferRenderData_k.Args.hull_integrity, GPGPU.core_memory.buffer(BufferType.MIRROR_HULL_INTEGRITY))
            .buf_arg(TransferRenderData_k.Args.entity_flags, GPGPU.core_memory.buffer(BufferType.MIRROR_ENTITY_FLAG))
            .buf_arg(TransferRenderData_k.Args.mesh_vertex_tables, GPGPU.core_memory.buffer(BufferType.MESH_VERTEX_TABLE))
            .buf_arg(TransferRenderData_k.Args.mesh_face_tables, GPGPU.core_memory.buffer(BufferType.MESH_FACE_TABLE))
            .buf_arg(TransferRenderData_k.Args.mesh_faces, GPGPU.core_memory.buffer(BufferType.MESH_FACE))
            .buf_arg(TransferRenderData_k.Args.points, GPGPU.core_memory.buffer(BufferType.MIRROR_POINT))
            .buf_arg(TransferRenderData_k.Args.point_hit_counts, GPGPU.core_memory.buffer(BufferType.MIRROR_POINT_HIT_COUNT))
            .buf_arg(TransferRenderData_k.Args.point_vertex_references, GPGPU.core_memory.buffer(BufferType.MIRROR_POINT_VERTEX_REFERENCE))
            .buf_arg(TransferRenderData_k.Args.uv_tables, GPGPU.core_memory.buffer(BufferType.VERTEX_UV_TABLE))
            .buf_arg(TransferRenderData_k.Args.texture_uvs, GPGPU.core_memory.buffer(BufferType.VERTEX_TEXTURE_UV));

        long k_ptr_scan_int_array_single = p_scan_int_array.kernel_ptr(Kernel.scan_int_single_block);
        long k_ptr_scan_int_array_multi = p_scan_int_array.kernel_ptr(Kernel.scan_int_multi_block);
        long k_ptr_scan_int_array_comp = p_scan_int_array.kernel_ptr(Kernel.complete_int_multi_block);
        k_scan_int_single_block = new ScanIntSingleBlock_k(GPGPU.ptr_render_queue, k_ptr_scan_int_array_single);
        k_scan_int_multi_block = new ScanIntMultiBlock_k(GPGPU.ptr_render_queue, k_ptr_scan_int_array_multi);
        k_complete_int_multi_block = new CompleteIntMultiBlock_k(GPGPU.ptr_render_queue, k_ptr_scan_int_array_comp);

        long k_ptr_scan_int2_array_single = p_scan_int2_array.kernel_ptr(Kernel.scan_int2_single_block);
        long k_ptr_scan_int2_array_multi = p_scan_int2_array.kernel_ptr(Kernel.scan_int2_multi_block);
        long k_ptr_scan_int2_array_comp = p_scan_int2_array.kernel_ptr(Kernel.complete_int2_multi_block);
        k_scan_int2_single_block = new ScanInt2SingleBlock_k(GPGPU.ptr_render_queue, k_ptr_scan_int2_array_single);
        k_scan_int2_multi_block = new ScanInt2MultiBlock_k(GPGPU.ptr_render_queue, k_ptr_scan_int2_array_multi);
        k_complete_int2_multi_block = new CompleteInt2MultiBlock_k(GPGPU.ptr_render_queue, k_ptr_scan_int2_array_comp);

        long k_ptr_scan_int_array_out_single = p_scan_int_array_out.kernel_ptr(Kernel.scan_int_single_block_out);
        long k_ptr_scan_int_array_out_multi = p_scan_int_array_out.kernel_ptr(Kernel.scan_int_multi_block_out);
        long k_ptr_scan_int_array_out_comp = p_scan_int_array_out.kernel_ptr(Kernel.complete_int_multi_block_out);
        k_scan_int_single_block_out = new ScanIntSingleBlockOut_k(GPGPU.ptr_render_queue, k_ptr_scan_int_array_out_single);
        k_scan_int_multi_block_out = new ScanIntMultiBlockOut_k(GPGPU.ptr_render_queue, k_ptr_scan_int_array_out_multi);
        k_complete_int_multi_block_out = new CompleteIntMultiBlockOut_k(GPGPU.ptr_render_queue, k_ptr_scan_int_array_out_comp);
    }

    private record BatchData( int[] raw_offsets, int total_instances, long mesh_details_ptr, long mesh_texture_ptr) { }

    private BatchData tick_CL()
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;

        GPGPU.cl_zero_buffer(GPGPU.ptr_render_queue, ptr_counters, mesh_size);
        GPGPU.cl_zero_buffer(GPGPU.ptr_render_queue, ptr_offsets, mesh_size);
        GPGPU.cl_zero_buffer(GPGPU.ptr_render_queue, svm_total, CLSize.cl_int);
        GPGPU.cl_zero_buffer(GPGPU.ptr_render_queue, ptr_mesh_transfer, ELEMENT_BUFFER_SIZE * 2);

        long[] hull_count = arg_long(GPGPU.core_memory.next_hull());

        long si = Editor.ACTIVE ? System.nanoTime() : 0;
        k_count_mesh_instances.call(hull_count);
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - si;
            Editor.queue_event("render_model_count_meshes", String.valueOf(e));
        }

        scan_int_out(ptr_counters, ptr_offsets, mesh_count);

        int total_instances = GPGPU.cl_read_svm_int(GPGPU.ptr_render_queue, svm_total);
        if (total_instances == 0)
        {
            return null;
        }

        if (Editor.ACTIVE)
        {
            Editor.queue_event("render_instance_count", String.valueOf(total_instances));
        }

        long details_size = (long)total_instances * CLSize.cl_int4;
        long texture_size = (long)total_instances * CLSize.cl_int;
        var mesh_details_ptr = GPGPU.new_empty_buffer(GPGPU.ptr_render_queue, details_size);
        var mesh_texture_ptr = GPGPU.new_empty_buffer(GPGPU.ptr_render_queue, texture_size);

        si = Editor.ACTIVE ? System.nanoTime() : 0;
        k_write_mesh_details
            .ptr_arg(WriteMeshDetails_k.Args.mesh_details, mesh_details_ptr)
            .ptr_arg(WriteMeshDetails_k.Args.mesh_texture, mesh_texture_ptr)
            .call(hull_count);
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - si;
            Editor.queue_event("render_model_write_details", String.valueOf(e));
        }

        si = Editor.ACTIVE ? System.nanoTime() : 0;
        k_count_mesh_batches
            .ptr_arg(CountMeshBatches_k.Args.mesh_details, mesh_details_ptr)
            .set_arg(CountMeshBatches_k.Args.count, total_instances)
            .call(GPGPU.global_single_size);
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - si;
            Editor.queue_event("render_model_count_batches", String.valueOf(e));
        }

        int total_batches = GPGPU.cl_read_svm_int(GPGPU.ptr_render_queue, svm_total);
        if (Editor.ACTIVE)
        {
            Editor.queue_event("render_batch_count", String.valueOf(total_batches));
        }
        long batch_index_size = (long) total_batches * CLSize.cl_int;

        var mesh_offset_ptr = GPGPU.cl_new_pinned_buffer(batch_index_size);

        si = Editor.ACTIVE ? System.nanoTime() : 0;
        k_calculate_batch_offsets
            .ptr_arg(CalculateBatchOffsets_k.Args.mesh_offsets, mesh_offset_ptr)
            .ptr_arg(CalculateBatchOffsets_k.Args.mesh_details, mesh_details_ptr)
            .set_arg(CalculateBatchOffsets_k.Args.count, total_instances)
            .call(GPGPU.global_single_size);
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - si;
            Editor.queue_event("render_model_batch_offsets", String.valueOf(e));
        }

        si = Editor.ACTIVE ? System.nanoTime() : 0;
        int[] raw_offsets = GPGPU.cl_read_pinned_int_buffer(GPGPU.ptr_render_queue, mesh_offset_ptr, CLSize.cl_int, total_batches);
        GPGPU.cl_release_buffer(mesh_offset_ptr);
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - si;
            Editor.queue_event("render_buffer_read", String.valueOf(e));
        }
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("render_cl_cycle", String.valueOf(e));
        }
        return new BatchData(raw_offsets, total_instances, mesh_details_ptr, mesh_texture_ptr);
    }

    private void tick_GL(BatchData batch_data)
    {
        glBindVertexArray(vao);
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, cbo);

        shader.use();
        for (int i  = 0; i < textures.length; i++)
        {
            textures[i].bind(i);
        }

        var control_components = ecs.get_components(Component.ControlPoints);
        ControlPoints control_points = null;
        for (Map.Entry<String, GameComponent> entry : control_components.entrySet())
        {
            GameComponent component = entry.getValue();
            control_points = Component.ControlPoints.coerce(component);
        }

        assert control_points != null : "Component was null";
        Objects.requireNonNull(control_points);

        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());
        shader.uploadIntArray("uTextures", texture_slots);
        shader.uploadvec2f("uMouse", control_points.get_world_target());
        shader.uploadvec2f("uCamera", uniformGrid.getWorld_position());

        long si = Editor.ACTIVE ? System.nanoTime() : 0;
        for (int current_batch = 0; current_batch < batch_data.raw_offsets.length; current_batch++)
        {
            int next_batch = current_batch + 1;
            int offset = batch_data.raw_offsets[current_batch];
            int count = next_batch == batch_data.raw_offsets.length
                ? batch_data.total_instances - offset
                : batch_data.raw_offsets[next_batch] - offset;

            long st = Editor.ACTIVE ? System.nanoTime() : 0;

            k_transfer_detail_data
                .ptr_arg(TransferDetailData_k.Args.mesh_details, batch_data.mesh_details_ptr)
                .set_arg(TransferDetailData_k.Args.offset, offset)
                .call(arg_long(count));

            if (Editor.ACTIVE)
            {
                long e = System.nanoTime() - st;
                Editor.queue_event("render_detail_transfer", String.valueOf(e));
            }

            scan_int2(ptr_mesh_transfer, count);

            st = Editor.ACTIVE ? System.nanoTime() : 0;

            k_transfer_render_data
                .share_mem(ptr_command_buffer)
                .share_mem(ptr_element_buffer)
                .share_mem(ptr_vertex_buffer)
                .share_mem(ptr_uv_buffer)
                .share_mem(ptr_color_buffer)
                .share_mem(ptr_slot_buffer)
                .ptr_arg(TransferRenderData_k.Args.mesh_details, batch_data.mesh_details_ptr)
                .ptr_arg(TransferRenderData_k.Args.mesh_texture, batch_data.mesh_texture_ptr)
                .set_arg(TransferRenderData_k.Args.offset, offset)
                .call(arg_long(count));

            if (Editor.ACTIVE)
            {
                long e = System.nanoTime() - st;
                Editor.queue_event("render_data_transfer", String.valueOf(e));
            }

            glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, 0, count, 0);
        }

        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - si;
            Editor.queue_event("render_model_batch_loop", String.valueOf(e));
        }

        shader.detach();
        glBindVertexArray(0);
    }

    @Override
    public void tick(float dt)
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;

        var batch_data = tick_CL();
        if (batch_data == null) return;

        tick_GL(batch_data);

        GPGPU.cl_release_buffer(batch_data.mesh_details_ptr);
        GPGPU.cl_release_buffer(batch_data.mesh_texture_ptr);

        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("render_model", String.valueOf(e));
        }
    }

    private void scan_int(long data_ptr, int n)
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;

        int k = GPGPU.work_group_count(n);
        if (k == 1)
        {
            scan_single_block_int(data_ptr, n);
        }
        else
        {
            scan_multi_block_int(data_ptr, n, k);
        }

        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("render_model_scan_int", String.valueOf(e));
        }
    }

    private void scan_int_out(long data_ptr, long o_data_ptr, int n)
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;

        int k = GPGPU.work_group_count(n);
        if (k == 1)
        {
            scan_single_block_int_out(data_ptr, o_data_ptr, n);
        }
        else
        {
            scan_multi_block_int_out(data_ptr, o_data_ptr, n, k);
        }

        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("render_model_scan_int_out", String.valueOf(e));
        }
    }

    private void scan_int2(long data_ptr, int n)
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;

        int k = GPGPU.work_group_count(n);
        if (k == 1)
        {
            scan_single_block_int2(data_ptr, n);
        }
        else
        {
            scan_multi_block_int2(data_ptr, n, k);
        }

        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("render_model_scan_int2", String.valueOf(e));
        }
    }

    private void scan_single_block_int(long data_ptr, int n)
    {
        long local_buffer_size = CLSize.cl_int * GPGPU.max_scan_block_size;

        k_scan_int_single_block
            .ptr_arg(ScanIntSingleBlock_k.Args.data, data_ptr)
            .loc_arg(ScanIntSingleBlock_k.Args.buffer, local_buffer_size)
            .set_arg(ScanIntSingleBlock_k.Args.n, n)
            .call(GPGPU.local_work_default, GPGPU.local_work_default);
    }

    private void scan_multi_block_int(long data_ptr, int n, int k)
    {
        long local_buffer_size = CLSize.cl_int * GPGPU.max_scan_block_size;
        long gx = k * GPGPU.max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) CLSize.cl_int * ((long) part_size));

        var part_data = GPGPU.cl_new_buffer(part_buf_size);

        k_scan_int_multi_block
            .ptr_arg(ScanIntMultiBlock_k.Args.data, data_ptr)
            .loc_arg(ScanIntMultiBlock_k.Args.buffer, local_buffer_size)
            .ptr_arg(ScanIntMultiBlock_k.Args.part, part_data)
            .set_arg(ScanIntMultiBlock_k.Args.n, n)
            .call(global_work_size, GPGPU.local_work_default);

        scan_int(part_data, part_size);

        k_complete_int_multi_block
            .ptr_arg(CompleteIntMultiBlock_k.Args.data, data_ptr)
            .loc_arg(CompleteIntMultiBlock_k.Args.buffer, local_buffer_size)
            .ptr_arg(CompleteIntMultiBlock_k.Args.part, part_data)
            .set_arg(CompleteIntMultiBlock_k.Args.n, n)
            .call(global_work_size, GPGPU.local_work_default);

        GPGPU.cl_release_buffer(part_data);
    }

    private void scan_single_block_int_out(long data_ptr, long o_data_ptr, int n)
    {
        long local_buffer_size = CLSize.cl_int * GPGPU.max_scan_block_size;

        k_scan_int_single_block_out
            .ptr_arg(ScanIntSingleBlockOut_k.Args.input, data_ptr)
            .ptr_arg(ScanIntSingleBlockOut_k.Args.output, o_data_ptr)
            .loc_arg(ScanIntSingleBlockOut_k.Args.buffer, local_buffer_size)
            .set_arg(ScanIntSingleBlockOut_k.Args.n, n)
            .call(GPGPU.local_work_default, GPGPU.local_work_default);
    }

    private void scan_multi_block_int_out(long data_ptr, long o_data_ptr, int n, int k)
    {
        long local_buffer_size = CLSize.cl_int * GPGPU.max_scan_block_size;
        long gx = k * GPGPU.max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) CLSize.cl_int * ((long) part_size));
        var part_data = GPGPU.cl_new_buffer(part_buf_size);

        k_scan_int_multi_block_out
            .ptr_arg(ScanIntMultiBlockOut_k.Args.input, data_ptr)
            .ptr_arg(ScanIntMultiBlockOut_k.Args.output, o_data_ptr)
            .loc_arg(ScanIntMultiBlockOut_k.Args.buffer, local_buffer_size)
            .ptr_arg(ScanIntMultiBlockOut_k.Args.part, part_data)
            .set_arg(ScanIntMultiBlockOut_k.Args.n, n)
            .call(global_work_size, GPGPU.local_work_default);

        scan_int(part_data, part_size);

        k_complete_int_multi_block_out
            .ptr_arg(CompleteIntMultiBlockOut_k.Args.output, o_data_ptr)
            .loc_arg(CompleteIntMultiBlockOut_k.Args.buffer, local_buffer_size)
            .ptr_arg(CompleteIntMultiBlockOut_k.Args.part, part_data)
            .set_arg(CompleteIntMultiBlockOut_k.Args.n, n)
            .call(global_work_size, GPGPU.local_work_default);

        GPGPU.cl_release_buffer(part_data);
    }

    private void scan_single_block_int2(long data_ptr, int n)
    {
        long local_buffer_size = CLSize.cl_int2 * GPGPU.max_scan_block_size;

        k_scan_int2_single_block
            .ptr_arg(ScanInt2SingleBlock_k.Args.data, data_ptr)
            .loc_arg(ScanInt2SingleBlock_k.Args.buffer, local_buffer_size)
            .set_arg(ScanInt2SingleBlock_k.Args.n, n)
            .call(GPGPU.local_work_default, GPGPU.local_work_default);
    }

    private void scan_multi_block_int2(long data_ptr, int n, int k)
    {
        long local_buffer_size = CLSize.cl_int2 * GPGPU.max_scan_block_size;
        long gx = k * GPGPU.max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) CLSize.cl_int2 * ((long) part_size));

        var part_data = GPGPU.cl_new_buffer(part_buf_size);

        k_scan_int2_multi_block
            .ptr_arg(ScanInt2MultiBlock_k.Args.data, data_ptr)
            .loc_arg(ScanInt2MultiBlock_k.Args.buffer, local_buffer_size)
            .ptr_arg(ScanInt2MultiBlock_k.Args.part, part_data)
            .set_arg(ScanInt2MultiBlock_k.Args.n, n)
            .call(global_work_size, GPGPU.local_work_default);

        scan_int2(part_data, part_size);

        k_complete_int2_multi_block
            .ptr_arg(CompleteInt2MultiBlock_k.Args.data, data_ptr)
            .loc_arg(CompleteInt2MultiBlock_k.Args.buffer, local_buffer_size)
            .ptr_arg(CompleteInt2MultiBlock_k.Args.part, part_data)
            .set_arg(CompleteInt2MultiBlock_k.Args.n, n)
            .call(global_work_size, GPGPU.local_work_default);

        GPGPU.cl_release_buffer(part_data);
    }

    @Override
    public void shutdown()
    {
        glDeleteVertexArrays(vao);
        glDeleteBuffers(cbo);
        glDeleteBuffers(ebo);
        glDeleteBuffers(vbo_position);
        glDeleteBuffers(vbo_texture_uv);
        glDeleteBuffers(vbo_color);
        glDeleteBuffers(vbo_texture_slot);

        shader.destroy();
        p_mesh_query.destroy();
        for (var t : textures){ t.destroy(); }
        GPGPU.cl_release_buffer(ptr_element_buffer);
        GPGPU.cl_release_buffer(ptr_vertex_buffer);
        GPGPU.cl_release_buffer(ptr_command_buffer);
        GPGPU.cl_release_buffer(ptr_uv_buffer);
        GPGPU.cl_release_buffer(svm_total);
        GPGPU.cl_release_buffer(ptr_query);
        GPGPU.cl_release_buffer(ptr_counters);
        GPGPU.cl_release_buffer(ptr_offsets);
        GPGPU.cl_release_buffer(ptr_mesh_transfer);
    }
}
