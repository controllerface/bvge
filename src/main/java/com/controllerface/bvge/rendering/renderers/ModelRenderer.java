package com.controllerface.bvge.rendering.renderers;

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
import com.controllerface.bvge.gpu.cl.GPUScanScalarIntOut;
import com.controllerface.bvge.gpu.cl.GPUScanVectorInt2;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.Kernel;
import com.controllerface.bvge.gpu.cl.kernels.rendering.*;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.gpu.cl.programs.MeshQuery;
import com.controllerface.bvge.gpu.cl.programs.ScanIntArrayOut;
import com.controllerface.bvge.gpu.gl.GLUtils;
import com.controllerface.bvge.gpu.gl.buffers.GL_CommandBuffer;
import com.controllerface.bvge.gpu.gl.buffers.GL_ElementBuffer;
import com.controllerface.bvge.gpu.gl.buffers.GL_VertexArray;
import com.controllerface.bvge.gpu.gl.buffers.GL_VertexBuffer;
import com.controllerface.bvge.gpu.gl.shaders.GL_Shader;
import com.controllerface.bvge.gpu.gl.shaders.GL_ShaderType;
import com.controllerface.bvge.gpu.gl.textures.GL_Texture2D;
import com.controllerface.bvge.memory.types.ReferenceBufferType;
import com.controllerface.bvge.memory.types.RenderBufferType;
import com.controllerface.bvge.models.geometry.Model;
import com.controllerface.bvge.models.geometry.ModelRegistry;
import com.controllerface.bvge.physics.UniformGrid;

import java.util.LinkedHashSet;
import java.util.Objects;

import static com.controllerface.bvge.game.Constants.Rendering.*;
import static com.controllerface.bvge.gpu.cl.CLUtils.arg_long;
import static org.lwjgl.opengl.GL45C.*;

public class ModelRenderer extends GameSystem
{
    private final UniformGrid uniformGrid;

    private static final int ELEMENT_BUFFER_SIZE = MAX_BATCH_SIZE * Integer.BYTES;
    private static final int TEXTURE_BUFFER_SIZE = MAX_BATCH_SIZE * SCALAR_FLOAT_SIZE;
    private static final int COMMAND_BUFFER_SIZE = MAX_BATCH_SIZE * Integer.BYTES * 5;
    private static final int VERTEX_BUFFER_SIZE  = MAX_BATCH_SIZE * VECTOR_FLOAT_4D_SIZE;
    private static final int UV_BUFFER_SIZE      = MAX_BATCH_SIZE * VECTOR_FLOAT_2D_SIZE;
    private static final int COLOR_BUFFER_SIZE   = MAX_BATCH_SIZE * VECTOR_FLOAT_4D_SIZE;
    private static final int POSITION_ATTRIBUTE  = 0;
    private static final int UV_COORD_ATTRIBUTE  = 1;
    private static final int COLOR_ATTRIBUTE     = 2;
    private static final int TEXTURE_ATTRIBUTE   = 3;

    // todo: base this off of number of texture units
    private final int[] texture_slots = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    private final GPUProgram p_mesh_query = new MeshQuery();
    private final GPUProgram p_scan_int_array_out = new ScanIntArrayOut();

    private GPUScanScalarIntOut gpu_int_scan_out;
    private GPUScanVectorInt2 gpu_int2_scan;

    private GL_VertexArray vao;
    private GL_ElementBuffer ebo;
    private GL_CommandBuffer cbo;
    private GL_VertexBuffer vbo_position;
    private GL_VertexBuffer vbo_texture_uv;
    private GL_VertexBuffer vbo_color;
    private GL_VertexBuffer vbo_texture_slot;

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
    private long svm_total;

    private int[] raw_query;
    private int mesh_count;
    private long mesh_size;

    private GL_Texture2D[] textures;
    private GL_Shader shader;

    private GPUKernel k_count_mesh_instances;
    private GPUKernel k_write_mesh_details;
    private GPUKernel k_count_mesh_batches;
    private GPUKernel k_calculate_batch_offsets;
    private GPUKernel k_transfer_detail_data;
    private GPUKernel k_transfer_render_data;

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
        shader = GPU.GL.new_shader(shader_file, GL_ShaderType.TWO_STAGE);

        Model[] models = new Model[model_ids.length];

        for (int i = 0; i < model_ids.length; i++)
        {
            var model = ModelRegistry.get_model_by_index(model_ids[i]);
            models[i] = model;
            mesh_count += model.meshes().length;
        }

        mesh_size = (long)mesh_count * CL_DataTypes.cl_int.size();
        raw_query = new int[mesh_count * 2]; // int2

        var texture_buffer = new LinkedHashSet<GL_Texture2D>();

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

        textures = new GL_Texture2D[texture_buffer.size()];
        int texture_index = 0;
        for (var texture : texture_buffer)
        {
            textures[texture_index++] = texture;
        }

        vao = GPU.GL.new_vao();
        ebo = GPU.GL.dynamic_element_buffer(vao, ELEMENT_BUFFER_SIZE);
        cbo = GPU.GL.dynamic_command_buffer(vao, COMMAND_BUFFER_SIZE);

        vbo_position     = GPU.GL.new_buffer_vec4(vao, POSITION_ATTRIBUTE, VERTEX_BUFFER_SIZE);
        vbo_texture_uv   = GPU.GL.new_buffer_vec2(vao, UV_COORD_ATTRIBUTE, UV_BUFFER_SIZE);
        vbo_color        = GPU.GL.new_buffer_vec4(vao, COLOR_ATTRIBUTE, COLOR_BUFFER_SIZE);
        vbo_texture_slot = GPU.GL.new_buffer_float(vao, TEXTURE_ATTRIBUTE, TEXTURE_BUFFER_SIZE);

        vao.enable_attribute(POSITION_ATTRIBUTE);
        vao.enable_attribute(UV_COORD_ATTRIBUTE);
        vao.enable_attribute(COLOR_ATTRIBUTE);
        vao.enable_attribute(TEXTURE_ATTRIBUTE);
    }

    private void init_CL()
    {
        ptr_command_buffer = GPGPU.share_memory(cbo.id());
        ptr_element_buffer = GPGPU.share_memory(ebo.id());
        ptr_vertex_buffer  = GPGPU.share_memory(vbo_position.id());
        ptr_uv_buffer      = GPGPU.share_memory(vbo_texture_uv.id());
        ptr_color_buffer   = GPGPU.share_memory(vbo_color.id());
        ptr_slot_buffer    = GPGPU.share_memory(vbo_texture_slot.id());
        svm_total          = GPGPU.cl_new_pinned_int();
        ptr_query          = GPGPU.new_mutable_buffer(raw_query);
        ptr_counters       = GPGPU.new_empty_buffer(GPGPU.ptr_render_queue, mesh_size);
        ptr_offsets        = GPGPU.new_empty_buffer(GPGPU.ptr_render_queue, mesh_size);
        ptr_mesh_transfer  = GPGPU.new_empty_buffer(GPGPU.ptr_render_queue, ELEMENT_BUFFER_SIZE * 2);

        p_mesh_query.init();
        p_scan_int_array_out.init();

        gpu_int2_scan    = new GPUScanVectorInt2(GPGPU.ptr_render_queue);
        gpu_int_scan_out = new GPUScanScalarIntOut(GPGPU.ptr_render_queue);

        long k_ptr_count_instances = p_mesh_query.kernel_ptr(Kernel.count_mesh_instances);
        k_count_mesh_instances = new CountMeshInstances_k(GPGPU.ptr_render_queue, k_ptr_count_instances)
            .ptr_arg(CountMeshInstances_k.Args.counters, ptr_counters)
            .ptr_arg(CountMeshInstances_k.Args.query, ptr_query)
            .ptr_arg(CountMeshInstances_k.Args.total, svm_total)
            .set_arg(CountMeshInstances_k.Args.count, mesh_count)
            .buf_arg(CountMeshInstances_k.Args.hull_mesh_ids, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL_MESH_ID))
            .buf_arg(CountMeshInstances_k.Args.hull_flags, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL_FLAG))
            .buf_arg(CountMeshInstances_k.Args.hull_entity_ids, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL_ENTITY_ID))
            .buf_arg(CountMeshInstances_k.Args.entity_flags, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_ENTITY_FLAG));

        long k_ptr_write_details = p_mesh_query.kernel_ptr(Kernel.write_mesh_details);
        k_write_mesh_details = new WriteMeshDetails_k(GPGPU.ptr_render_queue, k_ptr_write_details)
            .ptr_arg(WriteMeshDetails_k.Args.counters, ptr_counters)
            .ptr_arg(WriteMeshDetails_k.Args.query, ptr_query)
            .ptr_arg(WriteMeshDetails_k.Args.offsets, ptr_offsets)
            .set_arg(WriteMeshDetails_k.Args.count, mesh_count)
            .buf_arg(WriteMeshDetails_k.Args.hull_mesh_ids, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL_MESH_ID))
            .buf_arg(WriteMeshDetails_k.Args.hull_flags, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL_FLAG))
            .buf_arg(WriteMeshDetails_k.Args.hull_entity_ids, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL_ENTITY_ID))
            .buf_arg(WriteMeshDetails_k.Args.entity_flags, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_ENTITY_FLAG))
            .buf_arg(WriteMeshDetails_k.Args.mesh_vertex_tables, GPGPU.core_memory.get_buffer(ReferenceBufferType.MESH_VERTEX_TABLE))
            .buf_arg(WriteMeshDetails_k.Args.mesh_face_tables, GPGPU.core_memory.get_buffer(ReferenceBufferType.MESH_FACE_TABLE));

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
            .buf_arg(TransferRenderData_k.Args.hull_point_tables, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL_POINT_TABLE))
            .buf_arg(TransferRenderData_k.Args.hull_mesh_ids, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL_MESH_ID))
            .buf_arg(TransferRenderData_k.Args.hull_entity_ids, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL_ENTITY_ID))
            .buf_arg(TransferRenderData_k.Args.hull_flags, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL_FLAG))
            .buf_arg(TransferRenderData_k.Args.hull_uv_offsets, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL_UV_OFFSET))
            .buf_arg(TransferRenderData_k.Args.hull_integrity, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL_INTEGRITY))
            .buf_arg(TransferRenderData_k.Args.entity_flags, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_ENTITY_FLAG))
            .buf_arg(TransferRenderData_k.Args.mesh_vertex_tables, GPGPU.core_memory.get_buffer(ReferenceBufferType.MESH_VERTEX_TABLE))
            .buf_arg(TransferRenderData_k.Args.mesh_face_tables, GPGPU.core_memory.get_buffer(ReferenceBufferType.MESH_FACE_TABLE))
            .buf_arg(TransferRenderData_k.Args.mesh_faces, GPGPU.core_memory.get_buffer(ReferenceBufferType.MESH_FACE))
            .buf_arg(TransferRenderData_k.Args.points, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_POINT))
            .buf_arg(TransferRenderData_k.Args.point_hit_counts, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_POINT_HIT_COUNT))
            .buf_arg(TransferRenderData_k.Args.point_vertex_references, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_POINT_VERTEX_REFERENCE))
            .buf_arg(TransferRenderData_k.Args.uv_tables, GPGPU.core_memory.get_buffer(ReferenceBufferType.VERTEX_UV_TABLE))
            .buf_arg(TransferRenderData_k.Args.texture_uvs, GPGPU.core_memory.get_buffer(ReferenceBufferType.VERTEX_TEXTURE_UV));
    }

    private record BatchData( int[] raw_offsets, int total_instances, long mesh_details_ptr, long mesh_texture_ptr) { }

    private BatchData tick_CL()
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;

        GPGPU.cl_zero_buffer(GPGPU.ptr_render_queue, ptr_counters, mesh_size);
        GPGPU.cl_zero_buffer(GPGPU.ptr_render_queue, ptr_offsets, mesh_size);
        GPGPU.cl_zero_buffer(GPGPU.ptr_render_queue, svm_total, CL_DataTypes.cl_int.size());
        GPGPU.cl_zero_buffer(GPGPU.ptr_render_queue, ptr_mesh_transfer, ELEMENT_BUFFER_SIZE * 2);

        int hull_count = GPGPU.core_memory.sector_container().next_hull();
        int hull_size = GPGPU.calculate_preferred_global_size(hull_count);
        long[] hull_global_size = arg_long(hull_size);

        long si = Editor.ACTIVE ? System.nanoTime() : 0;
        k_count_mesh_instances
            .set_arg(CountMeshInstances_k.Args.max_hull, hull_count)
            .call(hull_global_size, GPGPU.preferred_work_size);
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - si;
            Editor.queue_event("render_model_count_meshes", String.valueOf(e));
        }

        gpu_int_scan_out.scan_int_out(ptr_counters, ptr_offsets, mesh_count);

        int total_instances = GPGPU.cl_read_pinned_int(GPGPU.ptr_render_queue, svm_total);
        if (total_instances == 0)
        {
            return null;
        }

        if (Editor.ACTIVE)
        {
            Editor.queue_event("render_instance_count", String.valueOf(total_instances));
        }

        long details_size = (long)total_instances * CL_DataTypes.cl_int4.size();
        long texture_size = (long)total_instances * CL_DataTypes.cl_int.size();
        var mesh_details_ptr = GPGPU.new_empty_buffer(GPGPU.ptr_render_queue, details_size);
        var mesh_texture_ptr = GPGPU.new_empty_buffer(GPGPU.ptr_render_queue, texture_size);

        si = Editor.ACTIVE ? System.nanoTime() : 0;
        k_write_mesh_details
            .ptr_arg(WriteMeshDetails_k.Args.mesh_details, mesh_details_ptr)
            .ptr_arg(WriteMeshDetails_k.Args.mesh_texture, mesh_texture_ptr)
            .set_arg(WriteMeshDetails_k.Args.max_hull, hull_count)
            .call(hull_global_size, GPGPU.preferred_work_size);
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - si;
            Editor.queue_event("render_model_write_details", String.valueOf(e));
        }

        si = Editor.ACTIVE ? System.nanoTime() : 0;
        k_count_mesh_batches
            .ptr_arg(CountMeshBatches_k.Args.mesh_details, mesh_details_ptr)
            .set_arg(CountMeshBatches_k.Args.count, total_instances)
            .call_task();
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - si;
            Editor.queue_event("render_model_count_batches", String.valueOf(e));
        }

        int total_batches = GPGPU.cl_read_pinned_int(GPGPU.ptr_render_queue, svm_total);
        if (Editor.ACTIVE)
        {
            Editor.queue_event("render_batch_count", String.valueOf(total_batches));
        }
        long batch_index_size = (long) total_batches * CL_DataTypes.cl_int.size();

        var mesh_offset_ptr = GPGPU.cl_new_pinned_buffer(batch_index_size);

        si = Editor.ACTIVE ? System.nanoTime() : 0;
        k_calculate_batch_offsets
            .ptr_arg(CalculateBatchOffsets_k.Args.mesh_offsets, mesh_offset_ptr)
            .ptr_arg(CalculateBatchOffsets_k.Args.mesh_details, mesh_details_ptr)
            .set_arg(CalculateBatchOffsets_k.Args.count, total_instances)
            .call_task();
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - si;
            Editor.queue_event("render_model_batch_offsets", String.valueOf(e));
        }

        si = Editor.ACTIVE ? System.nanoTime() : 0;
        int[] raw_offsets = GPGPU.cl_read_pinned_int_buffer(GPGPU.ptr_render_queue, mesh_offset_ptr, CL_DataTypes.cl_int.size(), total_batches);
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
        vao.bind();
        cbo.bind();

        shader.use();
        for (int i  = 0; i < textures.length; i++)
        {
            textures[i].bind(i);
        }

        PlayerInput player_input = ComponentType.InputState.forEntity(ecs, Constants.PLAYER_ID);
        assert player_input != null : "Component was null";
        Objects.requireNonNull(player_input);

        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());
        shader.uploadIntArray("uTextures", texture_slots);
        shader.uploadvec2f("uMouse", player_input.get_world_target());
        shader.uploadvec2f("uCamera", uniformGrid.getWorld_position());

        long si = Editor.ACTIVE ? System.nanoTime() : 0;
        for (int current_batch = 0; current_batch < batch_data.raw_offsets.length; current_batch++)
        {
            int next_batch = current_batch + 1;
            int offset = batch_data.raw_offsets[current_batch];
            int count = next_batch == batch_data.raw_offsets.length
                ? batch_data.total_instances - offset
                : batch_data.raw_offsets[next_batch] - offset;
            int count_size = GPGPU.calculate_preferred_global_size(count);
            long[] count_global_size = arg_long(count_size);
            long st = Editor.ACTIVE ? System.nanoTime() : 0;

            k_transfer_detail_data
                .ptr_arg(TransferDetailData_k.Args.mesh_details, batch_data.mesh_details_ptr)
                .set_arg(TransferDetailData_k.Args.offset, offset)
                .set_arg(TransferDetailData_k.Args.max_mesh, count)
                .call(count_global_size, GPGPU.preferred_work_size);

            if (Editor.ACTIVE)
            {
                long e = System.nanoTime() - st;
                Editor.queue_event("render_detail_transfer", String.valueOf(e));
            }

            gpu_int2_scan.scan_int2(ptr_mesh_transfer, count);

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
                .set_arg(TransferRenderData_k.Args.max_index, count)
                .call(count_global_size, GPGPU.preferred_work_size);

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
        vao.unbind();
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

    @Override
    public void shutdown()
    {
        vao.release();
        cbo.release();
        ebo.release();
        vbo_position.release();
        vbo_texture_uv.release();
        vbo_color.release();
        vbo_texture_slot.release();

        gpu_int_scan_out.release();
        gpu_int2_scan.release();

        shader.release();
        p_mesh_query.release();
        for (var t : textures){ t.release(); }
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
