package com.controllerface.bvge.rendering.renderers;

import com.controllerface.bvge.core.Window;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.ComponentType;
import com.controllerface.bvge.editor.Editor;
import com.controllerface.bvge.game.Constants;
import com.controllerface.bvge.game.PlayerInput;
import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.rendering.*;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.gpu.cl.programs.rendering.MeshQuery;
import com.controllerface.bvge.gpu.cl.programs.scan.GPUScanScalarIntOut;
import com.controllerface.bvge.gpu.cl.programs.scan.GPUScanVectorInt2;
import com.controllerface.bvge.gpu.cl.programs.scan.ScanIntArrayOut;
import com.controllerface.bvge.gpu.gl.buffers.GL_CommandBuffer;
import com.controllerface.bvge.gpu.gl.buffers.GL_ElementBuffer;
import com.controllerface.bvge.gpu.gl.buffers.GL_VertexArray;
import com.controllerface.bvge.gpu.gl.buffers.GL_VertexBuffer;
import com.controllerface.bvge.gpu.gl.shaders.GL_Shader;
import com.controllerface.bvge.gpu.gl.shaders.GL_ShaderType;
import com.controllerface.bvge.gpu.gl.textures.GL_Texture2D;
import com.controllerface.bvge.models.geometry.Model;
import com.controllerface.bvge.models.geometry.ModelRegistry;
import com.controllerface.bvge.physics.UniformGrid;
import com.controllerface.bvge.rendering.Renderer;

import java.util.LinkedHashSet;
import java.util.Objects;

import static com.controllerface.bvge.game.Constants.Rendering.*;
import static com.controllerface.bvge.gpu.GPU.CL.arg_long;
import static org.lwjgl.opengl.GL45C.*;

public class ModelRenderer implements Renderer
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

    private CL_Buffer command_buf;
    private CL_Buffer element_buf;
    private CL_Buffer vertex_buf;
    private CL_Buffer uv_buf;
    private CL_Buffer color_buf;
    private CL_Buffer slot_buf;
    private CL_Buffer query_buf;
    private CL_Buffer counter_buf;
    private CL_Buffer offset_buf;
    private CL_Buffer mesh_transfer_buf;
    private CL_Buffer total_buf;

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
    private final ECS ecs;

    public ModelRenderer(ECS ecs, UniformGrid uniformGrid, int ... model_ids)
    {
        this.ecs = ecs;
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
        command_buf        = GPU.CL.gl_share_memory(GPU.compute.context, cbo);
        element_buf        = GPU.CL.gl_share_memory(GPU.compute.context, ebo);
        vertex_buf         = GPU.CL.gl_share_memory(GPU.compute.context, vbo_position);
        uv_buf             = GPU.CL.gl_share_memory(GPU.compute.context, vbo_texture_uv);
        color_buf          = GPU.CL.gl_share_memory(GPU.compute.context, vbo_color);
        slot_buf           = GPU.CL.gl_share_memory(GPU.compute.context, vbo_texture_slot);
        total_buf          = GPU.CL.new_pinned_int(GPU.compute.context);
        query_buf          = GPU.CL.new_mutable_buffer(GPU.compute.context, raw_query);
        counter_buf        = GPU.CL.new_empty_buffer(GPU.compute.context, GPU.compute.render_queue, mesh_size);
        offset_buf         = GPU.CL.new_empty_buffer(GPU.compute.context, GPU.compute.render_queue, mesh_size);
        mesh_transfer_buf  = GPU.CL.new_empty_buffer(GPU.compute.context, GPU.compute.render_queue, ELEMENT_BUFFER_SIZE * 2);

        p_mesh_query.init();
        p_scan_int_array_out.init();

        gpu_int2_scan    = new GPUScanVectorInt2(GPU.compute.render_queue);
        gpu_int_scan_out = new GPUScanScalarIntOut(GPU.compute.render_queue);

        k_calculate_batch_offsets = new CalculateBatchOffsets_k(GPU.compute.render_queue, p_mesh_query);

        k_count_mesh_instances = new CountMeshInstances_k(GPU.compute.render_queue, p_mesh_query)
            .init(counter_buf, query_buf, total_buf, mesh_count);

        k_write_mesh_details = new WriteMeshDetails_k(GPU.compute.render_queue, p_mesh_query)
            .init(counter_buf, query_buf, offset_buf, mesh_count);

        k_count_mesh_batches = new CountMeshBatches_k(GPU.compute.render_queue, p_mesh_query)
            .init(total_buf, MAX_BATCH_SIZE);

        k_transfer_detail_data = new TransferDetailData_k(GPU.compute.render_queue, p_mesh_query)
            .init(mesh_transfer_buf);

        k_transfer_render_data = new TransferRenderData_k(GPU.compute.render_queue, p_mesh_query)
            .init(command_buf, element_buf, vertex_buf, uv_buf, color_buf, slot_buf, mesh_transfer_buf);
    }

    private record BatchData(int[] raw_offsets, int total_instances, CL_Buffer mesh_details_buf, CL_Buffer mesh_texture_buf) { }

    private BatchData tick_CL()
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;

        GPU.CL.zero_buffer(GPU.compute.render_queue, counter_buf, mesh_size);
        GPU.CL.zero_buffer(GPU.compute.render_queue, offset_buf, mesh_size);
        GPU.CL.zero_buffer(GPU.compute.render_queue, total_buf, CL_DataTypes.cl_int.size());
        GPU.CL.zero_buffer(GPU.compute.render_queue, mesh_transfer_buf, ELEMENT_BUFFER_SIZE * 2);

        int hull_count = GPU.memory.sector_container().next_hull();
        int hull_size = GPU.compute.calculate_preferred_global_size(hull_count);
        long[] hull_global_size = arg_long(hull_size);

        long si = Editor.ACTIVE ? System.nanoTime() : 0;
        k_count_mesh_instances
            .set_arg(CountMeshInstances_k.Args.max_hull, hull_count)
            .call(hull_global_size, GPU.compute.preferred_work_size);

        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - si;
            Editor.queue_event("render_model_count_meshes", String.valueOf(e));
        }

        gpu_int_scan_out.scan_int_out(counter_buf.ptr(), offset_buf.ptr(), mesh_count);

        int total_instances = GPU.CL.read_pinned_int(GPU.compute.render_queue, total_buf);
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
        var mesh_details_ptr = GPU.CL.new_empty_buffer(GPU.compute.context, GPU.compute.render_queue, details_size);
        var mesh_texture_ptr = GPU.CL.new_empty_buffer(GPU.compute.context, GPU.compute.render_queue, texture_size);

        si = Editor.ACTIVE ? System.nanoTime() : 0;
        k_write_mesh_details
            .buf_arg(WriteMeshDetails_k.Args.mesh_details, mesh_details_ptr)
            .buf_arg(WriteMeshDetails_k.Args.mesh_texture, mesh_texture_ptr)
            .set_arg(WriteMeshDetails_k.Args.max_hull, hull_count)
            .call(hull_global_size, GPU.compute.preferred_work_size);
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - si;
            Editor.queue_event("render_model_write_details", String.valueOf(e));
        }

        si = Editor.ACTIVE ? System.nanoTime() : 0;
        k_count_mesh_batches
            .buf_arg(CountMeshBatches_k.Args.mesh_details, mesh_details_ptr)
            .set_arg(CountMeshBatches_k.Args.count, total_instances)
            .call_task();
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - si;
            Editor.queue_event("render_model_count_batches", String.valueOf(e));
        }

        int total_batches = GPU.CL.read_pinned_int(GPU.compute.render_queue, total_buf);
        if (Editor.ACTIVE)
        {
            Editor.queue_event("render_batch_count", String.valueOf(total_batches));
        }
        long batch_index_size = (long) total_batches * CL_DataTypes.cl_int.size();

        var mesh_offset_buf = GPU.CL.new_pinned_buffer(GPU.compute.context, batch_index_size);

        si = Editor.ACTIVE ? System.nanoTime() : 0;
        k_calculate_batch_offsets
            .buf_arg(CalculateBatchOffsets_k.Args.mesh_offsets, mesh_offset_buf)
            .buf_arg(CalculateBatchOffsets_k.Args.mesh_details, mesh_details_ptr)
            .set_arg(CalculateBatchOffsets_k.Args.count, total_instances)
            .call_task();

        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - si;
            Editor.queue_event("render_model_batch_offsets", String.valueOf(e));
        }

        si = Editor.ACTIVE ? System.nanoTime() : 0;
        int[] raw_offsets = GPU.CL.read_pinned_int_buffer(GPU.compute.render_queue, mesh_offset_buf, CL_DataTypes.cl_int.size(), total_batches);
        mesh_offset_buf.release();
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
            int count_size = GPU.compute.calculate_preferred_global_size(count);
            long[] count_global_size = arg_long(count_size);
            long st = Editor.ACTIVE ? System.nanoTime() : 0;

            k_transfer_detail_data
                .buf_arg(TransferDetailData_k.Args.mesh_details, batch_data.mesh_details_buf)
                .set_arg(TransferDetailData_k.Args.offset, offset)
                .set_arg(TransferDetailData_k.Args.max_mesh, count)
                .call(count_global_size, GPU.compute.preferred_work_size);

            if (Editor.ACTIVE)
            {
                long e = System.nanoTime() - st;
                Editor.queue_event("render_detail_transfer", String.valueOf(e));
            }

            gpu_int2_scan.scan_int2(mesh_transfer_buf.ptr(), count);

            st = Editor.ACTIVE ? System.nanoTime() : 0;

            k_transfer_render_data
                .share_mem(command_buf)
                .share_mem(element_buf)
                .share_mem(vertex_buf)
                .share_mem(uv_buf)
                .share_mem(color_buf)
                .share_mem(slot_buf)
                .buf_arg(TransferRenderData_k.Args.mesh_details, batch_data.mesh_details_buf)
                .buf_arg(TransferRenderData_k.Args.mesh_texture, batch_data.mesh_texture_buf)
                .set_arg(TransferRenderData_k.Args.offset, offset)
                .set_arg(TransferRenderData_k.Args.max_index, count)
                .call(count_global_size, GPU.compute.preferred_work_size);

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
    public void render()
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;

        var batch_data = tick_CL();
        if (batch_data == null) return;

        tick_GL(batch_data);

        batch_data.mesh_details_buf.release();
        batch_data.mesh_texture_buf.release();

        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("render_model", String.valueOf(e));
        }
    }

    @Override
    public void destroy()
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

        element_buf.release();
        vertex_buf.release();
        command_buf.release();
        uv_buf.release();
        color_buf.release();
        slot_buf.release();
        total_buf.release();
        query_buf.release();
        counter_buf.release();
        offset_buf.release();
        mesh_transfer_buf.release();
    }
}
