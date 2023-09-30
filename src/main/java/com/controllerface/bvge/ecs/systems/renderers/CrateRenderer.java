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

import java.util.ArrayList;
import java.util.List;

import static com.controllerface.bvge.util.Constants.Rendering.*;
import static org.lwjgl.opengl.GL15.*;
import static org.lwjgl.opengl.GL20.*;
import static org.lwjgl.opengl.GL30.glBindVertexArray;
import static org.lwjgl.opengl.GL30.glGenVertexArrays;
import static org.lwjgl.opengl.GL31.glDrawArraysInstanced;
import static org.lwjgl.opengl.GL33.glVertexAttribDivisor;

public class CrateRenderer extends GameSystem
{
    public static final int TRANSFORM_VERTEX_COUNT = Constants.Rendering.MAX_BATCH_SIZE * VECTOR_4D_LENGTH;
    public static final int TRANSFORM_BUFFER_SIZE = TRANSFORM_VERTEX_COUNT * Float.BYTES;

    private final AbstractShader shader;
    private final Texture texture;
    private final List<CrateRenderBatch> batches;
    private int model_buffer_id;
    private int transform_buffer_id;
    private int texture_uv_buffer_id;
    private int color_buffer_id;

    public CrateRenderer(ECS ecs)
    {
        super(ecs);
        this.batches = new ArrayList<>();
        this.shader = Assets.shader("box_model.glsl");
        this.texture = Assets.texture("src/main/resources/img/crate.png");
        start();
    }

    public void start()
    {
        var mdl = Models.get_model_by_index(Models.CRATE_MODEL);
        var base_model = mdl.meshes()[0];
        var vbo_model = new float[12];
        vbo_model[0] = base_model.vertices()[0].x();  // tri 1 // p0 x
        vbo_model[1] = base_model.vertices()[0].y();           // p0 y

        vbo_model[2] = base_model.vertices()[1].x();           // p1 x
        vbo_model[3] = base_model.vertices()[1].y();           // p1 y

        vbo_model[4] = base_model.vertices()[2].x();           // p2 x
        vbo_model[5] = base_model.vertices()[2].y();           // p2 y

        vbo_model[6] = base_model.vertices()[0].x();  // tri 2 // p0 x
        vbo_model[7] = base_model.vertices()[0].y();           // p0 y

        vbo_model[8] = base_model.vertices()[2].x();           // p2 x
        vbo_model[9] = base_model.vertices()[2].y();           // p2 y

        vbo_model[10] = base_model.vertices()[3].x();          // p3 x
        vbo_model[11] = base_model.vertices()[3].y();          // p3 y


        // todo: update model loader to import UV co-ords
        var vbo_tex_coords = new float[12];
        vbo_tex_coords[0] = 0f;  // tri 1 // p1 u
        vbo_tex_coords[1] = 0f;           // p1 v

        vbo_tex_coords[2] = 1f;           // p2 u
        vbo_tex_coords[3] = 0f;           // p2 v

        vbo_tex_coords[4] = 1f;           // p3 u
        vbo_tex_coords[5] = 1f;           // p3 v

        vbo_tex_coords[6] = 0f;  // tri 2 // p1 u
        vbo_tex_coords[7] = 0f;           // p1 v

        vbo_tex_coords[8] = 1f;           // p3 u
        vbo_tex_coords[9] = 1f;           // p3 v

        vbo_tex_coords[10] = 0f;          // p4 u
        vbo_tex_coords[11] = 1f;          // p4 v


        // todo: decide if colors should be generated per-instance for variety or possibly algorithmically based
        //  on some other factors, like a modulus check based on object index
        var vbo_colors = new float[24];
        vbo_colors[0] = 0.5f;   // tri 1 // p1 r
        vbo_colors[1] = 0.35f;           // p1 g
        vbo_colors[2] = 0.05f;           // p1 b
        vbo_colors[3] = 1f;              // p1 a

        vbo_colors[4] = 0.23f;           // p2 r
        vbo_colors[5] = 0.21f;           // p2 g
        vbo_colors[6] = 0.2f;            // p2 b
        vbo_colors[7] = 1f;              // p2 a

        vbo_colors[8] = 0.55f;           // p3 r
        vbo_colors[9] = 0.5f;            // p3 g
        vbo_colors[10] = 0.48f;          // p3 b
        vbo_colors[11] = 1f;             // p3 a

        vbo_colors[12] = 0.5f;  // tri 2 // p1 r
        vbo_colors[13] = 0.35f;          // p1 g
        vbo_colors[14] = 0.05f;          // p1 b
        vbo_colors[15] = 1f;             // p1 a

        vbo_colors[16] = 0.55f;          // p3 r
        vbo_colors[17] = 0.5f;           // p3 g
        vbo_colors[18] = 0.48f;          // p3 b
        vbo_colors[19] = 1f;             // p3 a

        vbo_colors[20] = .5f;            // p4 r
        vbo_colors[21] = .5f;            // p4 g
        vbo_colors[22] = .5f;            // p4 b
        vbo_colors[23] = 1f;             // p4 a


        // load model data
        model_buffer_id = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, model_buffer_id);
        glBufferData(GL_ARRAY_BUFFER, vbo_model, GL_STATIC_DRAW);

        texture_uv_buffer_id = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, texture_uv_buffer_id);
        glBufferData(GL_ARRAY_BUFFER, vbo_tex_coords, GL_STATIC_DRAW);

        color_buffer_id = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, color_buffer_id);
        glBufferData(GL_ARRAY_BUFFER, vbo_colors, GL_STATIC_DRAW);

        // create buffer for transforms, batches will use this during the rendering process
        transform_buffer_id = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, transform_buffer_id); // this attribute comes from a different vertex buffer
        glBufferData(GL_ARRAY_BUFFER, TRANSFORM_BUFFER_SIZE, GL_DYNAMIC_DRAW);

        // share the buffer with the CL context
        GPU.share_memory(transform_buffer_id);

        // unbind
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    private void render()
    {
        for (CrateRenderBatch batch : batches)
        {
            batch.render();
        }
    }

    @Override
    public void tick(float dt)
    {
        // todo: will need to account for this happening more than once
        if (Models.is_model_dirty(Models.CRATE_MODEL))
        {
            var instances = Models.get_model_instances(Models.CRATE_MODEL);
            int[] indices = new int[instances.size()];
            int[] counter = new int[1];
            instances.forEach(integer -> indices[counter[0]++] = integer);

            // get the number of models that need to be rendered
            var model_count = Models.get_instance_count(Models.CRATE_MODEL);

            var needed_batches = model_count / Constants.Rendering.MAX_BATCH_SIZE;
            var r = model_count % Constants.Rendering.MAX_BATCH_SIZE;
            if (r > 0)
            {
                needed_batches++;
            }
            while (needed_batches > batches.size())
            {
                var b = new CrateRenderBatch(shader,
                    texture,
                    transform_buffer_id,
                    model_buffer_id,
                    texture_uv_buffer_id,
                    color_buffer_id);
                batches.add(b);
            }

            int next = 0;
            int offset = 0;
            for (int i = model_count; i > 0; i -= Constants.Rendering.MAX_BATCH_SIZE)
            {
                int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, i);
                int[] block = new int[count];
                System.arraycopy(indices, offset, block, 0, count);
                var b = batches.get(next++);
                b.setModelCount(count);
                b.setIndices(block);
                b.start();
                offset += count;
            }

            Models.set_model_clean(Models.CRATE_MODEL);
        }

        render();
    }

    private static class CrateRenderBatch
    {
        private final AbstractShader shader;
        private final Texture texture;
        private int vao;
        private int mesh_count;
        private int index_buffer_id;
        private final int transform_buffer_ID;
        private final int model_buffer_id;
        private final int texture_uv_buffer_id;
        private final int color_buffer_id;
        private final int[] texture_slots = { 0 };
        private int[] indices; // will be large enough to hold a full batch, but may only contain a partial one

        public CrateRenderBatch(AbstractShader shader,
                                Texture texture,
                                int transform_buffer_ID,
                                int model_buffer_id,
                                int texture_uv_buffer_id,
                                int color_buffer_id)
        {
            this.shader = shader;
            this.texture = texture;
            this.transform_buffer_ID = transform_buffer_ID;
            this.model_buffer_id = model_buffer_id;
            this.texture_uv_buffer_id = texture_uv_buffer_id;
            this.color_buffer_id = color_buffer_id;
        }

        public void start()
        {
            // Generate and bind a Vertex Array Object
            vao = glGenVertexArrays();
            glBindVertexArray(vao); // this sets this VAO as being active

            index_buffer_id = glGenBuffers();
            glBindBuffer(GL_ARRAY_BUFFER, index_buffer_id);
            glBufferData(GL_ARRAY_BUFFER, indices, GL_STATIC_DRAW);

            glBindBuffer(GL_ARRAY_BUFFER, model_buffer_id);
            glVertexAttribPointer(0, VECTOR_2D_LENGTH, GL_FLOAT, false, VECTOR_FLOAT_2D_SIZE, 0);

            glBindBuffer(GL_ARRAY_BUFFER, transform_buffer_ID);
            glVertexAttribPointer(1, VECTOR_4D_LENGTH, GL_FLOAT, false, VECTOR_FLOAT_4D_SIZE, 0);
            glVertexAttribDivisor(1, 1);

            glBindBuffer(GL_ARRAY_BUFFER, texture_uv_buffer_id);
            glVertexAttribPointer(2, VECTOR_2D_LENGTH, GL_FLOAT, false, VECTOR_FLOAT_2D_SIZE, 0);

            glBindBuffer(GL_ARRAY_BUFFER, color_buffer_id);
            glVertexAttribPointer(3, VECTOR_4D_LENGTH, GL_FLOAT, false, VECTOR_FLOAT_4D_SIZE, 0);

            // share the buffer with the CL context
            GPU.share_memory(index_buffer_id);

            // unbind
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            // unbind the vao since we're done defining things for now
            glBindVertexArray(0);
        }

        public void clear()
        {
            mesh_count = 0;
        }

        public void setIndices(int[] indices)
        {
            this.indices = indices;
        }

        public void setModelCount(int numModels)
        {
            this.mesh_count = numModels;
        }


        public void render()
        {
            glBindVertexArray(vao);

            shader.use();
            texture.bind(GL_TEXTURE0);

            shader.uploadMat4f("uVP", Window.get().camera().get_uVP());
            shader.uploadIntArray("uTextures", texture_slots);

            // this call moves the physics transforms of the tracked hulls into the buffer for rendering
            // todo: this will need to change to use bone weights, but possibly processing them in CL
            //  not in GL
            GPU.GL_transforms(index_buffer_id, transform_buffer_ID, mesh_count);


            glEnableVertexAttribArray(0);
            glEnableVertexAttribArray(1);
            glEnableVertexAttribArray(2);
            glEnableVertexAttribArray(3);

            glDrawArraysInstanced(GL_TRIANGLES, 0, 6, mesh_count);

            glDisableVertexAttribArray(0);
            glDisableVertexAttribArray(1);
            glDisableVertexAttribArray(2);
            glDisableVertexAttribArray(3);
            glBindVertexArray(0);

            shader.detach();
            texture.unbind();
        }
    }
}
