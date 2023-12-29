package com.controllerface.bvge.ecs.systems.renderers;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.HullIndexData;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.geometry.Models;
import com.controllerface.bvge.gl.AbstractShader;
import com.controllerface.bvge.gl.Texture;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;

import static com.controllerface.bvge.util.Constants.Rendering.*;
import static org.jocl.CL.clReleaseMemObject;
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
    private Texture texture;
    private final int[] texture_slots = {0};
    private HullIndexData crate_hulls;
    private int color_buffer_id;
    private int model_buffer_id;
    private int transform_buffer_id;
    private int texture_uv_buffer_id;
    private int vao;

    public CrateRenderer(ECS ecs)
    {
        super(ecs);
        this.shader = Assets.load_shader("box_model.glsl");
        init();
    }

    public void init()
    {
        var mdl = Models.get_model_by_index(Models.TEST_SQUARE_INDEX);
        this.texture = mdl.textures().get(0);
        var base_mesh = mdl.meshes()[0];
        var vbo_model = new float[12];
        var vbo_tex_coords = new float[12];
        int count = 0;

        for (int i = 0; i < base_mesh.faces().length; i++)
        {
            var face = base_mesh.faces()[i];
            var p0 = base_mesh.vertices()[face.p0()];
            var p1 = base_mesh.vertices()[face.p1()];
            var p2 = base_mesh.vertices()[face.p2()];

            vbo_model[count] = p0.x();                            //tri i     // p0 x
            vbo_model[count + 1] = p0.y();                                    // p0 y
            vbo_tex_coords[count] = p0.uv_data().get(0).x;                    // p0 u
            vbo_tex_coords[count + 1] = p0.uv_data().get(0).y;                // p0 v

            vbo_model[count + 2] = p1.x();                                    // p1 x
            vbo_model[count + 3] = p1.y();                                    // p1 y
            vbo_tex_coords[count + 2] = p1.uv_data().get(0).x;                // p0 u
            vbo_tex_coords[count + 3] = p1.uv_data().get(0).y;                // p0 v

            vbo_model[count + 4] = p2.x();                                    // p2 x
            vbo_model[count + 5] = p2.y();                                    // p2 y
            vbo_tex_coords[count + 4] = p2.uv_data().get(0).x;                // p0 u
            vbo_tex_coords[count + 5] = p2.uv_data().get(0).y;                // p0 v
            count += 6;
        }

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

        vao = glGenVertexArrays();
        glBindVertexArray(vao);

        // load model data
        model_buffer_id = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, model_buffer_id);
        glBufferData(GL_ARRAY_BUFFER, vbo_model, GL_STATIC_DRAW);
        glVertexAttribPointer(0, VECTOR_2D_LENGTH, GL_FLOAT, false, VECTOR_FLOAT_2D_SIZE, 0);

        // create buffer for transforms, batches will use this during the rendering process
        transform_buffer_id = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, transform_buffer_id); // this attribute comes from a different vertex buffer
        glBufferData(GL_ARRAY_BUFFER, TRANSFORM_BUFFER_SIZE, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(1, VECTOR_4D_LENGTH, GL_FLOAT, false, VECTOR_FLOAT_4D_SIZE, 0);
        glVertexAttribDivisor(1, 1);

        texture_uv_buffer_id = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, texture_uv_buffer_id);
        glBufferData(GL_ARRAY_BUFFER, vbo_tex_coords, GL_STATIC_DRAW);
        glVertexAttribPointer(2, VECTOR_2D_LENGTH, GL_FLOAT, false, VECTOR_FLOAT_2D_SIZE, 0);

        color_buffer_id = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, color_buffer_id);
        glBufferData(GL_ARRAY_BUFFER, vbo_colors, GL_STATIC_DRAW);
        glVertexAttribPointer(3, VECTOR_4D_LENGTH, GL_FLOAT, false, VECTOR_FLOAT_4D_SIZE, 0);

        // share the buffer with the CL context
        GPU.share_memory(transform_buffer_id);

        // unbind
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // unbind the vao since we're done defining things for now
        glBindVertexArray(0);
    }


    @Override
    public void tick(float dt)
    {
        if (crate_hulls != null && crate_hulls.indices() != null)
        {
            clReleaseMemObject(crate_hulls.indices());
        }
        crate_hulls = GPU.GL_hull_filter(Models.TEST_SQUARE_INDEX);

        if (crate_hulls.count() == 0)
        {
            return;
        }

        glBindVertexArray(vao);

        shader.use();
        texture.bind(GL_TEXTURE0);

        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());
        shader.uploadIntArray("uTextures", texture_slots);

        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glEnableVertexAttribArray(2);
        glEnableVertexAttribArray(3);

        int offset = 0;
        for (int remaining = crate_hulls.count(); remaining > 0; remaining -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, remaining);
            GPU.GL_transforms(transform_buffer_id, crate_hulls.indices(), count, offset);
            glDrawArraysInstanced(GL_TRIANGLES, 0, 6, count);
            offset += count;
        }

        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        glDisableVertexAttribArray(2);
        glDisableVertexAttribArray(3);
        glBindVertexArray(0);

        shader.detach();
        texture.unbind();
    }
}
