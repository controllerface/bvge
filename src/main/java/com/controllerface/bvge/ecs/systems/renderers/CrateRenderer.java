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
import static org.lwjgl.opengl.GL31.glDrawElementsInstanced;
import static org.lwjgl.opengl.GL33.glVertexAttribDivisor;

public class CrateRenderer extends GameSystem
{
    public static final int TRANSFORM_VERTEX_COUNT = Constants.Rendering.MAX_BATCH_SIZE * VECTOR_4D_LENGTH;
    public static final int TRANSFORM_BUFFER_SIZE = TRANSFORM_VERTEX_COUNT * Float.BYTES;
    private final AbstractShader shader;
    private Texture texture;
    private final int[] texture_slots = {0};
    private HullIndexData crate_hulls;
    private int transform_buffer_id;
    private int vao;

    public CrateRenderer(ECS ecs)
    {
        super(ecs);
        this.shader = Assets.load_shader("box_model.glsl");
        init();
    }

    public void init()
    {
        var model = Models.get_model_by_index(Models.TEST_SQUARE_INDEX);
        this.texture = model.textures().get(0);
        var base_mesh = model.meshes()[0];
        var raw = base_mesh.raw_copy();

        vao = glGenVertexArrays();
        glBindVertexArray(vao);

        int element_buffer_id = glGenBuffers();
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_id);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, raw.r_faces(), GL_STATIC_DRAW);

        int model_buffer_id = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, model_buffer_id);
        glBufferData(GL_ARRAY_BUFFER, raw.r_vertices(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, VECTOR_2D_LENGTH, GL_FLOAT, false, VECTOR_FLOAT_2D_SIZE, 0);

        int texture_uv_buffer_id = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, texture_uv_buffer_id);
        glBufferData(GL_ARRAY_BUFFER, raw.r_uv_coords(), GL_STATIC_DRAW);
        glVertexAttribPointer(1, VECTOR_2D_LENGTH, GL_FLOAT, false, VECTOR_FLOAT_2D_SIZE, 0);

        transform_buffer_id = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, transform_buffer_id);
        glBufferData(GL_ARRAY_BUFFER, TRANSFORM_BUFFER_SIZE, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(2, VECTOR_4D_LENGTH, GL_FLOAT, false, VECTOR_FLOAT_4D_SIZE, 0);
        glVertexAttribDivisor(2, 1);

        GPU.share_memory(transform_buffer_id);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
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

        int offset = 0;
        for (int remaining = crate_hulls.count(); remaining > 0; remaining -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, remaining);
            GPU.GL_transforms(transform_buffer_id, crate_hulls.indices(), count, offset);
            glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, count);
            offset += count;
        }

        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        glDisableVertexAttribArray(2);
        glBindVertexArray(0);

        shader.detach();
        texture.unbind();
    }
}