package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.HullIndexData;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.geometry.Models;
import com.controllerface.bvge.gl.AbstractShader;
import com.controllerface.bvge.gl.GLUtils;
import com.controllerface.bvge.gl.Texture;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;

import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_4D_LENGTH;
import static org.lwjgl.opencl.CL12.clReleaseMemObject;
import static org.lwjgl.opengl.GL30.glBindVertexArray;
import static org.lwjgl.opengl.GL31.glDrawElementsInstanced;
import static org.lwjgl.opengl.GL45C.*;

public class CrateRenderer extends GameSystem
{
    public static final int TRANSFORM_VERTEX_COUNT = Constants.Rendering.MAX_BATCH_SIZE * VECTOR_4D_LENGTH;
    public static final int TRANSFORM_BUFFER_SIZE = TRANSFORM_VERTEX_COUNT * Float.BYTES;

    private static final int POSITION_ATTRIBUTE = 0;
    private static final int UV_COORD_ATTRIBUTE = 1;
    private static final int TRANSFORM_ATTRIBUTE = 2;

    private final AbstractShader shader;
    private Texture texture;
    private final int[] texture_slots = {0};
    private HullIndexData crate_hulls;
    private int transform_buffer_id;
    private int vao_id;

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

        vao_id = glCreateVertexArrays();

        int element_buffer_id = glCreateBuffers();
        glNamedBufferData(element_buffer_id, raw.r_faces(), GL_STATIC_DRAW);
        glVertexArrayElementBuffer(vao_id, element_buffer_id);

        GLUtils.fill_buffer_vec2(vao_id, POSITION_ATTRIBUTE, raw.r_vertices());
        GLUtils.fill_buffer_vec2(vao_id, UV_COORD_ATTRIBUTE, raw.r_uv_coords());

        transform_buffer_id = GLUtils.new_buffer_vec4(vao_id, TRANSFORM_ATTRIBUTE, TRANSFORM_BUFFER_SIZE);
        glVertexArrayBindingDivisor(vao_id, TRANSFORM_ATTRIBUTE, 1);

        GPU.share_memory(transform_buffer_id);
    }


    @Override
    public void tick(float dt)
    {
        if (crate_hulls != null && crate_hulls.indices() != -1)
        {
            clReleaseMemObject(crate_hulls.indices());
        }
        crate_hulls = GPU.GL_hull_filter(Models.TEST_SQUARE_INDEX);

        if (crate_hulls.count() == 0)
        {
            return;
        }

        glBindVertexArray(vao_id);

        shader.use();
        texture.bind(0);

        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());
        shader.uploadIntArray("uTextures", texture_slots);

        glEnableVertexArrayAttrib(vao_id, POSITION_ATTRIBUTE);
        glEnableVertexArrayAttrib(vao_id, UV_COORD_ATTRIBUTE);
        glEnableVertexArrayAttrib(vao_id, TRANSFORM_ATTRIBUTE);

        int offset = 0;
        for (int remaining = crate_hulls.count(); remaining > 0; remaining -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, remaining);
            GPU.GL_transforms(transform_buffer_id, crate_hulls.indices(), count, offset);
            glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, count);
            offset += count;
        }

        glDisableVertexArrayAttrib(vao_id, POSITION_ATTRIBUTE);
        glDisableVertexArrayAttrib(vao_id, UV_COORD_ATTRIBUTE);
        glDisableVertexArrayAttrib(vao_id, TRANSFORM_ATTRIBUTE);

        glBindVertexArray(0);

        shader.detach();
    }
}