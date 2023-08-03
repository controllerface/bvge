package com.controllerface.bvge.ecs.systems.renderers;

import com.controllerface.bvge.cl.OpenCL;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.ecs.systems.renderers.batches.BoxRenderBatch;
import com.controllerface.bvge.ecs.systems.renderers.batches.CircleRenderBatch;
import com.controllerface.bvge.gl.Models;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.gl.Texture;
import com.controllerface.bvge.util.AssetPool;
import com.controllerface.bvge.util.Constants;

import java.util.ArrayList;
import java.util.List;

import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_4D_LENGTH;
import static org.lwjgl.opengl.GL15.*;

public class CircleRenderer extends GameSystem
{
    public static final int TRANSFORM_VERTEX_COUNT = Constants.Rendering.MAX_BATCH_SIZE * VECTOR_4D_LENGTH;
    public static final int TRANSFORM_BUFFER_SIZE = TRANSFORM_VERTEX_COUNT * Float.BYTES;

    private final int model_index = 0;
    private final Shader shader;
    //private final Texture texture;
    private final List<CircleRenderBatch> batches;
    private int model_buffer_id;
    private int transform_buffer_id;
    private int texture_uv_buffer_id;
    private int color_buffer_id;


    public CircleRenderer(ECS ecs)
    {
        super(ecs);
        this.batches = new ArrayList<>();
        this.shader = AssetPool.getShader("circle_shader.glsl");
        //this.texture = AssetPool.getTexture("src/main/resources/img/crate.png");
        //start();
    }

    public void start()
    {
        // todo: define a square model (quad) to draw the circle onto. The quad size will scale
        //  the same as the circle size, this should allow circles to function the same as other
        //  objects in terms of rendering.
        var base_model = Models.get_model_by_index(model_index);
        var vbo_model = new float[12];
        vbo_model[0] = base_model[0];  // tri 1 // p1 x
        vbo_model[1] = base_model[1];           // p1 y

        vbo_model[2] = base_model[2];           // p2 x
        vbo_model[3] = base_model[3];           // p2 y

        vbo_model[4] = base_model[4];           // p3 x
        vbo_model[5] = base_model[5];           // p3 y

        vbo_model[6] = base_model[0];  // tri 2 // p1 x
        vbo_model[7] = base_model[1];           // p1 y

        vbo_model[8] = base_model[4];           // p3 x
        vbo_model[9] = base_model[5];           // p3 y

        vbo_model[10] = base_model[6];          // p4 x
        vbo_model[11] = base_model[7];          // p4 y


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

        //fColor = vec3(0.5f, 0.35f, 0.05f);
        //fColor = vec3(0.23, 0.21, 0.2);
        //fColor = vec4(0.55, 0.5, 0.48, 1);

        var vbo_colors = new float[24];
        vbo_colors[0] = 0.5f; // tri 1 // p1 r
        vbo_colors[1] = 0.35f;          // p1 g
        vbo_colors[2] = 0.05f;          // p1 b
        vbo_colors[3] = 1f;          // p1 a

        vbo_colors[4] = 0.23f;          // p2 r
        vbo_colors[5] = 0.21f;          // p2 g
        vbo_colors[6] = 0.2f;          // p2 b
        vbo_colors[7] = 1f;          // p2 a

        vbo_colors[8] = 0.55f;          // p3 r
        vbo_colors[9] = 0.5f;          // p3 g
        vbo_colors[10] = 0.48f;          // p3 b
        vbo_colors[11] = 1f;          // p3 a

        vbo_colors[12] = 0.5f;  // tri 2 // p1 r
        vbo_colors[13] = 0.35f;          // p1 g
        vbo_colors[14] = 0.05f;          // p1 b
        vbo_colors[15] = 1f;             // p1 a

        vbo_colors[16] = 0.55f;           // p3 r
        vbo_colors[17] = 0.5f;            // p3 g
        vbo_colors[18] = 0.48f;          // p3 b
        vbo_colors[19] = 1f;             // p3 a

        vbo_colors[20] = .5f;          // p4 r
        vbo_colors[21] = .5f;          // p4 g
        vbo_colors[22] = .5f;          // p4 b
        vbo_colors[23] = 1f;          // p4 a



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
        OpenCL.share_memory(transform_buffer_id);

        // unbind
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    private void render()
    {
        for (CircleRenderBatch batch : batches)
        {
            batch.render();
        }
    }

    @Override
    public void run(float dt)
    {
//        // todo: will ned to account for this happening more than once
//        if (Models.is_model_dirty(model_index))
//        {
//            var instances = Models.get_model_instances(model_index);
//            int[] indices = new int[instances.size()];
//            int[] counter = new int[1];
//            instances.forEach(integer -> indices[counter[0]++] = integer);
//
//            // get the number of models that need to be rendered
//            var model_count = Models.get_instance_count(model_index);
//
//            var needed_batches = model_count / Constants.Rendering.MAX_BATCH_SIZE;
//            var r = model_count % Constants.Rendering.MAX_BATCH_SIZE;
//            if (r > 0)
//            {
//                needed_batches++;
//            }
//            while (needed_batches > batches.size())
//            {
//                var b = new CircleRenderBatch(shader,
//                    //texture,
//                    transform_buffer_id,
//                    model_buffer_id,
//                    texture_uv_buffer_id,
//                    color_buffer_id);
//                batches.add(b);
//            }
//
//            int next = 0;
//            int offset = 0;
//            for (int i = model_count; i > 0; i -= Constants.Rendering.MAX_BATCH_SIZE)
//            {
//                int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, i);
//                int[] block = new int[count];
//                System.arraycopy(indices, offset, block, 0, count);
//                var b = batches.get(next++);
//                b.setModelCount(count);
//                b.setIndices(block);
//                b.start();
//                offset += count;
//            }
//
//
//            Models.set_model_clean(model_index);
//        }
//
//        render();
    }

    @Override
    public void shutdown()
    {

    }
}
