package com.controllerface.bvge.ecs.systems.renderers;

import com.controllerface.bvge.cl.OpenCL;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.ecs.systems.renderers.batches.CircleRenderBatch;
import com.controllerface.bvge.geometry.Models;
import com.controllerface.bvge.gl.AbstractShader;
import com.controllerface.bvge.geometry.Meshes;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.util.Constants;

import java.util.ArrayList;
import java.util.List;

import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_4D_LENGTH;
import static org.lwjgl.opengl.GL15.*;

public class CircleRenderer extends GameSystem
{
    public static final int TRANSFORM_VERTEX_COUNT = Constants.Rendering.MAX_BATCH_SIZE * VECTOR_4D_LENGTH;
    public static final int TRANSFORM_BUFFER_SIZE = TRANSFORM_VERTEX_COUNT * Float.BYTES;

    private final AbstractShader shader;
    private final List<CircleRenderBatch> batches;
    private int transform_buffer_id;

    public CircleRenderer(ECS ecs)
    {
        super(ecs);
        this.batches = new ArrayList<>();
        this.shader = Assets.shader("circle_shader.glsl");
        start();
    }

    public void start()
    {
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
//        // todo: will need to account for this happening more than once
        if (Models.is_model_dirty(Models.CIRCLE_MODEL))
        {
            var instances = Models.get_model_instances(Models.CIRCLE_MODEL);
            int[] indices = new int[instances.size()];
            int[] counter = new int[1];
            instances.forEach(integer -> indices[counter[0]++] = integer);

            // get the number of models that need to be rendered
            var model_count = Models.get_instance_count(Models.CIRCLE_MODEL);

            var needed_batches = model_count / Constants.Rendering.MAX_BATCH_SIZE;
            var r = model_count % Constants.Rendering.MAX_BATCH_SIZE;
            if (r > 0)
            {
                needed_batches++;
            }
            while (needed_batches > batches.size())
            {
                var b = new CircleRenderBatch(shader, transform_buffer_id);
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


            Models.set_model_clean(Models.CIRCLE_MODEL);
        }

        render();
    }

    @Override
    public void shutdown()
    {

    }
}
