package com.controllerface.bvge.ecs.systems.renderers;

import com.controllerface.bvge.cl.OpenCL;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.ecs.systems.renderers.batches.BoxRenderBatch;
import com.controllerface.bvge.gl.Models;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.util.AssetPool;
import com.controllerface.bvge.util.Constants;

import java.util.ArrayList;
import java.util.List;

import static org.lwjgl.opengl.GL15.*;

public class BoxRenderer extends GameSystem
{
    public static final int VERTEX_SIZE = 2; // a vertex is 2 floats (x,y)
    public static final int VERTEX_SIZE_BYTES = VERTEX_SIZE * Float.BYTES;

    public static final int TRANSFORM_SIZE = 4; // a transform is 3 floats (x,y,w)
    public static final int TRANSFORM_SIZE_BYTES = TRANSFORM_SIZE * Float.BYTES;
    public static final int TRANSFORM_VERTEX_COUNT = Constants.Rendering.MAX_BATCH_SIZE * TRANSFORM_SIZE;
    public static final int TRANSFORM_BUFFER_SIZE = TRANSFORM_VERTEX_COUNT * Float.BYTES;

    private final int model_index = 0;
    private final Shader shader;
    private final List<BoxRenderBatch> batches;
    private int model_buffer_id;
    private int tranform_buffer_ID;

    public BoxRenderer(ECS ecs)
    {
        super(ecs);
        this.batches = new ArrayList<>();
        this.shader = AssetPool.getShader("box_model.glsl"); // todo: need new shader program
        start();
    }

    public void start()
    {
        // todo: for now, just scale the model manually by 10x, until scale transform is wired up
        var model = Models.get_model_by_index(model_index);
        model[0] = model[0] * 10;
        model[1] = model[1] * 10;
        model[2] = model[2] * 10;
        model[3] = model[3] * 10;
        model[4] = model[4] * 10;
        model[5] = model[5] * 10;
        model[6] = model[6] * 10;
        model[7] = model[7] * 10;

        // load model data
        model_buffer_id = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, model_buffer_id);
        glBufferData(GL_ARRAY_BUFFER, model, GL_STATIC_DRAW);

        // create buffer for transforms, batches will use this during the rendering process
        tranform_buffer_ID = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, tranform_buffer_ID); // this attribute comes from a different vertex buffer
        glBufferData(GL_ARRAY_BUFFER, TRANSFORM_BUFFER_SIZE, GL_DYNAMIC_DRAW);

        // share the buffer with the CL context
        OpenCL.share_memory(tranform_buffer_ID);

        // unbind
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    private void render()
    {
        for (BoxRenderBatch batch : batches)
        {
            batch.render();
        }
    }

    @Override
    public void run(float dt)
    {
        // todo: will ned to account for this happening more than once
        if (Models.is_model_dirty(model_index))
        {
            var instances = Models.get_model_instances(model_index);
            int[] indices = new int[instances.size()];
            int[] counter = new int[1];
            instances.forEach(integer -> indices[counter[0]++] = integer);

            // get the number of models that need to be rendered
            var model_count = Models.get_instance_count(model_index);
            System.out.println("debug: count="+ model_count);

            var needed_batches = model_count / Constants.Rendering.MAX_BATCH_SIZE;
            var r = model_count % Constants.Rendering.MAX_BATCH_SIZE;
            if (r > 0)
            {
                needed_batches++;
            }
            while (needed_batches > batches.size())
            {
                var b = new BoxRenderBatch(shader, tranform_buffer_ID, model_buffer_id);
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


            Models.set_model_clean(model_index);
        }

        render();
    }

    @Override
    public void shutdown()
    {

    }
}
