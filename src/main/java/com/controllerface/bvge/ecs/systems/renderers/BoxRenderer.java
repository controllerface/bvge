package com.controllerface.bvge.ecs.systems.renderers;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.cl.OpenCL;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.ecs.systems.renderers.batches.BoxRenderBatch;
import com.controllerface.bvge.ecs.systems.renderers.batches.EdgeRenderBatch;
import com.controllerface.bvge.gl.Models;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.util.AssetPool;
import com.controllerface.bvge.util.Constants;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.lwjgl.opengl.GL15.*;
import static org.lwjgl.opengl.GL20C.glVertexAttribPointer;
import static org.lwjgl.opengl.GL30.glGenVertexArrays;

public class BoxRenderer extends GameSystem
{
    private static final int VERTEX_SIZE = 2; // a vertex is 2 floats (x,y)
    private static final int VERTEX_SIZE_BYTES = VERTEX_SIZE * Float.BYTES;

    private static final int TRANSFORM_SIZE = 2; // a transform is 3 floats (x,y,w)
    private static final int TRANSFORM_SIZE_BYTES = TRANSFORM_SIZE * Float.BYTES;

    private static final int ORIGIN_VERTEX_COUNT = Constants.Rendering.MAX_BATCH_SIZE  * VERTEX_SIZE;
    private static final int ORIGIN_BUFFER_SIZE = ORIGIN_VERTEX_COUNT * Float.BYTES;

    private static final int TRANSFORM_VERTEX_COUNT = Constants.Rendering.MAX_BATCH_SIZE;
    private static final int TRANSFORM_BUFFER_SIZE = TRANSFORM_VERTEX_COUNT * Float.BYTES;


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
        // todo: need to just make the vbo for the model, and each batch will need its own vao

        // Generate and bind a Vertex Array Object
        //vaoID = glGenVertexArrays();
        //glBindVertexArray(vaoID); // this sets this VAO as being active

        // create a buffer for holding the transforms for each frame
        tranform_buffer_ID = glGenBuffers();
        model_buffer_id = glGenBuffers();


        var modl = Models.get_model_by_index(model_index);
        modl[0] = modl[0] * 10;
        modl[1] = modl[1] * 10;
        modl[2] = modl[2] * 10;
        modl[3] = modl[3] * 10;
        modl[4] = modl[4] * 10;
        modl[5] = modl[5] * 10;
        modl[6] = modl[6] * 10;
        modl[7] = modl[7] * 10;

        // load model data and configure the vertex attributes
        glBindBuffer(GL_ARRAY_BUFFER, model_buffer_id);
        glBufferData(GL_ARRAY_BUFFER, modl, GL_STATIC_DRAW);

        // create buffer for transforms and configure the instance divisor
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






//        var edge_count = Main.Memory.edgesCount();
//        var needed_batches = edge_count / Constants.Rendering.MAX_BATCH_SIZE;
//        var r = edge_count % Constants.Rendering.MAX_BATCH_SIZE;
//        if (r > 0)
//        {
//            needed_batches++;
//        }
//        while (needed_batches > batches.size())
//        {
//            var b = new EdgeRenderBatch(shader, vaoID, vboID);
//            batches.add(b);
//        }
//
//        int next = 0;
//        int offset = 0;
//        for (int i = edge_count; i > 0; i -= Constants.Rendering.MAX_BATCH_SIZE)
//        {
//            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, i);
//            var b = batches.get(next++);
//            b.setModelCount(count);
//            b.setOffset(offset);
//            offset += count;
//        }
        render();
    }

    @Override
    public void shutdown()
    {

    }
}
