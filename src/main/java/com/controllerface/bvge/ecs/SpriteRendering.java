package com.controllerface.bvge.ecs;

import com.controllerface.bvge.TransformEX;
import com.controllerface.bvge.rendering.Shader;
import com.controllerface.bvge.rendering.SpriteComponentEX;
import com.controllerface.bvge.rendering.SpriteRenderBatchEX;
import com.controllerface.bvge.rendering.Texture;
import com.controllerface.bvge.util.AssetPool;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class SpriteRendering extends SystemEX
{
    private Shader shader;
    private List<SpriteRenderBatchEX> batches;

    public SpriteRendering(ECS ecs)
    {
        super(ecs);
        this.batches = new ArrayList<>();
        this.shader = AssetPool.getShader("default.glsl");
    }

    private void add(SpriteComponentEX sprite)
    {
        boolean added = false;
        for (SpriteRenderBatchEX batch : batches)
        {
            if (batch.hasRoom() && batch.zIndex() == sprite.transform.zIndex)
            {
                Texture tex = sprite.getTexture();
                if (tex == null || (batch.hasTexture(tex) || batch.hasTextureRoom()))
                {
                    batch.addSprite(sprite);
                    added = true;
                    break;
                }
            }
        }

        if (!added)
        {
            SpriteRenderBatchEX newBatch = new SpriteRenderBatchEX(sprite.transform.zIndex, shader);
            newBatch.start();
            batches.add(newBatch);
            newBatch.addSprite(sprite);
            Collections.sort(batches);
        }
    }

    private void render()
    {
        shader.use();
        for (SpriteRenderBatchEX batch : batches)
        {
            batch.render();
            batch.clear();
        }
    }

    @Override
    public void run(float dt)
    {
        for (Map.Entry<String, Component_EX> entry : ecs.getComponents(Component.SpriteComponent).entrySet())
        {
            String entity = entry.getKey();
            Component_EX component = entry.getValue();
            SpriteComponentEX sprite = Component.SpriteComponent.coerce(component);
            var t = ecs.getComponentFor(entity, Component.Transform);
            TransformEX transform = Component.Transform.coerce(t);
            sprite.transform.position.x = transform.position.x;
            sprite.transform.position.y = transform.position.y;
            sprite.transform.scale.x = transform.scale.x;
            sprite.transform.scale.y = transform.scale.y;
            this.add(sprite);
        }
        render();
    }
}