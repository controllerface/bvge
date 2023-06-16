package com.controllerface.bvge.rendering;

import com.controllerface.bvge.TransformEX;
import com.controllerface.bvge.ecs.ComponentType;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.SystemEX;
import com.controllerface.bvge.util.AssetPool;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class RendererEX implements SystemEX
{
    private ECS ecs;
    private final int MAX_BATCH_SIZE = 1000;
    private List<SpriteRenderBatchEX> batches;

    private Shader currentShader;

    public RendererEX()
    {
        this.batches = new ArrayList<>();
        this.currentShader = AssetPool.getShader("default.glsl");
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
            SpriteRenderBatchEX newBatch = new SpriteRenderBatchEX(MAX_BATCH_SIZE,
                sprite.transform.zIndex,
                currentShader);
            newBatch.start();
            batches.add(newBatch);
            newBatch.addSprite(sprite);
            Collections.sort(batches);
        }
    }

    private void render()
    {
        currentShader.use();
        for (SpriteRenderBatchEX batch : batches)
        {
            batch.render();
        }
        batches.forEach(SpriteRenderBatchEX::clear);
    }

    @Override
    public void setup(ECS ecs)
    {
        this.ecs = ecs;
    }

    @Override
    public void run(float dt)
    {
        var sprites = ecs.getComponents(ComponentType.SpriteComponent);

        sprites.forEach((entity, component) ->
        {
            SpriteComponentEX rend = ComponentType.SpriteComponent.coerce(component);
            var transform = ecs.getComponentFor(entity, ComponentType.Transform);
            TransformEX t = ComponentType.Transform.coerce(transform);

            rend.transform.position.x = t.position.x;
            rend.transform.position.y = t.position.y;
            rend.transform.scale.x = t.scale.x;
            rend.transform.scale.y = t.scale.y;

            this.add(rend);

//            System.out.println("Entity: " + entity);
//            System.out.println("Transform: " + t);
//            System.out.println("Sprite: " + rend);
        });

        // 1: get sprite components

        // 2: make new batches

        // 3: render all batches
        render();
    }
}