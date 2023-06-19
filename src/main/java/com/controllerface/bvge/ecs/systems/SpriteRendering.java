package com.controllerface.bvge.ecs.systems;

import com.controllerface.bvge.Transform;
import com.controllerface.bvge.ecs.Component;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.GameComponent;
import com.controllerface.bvge.rendering.Shader;
import com.controllerface.bvge.rendering.SpriteComponent;
import com.controllerface.bvge.rendering.SpriteRenderBatch;
import com.controllerface.bvge.rendering.Texture;
import com.controllerface.bvge.util.AssetPool;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class SpriteRendering extends GameSystem
{
    private Shader shader;
    private List<SpriteRenderBatch> batches;

    public SpriteRendering(ECS ecs)
    {
        super(ecs);
        this.batches = new ArrayList<>();
        this.shader = AssetPool.getShader("default.glsl");
    }

    private void add(SpriteComponent sprite)
    {
        boolean added = false;
        for (SpriteRenderBatch batch : batches)
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
            SpriteRenderBatch newBatch = new SpriteRenderBatch(sprite.transform.zIndex, shader);
            newBatch.start();
            batches.add(newBatch);
            newBatch.addSprite(sprite);
            Collections.sort(batches);
        }
    }

    private void render()
    {
        shader.use();
        for (SpriteRenderBatch batch : batches)
        {
            batch.render();
            batch.clear();
        }
    }

    private boolean runyet = false;

    @Override
    public void run(float dt)
    {
        var spriteComponents = ecs.getComponents(Component.SpriteComponent);

        for (Map.Entry<String, GameComponent> entry : spriteComponents.entrySet())
        {
            String entity = entry.getKey();
            GameComponent component = entry.getValue();
            SpriteComponent sprite = Component.SpriteComponent.coerce(component);
            var t = ecs.getComponentFor(entity, Component.Transform);
            Transform transform = Component.Transform.coerce(t);
            sprite.transform.position.x = transform.position.x;
            sprite.transform.position.y = transform.position.y;
            sprite.transform.scale.x = transform.scale.x;
            sprite.transform.scale.y = transform.scale.y;
            if (!runyet)
            {
                this.add(sprite);
            }
        }

        render();
    }
}