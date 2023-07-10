package com.controllerface.bvge.ecs.systems.renderers;

import com.controllerface.bvge.data.FTransform;
import com.controllerface.bvge.ecs.components.Component;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.GameComponent;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.ecs.components.SpriteComponent;
import com.controllerface.bvge.gl.batches.SpriteRenderBatch;
import com.controllerface.bvge.gl.Texture;
import com.controllerface.bvge.util.AssetPool;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class SpriteRenderer extends GameSystem
{
    private Shader shader;
    private List<SpriteRenderBatch> batches;

    public SpriteRenderer(ECS ecs)
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
            FTransform transform = Component.Transform.coerce(t);
            sprite.transform.position.x = transform.pos_x();
            sprite.transform.position.y = transform.pos_y();
            sprite.transform.scale.x = transform.scale_x();
            sprite.transform.scale.y = transform.scale_y();
            this.add(sprite);
        }

        render();
    }
}