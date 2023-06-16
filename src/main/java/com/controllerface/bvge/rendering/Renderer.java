package com.controllerface.bvge.rendering;

import com.controllerface.bvge.GameObject;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Renderer
{
    private final int MAX_BATCH_SIZE = 1000;
    private List<SpriteRenderBatch> batches;

    private static Shader currentShader;

    public Renderer()
    {
        this.batches = new ArrayList<>();
    }

    public void add(GameObject go)
    {
        SpriteComponentOLD renderer = go.getComponent(SpriteComponentOLD.class);
        if (renderer != null)
        {
            add(renderer);
        }
    }

    private void add(SpriteComponentOLD sprite)
    {
        boolean added = false;
        for (SpriteRenderBatch batch : batches)
        {
            if (batch.hasRoom() && batch.zIndex() == sprite.gameObject.transform.zIndex)
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
            SpriteRenderBatch newBatch = new SpriteRenderBatch(MAX_BATCH_SIZE, sprite.gameObject.transform.zIndex);
            newBatch.start();
            batches.add(newBatch);
            newBatch.addSprite(sprite);
            Collections.sort(batches);
        }
    }

    public static void bindShader(Shader shader)
    {
        currentShader = shader;
    }

    public static Shader getBoundShader()
    {
        return currentShader;
    }

    public void render()
    {
        currentShader.use();
        for (SpriteRenderBatch batch : batches)
        {
            batch.render();
        }
    }
}