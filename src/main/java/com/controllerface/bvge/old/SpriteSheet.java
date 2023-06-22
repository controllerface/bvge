package com.controllerface.bvge.old;

import com.controllerface.bvge.ecs.Sprite;
import com.controllerface.bvge.gl.Texture;
import org.joml.Vector2f;

import java.util.ArrayList;
import java.util.List;

public class SpriteSheet
{
    private Texture texture;
    private List<Sprite> sprites;

    public SpriteSheet(Texture texture, int width, int height, int numSprites, int spacing)
    {
        this.sprites = new ArrayList<>();
        this.texture = texture;

        int currentX = 0;
        int currentY = texture.getHeight() - height;
        for (int i = 0; i < numSprites; i++)
        {
            float topY = (currentY + height) / (float) texture.getHeight();
            float rightX = (currentX + width) / (float) texture.getWidth();
            float leftX = currentX / (float) texture.getWidth();
            float bottomY = currentY / (float) texture.getHeight();

            Vector2f[] texCoords =
                {
                    new Vector2f(rightX, topY),
                    new Vector2f(rightX, bottomY),
                    new Vector2f(leftX, bottomY),
                    new Vector2f(leftX, topY),
                };

            Sprite sprite = new Sprite();
            sprite.setTexture(this.texture);
            sprite.setTexCoords(texCoords);
            sprite.setWidth(width);
            sprite.setHeight(height);
            this.sprites.add(sprite);
            currentX += width + spacing;
            if (currentX >= texture.getWidth())
            {
                currentX = 0;
                currentY -= height + spacing;
            }
        }
    }

    public Sprite getSprite(int index)
    {
        return this.sprites.get(index);
    }

    public int size()
    {
        return sprites.size();
    }
}

