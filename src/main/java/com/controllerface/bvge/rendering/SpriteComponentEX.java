package com.controllerface.bvge.rendering;

import com.controllerface.bvge.Transform;
import com.controllerface.bvge.ecs.Component_EX;
import org.joml.Vector2f;
import org.joml.Vector4f;

// A basic wrapper that renders a sprite
public class SpriteComponentEX implements Component_EX
{
    private Vector4f color = new Vector4f(1,1,1,1);
    private Sprite sprite = new Sprite();
    public Transform transform = new Transform();
    private boolean isDirty = true;

    public Texture getTexture()
    {
        return sprite.getTexture();
    }

    public Vector2f[] getTexCoords()
    {
        return sprite.getTexCoords();
    }

    public Vector4f getColor()
    {
        return color;
    }

//    @Override
//    public void start()
//    {
//        this.lastTransform = gameObject.transform.copy();
//    }
//
//    @Override
//    public void update(float dt)
//    {
//        if (!this.lastTransform.equals(this.gameObject.transform))
//        {
//            this.gameObject.transform.copy(this.lastTransform);
//            isDirty = true;
//        }
//    }
//
//    @Override
//    public void imgui()
//    {
////        if (JimGui.colorPicker4("Color Picker", this.color))
////        {
////            this.isDirty = true;
////        }
//    }

    public void setSprite(Sprite sprite)
    {
        this.sprite = sprite;
        this.isDirty = true;
    }

    public void setColor(Vector4f color)
    {
        if (!this.color.equals(color))
        {
            this.isDirty = true;
            this.color.set(color);
        }
    }

    public boolean isDirty()
    {
        return this.isDirty;
    }

    public void setClean()
    {
        this.isDirty = false;
    }

    public void setTexture(Texture texture)
    {
        this.sprite.setTexture(texture);
    }
}
