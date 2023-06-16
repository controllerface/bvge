package com.controllerface.bvge;

import com.controllerface.bvge.ecs.Component_EX;
import org.joml.Vector2f;

public class TransformEX implements Component_EX
{

    public Vector2f position;
    public Vector2f scale;
    public float rotation = 0.0f;
    public int zIndex;

    public TransformEX() {
        init(new Vector2f(), new Vector2f());
    }

    public TransformEX(Vector2f position) {
        init(position, new Vector2f());
    }

    public TransformEX(Vector2f position, Vector2f scale) {
        init(position, scale);
    }

    public void init(Vector2f position, Vector2f scale) {
        this.position = position;
        this.scale = scale;
        this.zIndex = 0;
    }

    public TransformEX copy() {
        return new TransformEX(new Vector2f(this.position), new Vector2f(this.scale));
    }


    public void copy(TransformEX to) {
        to.position.set(this.position);
        to.scale.set(this.scale);
    }

    @Override
    public boolean equals(Object o) {
        if (o == null) return false;
        if (!(o instanceof TransformEX)) return false;

        TransformEX t = (TransformEX)o;
        return t.position.equals(this.position) && t.scale.equals(this.scale) &&
            t.rotation == this.rotation && t.zIndex == this.zIndex;
    }
}