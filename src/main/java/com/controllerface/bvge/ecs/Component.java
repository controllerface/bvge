package com.controllerface.bvge.ecs;

import com.controllerface.bvge.rendering.Line2D;
import com.controllerface.bvge.rendering.SpriteComponent;
import com.controllerface.bvge.Transform;

public enum Component
{
    SpriteComponent(SpriteComponent.class),
    Transform(Transform.class),
    ControlPoints(ControlPoints.class),
    RigidBody2D(RigidBody2D.class),

    ;

    private final Class<? extends GameComponent> _class;

    Component(Class<? extends GameComponent> aClass)
    {
        _class = aClass;
    }

    public <T extends GameComponent> T coerce(Object componentClass)
    {
        if (componentClass == null) return null;
        assert componentClass != null : "Attempted to coerce null component";
        if (_class.isAssignableFrom(componentClass.getClass()))
        {
            @SuppressWarnings("unchecked")
            T t = (T) _class.cast(componentClass);
            return t;
        }
        assert false : "Attempted to coerce incompatible component";
        return null;
    }
}
