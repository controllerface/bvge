package com.controllerface.bvge.ecs.components;

import com.controllerface.bvge.data.BodyIndex;
import com.controllerface.bvge.data.LinearForce;

public enum Component
{
    SpriteComponent(SpriteComponent.class),
    ControlPoints(ControlPoints.class),
    RigidBody2D(BodyIndex.class),
    LinearForce(LinearForce.class),
    CameraFocus(CameraFocus.class),

    ;

    private final Class<? extends GameComponent> _class;

    Component(Class<? extends GameComponent> aClass)
    {
        _class = aClass;
    }

    public <T extends GameComponent> T coerce(Object componentClass)
    {
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
