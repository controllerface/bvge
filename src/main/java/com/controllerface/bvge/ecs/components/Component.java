package com.controllerface.bvge.ecs.components;

import com.controllerface.bvge.data.FBody2D;

public enum Component
{
    SpriteComponent(SpriteComponent.class),
    ControlPoints(ControlPoints.class),
    RigidBody2D(FBody2D.class),
    CameraFocus(CameraFocus.class),

    ;

    private final Class<? extends GameComponent> _class;

    Component(Class<? extends GameComponent> aClass)
    {
        _class = aClass;
    }

    public <T extends GameComponent> T coerce(Object componentClass)
    {
        assert componentClass != null : "Attempted p2 coerce null component";
        if (_class.isAssignableFrom(componentClass.getClass()))
        {
            @SuppressWarnings("unchecked")
            T t = (T) _class.cast(componentClass);
            return t;
        }
        assert false : "Attempted p2 coerce incompatible component";
        return null;
    }
}
