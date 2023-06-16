package com.controllerface.bvge.ecs;


import com.controllerface.bvge.TransformEX;
import com.controllerface.bvge.rendering.SpriteComponentEX;

public enum Component
{
    SpriteComponent(SpriteComponentEX.class),
    Transform(TransformEX.class),
    ControlPoints(ControlPoints.class),
    RigidBody2D(RigidBody2D.class),

    ;

    private final Class<? extends Component_EX> _class;

    Component(Class<? extends Component_EX> aClass)
    {
        _class = aClass;
    }

    public <T extends Component_EX> T coerce(Object componentClass)
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
