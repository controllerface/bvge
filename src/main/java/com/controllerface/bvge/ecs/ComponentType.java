package com.controllerface.bvge.ecs;


import com.controllerface.bvge.TransformEX;
import com.controllerface.bvge.rendering.SpriteComponentEX;

public enum ComponentType
{
    SpriteComponent(SpriteComponentEX.class),
    Transform(TransformEX.class)

    ;

    private final Class<? extends Component_EX> _class;

    ComponentType(Class<? extends Component_EX> aClass)
    {
        _class = aClass;
    }

    public <T extends Component_EX> T coerce(Object componentClass)
    {
        if (_class.isAssignableFrom(componentClass.getClass()))
        {
            @SuppressWarnings("unchecked")
            T t = (T) _class.cast(componentClass);
            return t;
        }
        return null;
    }
}
