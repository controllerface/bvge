package com.controllerface.bvge.ecs;


import com.controllerface.bvge.Component;
import com.controllerface.bvge.rendering.SpriteRenderer;

public enum ComponentType
{
    SpriteRenderer(SpriteRenderer.class),

    ;

    private final Class<? extends Component> _class;

    ComponentType(Class<? extends Component> aClass)
    {
        _class = aClass;
    }

    public <T extends Component> T coerce(Object componentClass)
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
