package com.controllerface.bvge.ecs.components;

public enum Component
{
    SpriteComponent(SpriteComponent.class),
    Transform(Transform.class),
    ControlPoints(com.controllerface.bvge.ecs.components.ControlPoints.class),
    RigidBody2D(com.controllerface.bvge.ecs.components.RigidBody2D.class),
    BoundingBox(QuadRectangle.class),

    ;

    private final Class<? extends GameComponent> _class;

    Component(Class<? extends GameComponent> aClass)
    {
        _class = aClass;
    }

    public <T extends GameComponent> T coerce(Object componentClass)
    {
        assert componentClass != null : "Attempted to coerce null component";
        if (componentClass == null) return null;
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
