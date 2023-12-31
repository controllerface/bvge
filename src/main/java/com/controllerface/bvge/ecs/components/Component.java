package com.controllerface.bvge.ecs.components;

import com.controllerface.bvge.ecs.ECS;

public enum Component
{
    ControlPoints(ControlPoints.class),
    Armature(ArmatureIndex.class),
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

    /**
     * Convenience method that reads and coerces a component from the ECS in a single call.
     *
     * @param ecs the ECS to read the component data from
     * @param entity the entity ID of the target entity
     * @return the extracted component, or null if the named component is not attached to the entity
     * @param <T> one of the component types mapped to a Component enum type defined in this class
     */
    public <T extends GameComponent> T forEntity(ECS ecs, String entity)
    {
        var componentClass = ecs.getComponentFor(entity, this);
        return componentClass == null ? null : coerce(componentClass);
    }
}
