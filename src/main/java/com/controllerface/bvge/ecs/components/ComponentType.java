package com.controllerface.bvge.ecs.components;

import com.controllerface.bvge.ecs.ECS;

public enum ComponentType
{
    Position      (Position.class),
    EntityId      (EntityIndex.class),
    MouseCursorId (EntityIndex.class),
    BlockCursorId (EntityIndex.class),
    MovementForce (FloatValue.class),
    JumpForce     (FloatValue.class),
    InputState    (InputState.class),
    BlockCursor   (BlockCursor.class),

    ;

    private final Class<? extends GameComponent> _class;

    ComponentType(Class<? extends GameComponent> aClass)
    {
        _class = aClass;
    }

    private <T extends GameComponent> T coerce(Object componentClass)
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
        var componentClass = ecs.get_component_for(entity, this);
        return componentClass == null ? null : coerce(componentClass);
    }
}
