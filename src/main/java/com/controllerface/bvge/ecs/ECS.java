package com.controllerface.bvge.ecs;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

public class ECS
{
    private long count = 0;
    private List<SystemEX> systemEXES = new CopyOnWriteArrayList<>();
    private Map<ComponentType, Map<String, Component_EX>> components = new ConcurrentHashMap<>();
    private Set<String> entities = ConcurrentHashMap.newKeySet();

    public ECS()
    {
        // register all components at creation time. This is necessary to ensure all the
        // available components can be used before any systems or entities make use of them
        Arrays.stream(ComponentType.values()).forEach(this::registerComponent);
    }

    private void registerComponent(ComponentType componentType)
    {
        // todo: define a "defaults" system that maps component enum type to a factory
        //  of some kind that generates a proper initial or "empty" value to use here.
        //  Then optionally, calling code may chose to provide a pre-constructed object
        //  that can be used to override the default.
        components.put(componentType, new HashMap<>());
    }


    public String registerEntity()
    {
        return registerEntity(null);
    }

    public String registerEntity(String id)
    {
        var targetId = id;
        if (id == null || id.isEmpty())
        {
            targetId = "entity_" + count++;
        }
        entities.add(targetId);
        return targetId;
    }


    public void registerSystem(SystemEX systemEX)
    {
        systemEX.setup(this);
        systemEXES.add(systemEX);
    }

    public void attachComponent(String id, ComponentType type, Component_EX component)
    {
        components.get(type).put(id, component);
    }

    public Component_EX getComponentFor(String id, ComponentType type)
    {
        return components.get(type).get(id);
    }

    /**
     * Retrieve the Map of components (by entity) for a given component.
     *
     * @param type the type of component map to retrieve
     * @return the components map for the given component type. may be empty
     */
    public Map<String, Component_EX> getComponents(ComponentType type)
    {
        return components.get(type);
    }

    public void run(float dt)
    {
        systemEXES.forEach(system_EX_ -> system_EX_.run(dt));
    }
}
