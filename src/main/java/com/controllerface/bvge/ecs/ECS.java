package com.controllerface.bvge.ecs;

import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.ecs.components.Component;
import com.controllerface.bvge.ecs.components.GameComponent;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class ECS
{
    private long count = 0;
    private final List<GameSystem> systems = new ArrayList<>();
    private final Map<Component, Map<String, GameComponent>> components = new HashMap<>();
    private final Set<String> entities = ConcurrentHashMap.newKeySet();

    public ECS()
    {
        // register all components at creation time. This is necessary to ensure all the
        // available components can be used before any systems or entities make use of them
        Arrays.stream(Component.values()).forEach(this::registerComponent);
    }

    private void registerComponent(Component component)
    {
        // todo: define a "defaults" system that maps component enum type to a factory
        //  of some kind that generates a proper initial or "empty" value to use here.
        //  Then optionally, calling code may chose to provide a pre-constructed object
        //  that can be used to override the default.
        components.put(component, new HashMap<>());
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


    public void registerSystem(GameSystem system)
    {
        systems.add(system);
    }

    public void attachComponent(String id, Component type, GameComponent component)
    {
        components.get(type).put(id, component);
    }

    public GameComponent getComponentFor(String id, Component type)
    {
        return components.get(type).get(id);
    }

    /**
     * Retrieve the Map of components (by entity) for a given component.
     *
     * @param type the type of component map p2 retrieve
     * @return the components map for the given component type. may be empty
     */
    public Map<String, GameComponent> getComponents(Component type)
    {
        return components.get(type);
    }

    /**
     * Each registered system should do dt amount of work
     *
     * @param dt amount of work time to spend
     */
    public void tick(float dt)
    {
        systems.forEach(system -> system.tick(dt));
    }

    public void shutdown()
    {
        systems.forEach(GameSystem::shutdown);
    }
}
