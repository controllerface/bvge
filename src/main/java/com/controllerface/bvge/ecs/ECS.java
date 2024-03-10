package com.controllerface.bvge.ecs;

import com.controllerface.bvge.ecs.components.Component;
import com.controllerface.bvge.ecs.components.GameComponent;
import com.controllerface.bvge.ecs.systems.GameSystem;

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
        for (Component component : Component.values())
        {
            registerComponent(component);
        }
    }

    private void registerComponent(Component component)
    {
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
     * @param type the type of component map to retrieve
     * @return the components map for the given component type. may be empty
     */
    public Map<String, GameComponent> getComponents(Component type)
    {
        return components.get(type);
    }

    /**
     * Each registered system should do an amount of work equal to the delta time (dt).
     *
     * @param dt amount of work time to spend
     */
    public void tick(float dt)
    {
        for (GameSystem system : systems)
        {
            system.tick(dt);
        }
    }

    public void shutdown()
    {
        for (GameSystem system : systems)
        {
            system.shutdown();
        }
    }
}
