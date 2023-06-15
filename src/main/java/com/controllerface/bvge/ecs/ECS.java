package com.controllerface.bvge.ecs;

import com.controllerface.bvge.Component;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

public class ECS
{
    private long count = 0;
    private List<System> systems = new CopyOnWriteArrayList<>();
    private Map<ComponentType, Map<String, Component>> components = new ConcurrentHashMap<>();
    private Set<String> entities = ConcurrentHashMap.newKeySet();

    public ECS()
    {
        // register all components
        Arrays.stream(ComponentType.values())
            .forEach(this::registerComponent);
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

    private void registerComponent(ComponentType componentType)
    {
        components.put(componentType, new HashMap<>());
    }

    public void registerSystem(System system)
    {
        system.setup(this);
        systems.add(system);
    }

    public void attachComponent(String id, ComponentType type, Component component)
    {
        components.get(type).put(id, component);
    }

    public Component getComponentFor(String id, ComponentType type)
    {
        return components.get(type).get(id);
    }

    public Map<String, Component> getComponents(ComponentType type)
    {
        return components.get(type);
    }

    public void run(float dt)
    {
        systems.forEach(system_ -> system_.run(dt));
    }
}
