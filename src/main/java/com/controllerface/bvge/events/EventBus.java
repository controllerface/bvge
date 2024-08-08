package com.controllerface.bvge.events;

import java.util.*;

public class EventBus
{
    private final Map<Event.Type, List<Queue<Event>>> subscribers = Collections.synchronizedMap(new HashMap<>());

    public void register(Queue<Event> sink, Event.Type ... types)
    {
        for (var type : types)
        {
            subscribers.computeIfAbsent(type, (_) -> new ArrayList<>()).add(sink);
        }
    }

    public void emit_event(Event event)
    {
        var sinks = subscribers.get(event.type());
        if (sinks != null)
        {
            for (var sink : sinks)
            {
                sink.add(event);
            }
        }
    }
}
