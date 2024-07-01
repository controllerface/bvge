package com.controllerface.bvge.window.events;

import java.util.*;

public class EventBus
{
    private final Map<EventType, List<Queue<Event>>> subscribers = Collections.synchronizedMap(new HashMap<>());

    public void register(Queue<Event> sink, EventType ... types)
    {
        if (types.length == 0) return;

        for (var type : types)
        {
            subscribers.computeIfAbsent(type, (_) -> new ArrayList<>()).add(sink);
        }
    }

    public void report_event(Event event)
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
