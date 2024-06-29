package com.controllerface.bvge.window;

import java.util.*;

public class EventBus
{
    private Map<EventType, List<Queue<EventType>>> subscribers = Collections.synchronizedMap(new HashMap<>());

    public void register(Queue<EventType> sink, EventType ... types)
    {
        if (types.length == 0) return;

        for (var type : types)
        {
            subscribers.computeIfAbsent(type, (_) -> new ArrayList<>()).add(sink);
        }
    }

    public void report_event(EventType eventType)
    {
        var sinks = subscribers.get(eventType);
        if (sinks != null)
        {
            for (var sink : sinks)
            {
                sink.add(eventType);
            }
        }
    }
}
