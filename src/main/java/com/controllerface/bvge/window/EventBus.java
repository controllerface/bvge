package com.controllerface.bvge.window;

import java.util.*;

public class EventBus
{
    private final Map<EventType, List<Queue<Event>>> subscribers = Collections.synchronizedMap(new HashMap<>());

    public enum EventType
    {
        WINDOW_RESIZE,
        INVENTORY,
        NEXT_ITEM,
        PREV_ITEM,
        PLACING_ITEM,
    }

    public interface Event { EventType type(); }
    public record WindowEvent(EventType type) implements Event { };
    public record MessageEvent(EventType type, String message) implements Event { };

    public static MessageEvent msg(EventType type, String message)
    {
        return new MessageEvent(type, message);
    }

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
