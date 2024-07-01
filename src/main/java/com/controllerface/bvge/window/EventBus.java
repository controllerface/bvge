package com.controllerface.bvge.window;

import java.util.*;

public class EventBus
{
    private final Map<EventType, List<Queue<Event>>> subscribers = Collections.synchronizedMap(new HashMap<>());

    public interface Event { EventType type(); }
    public record WindowEvent(EventType type) implements Event { }
    public record InventoryEvent(EventType type) implements Event { }
    public record InputEvent(EventType type) implements Event { }
    public record MessageEvent(EventType type, String message) implements Event { }

    public enum EventType
    {
        WINDOW_RESIZE,
        ITEM_CHANGE,
        NEXT_ITEM,
        PREV_ITEM,
        ITEM_PLACING,
    }

    public static MessageEvent message(EventType type, String message)
    {
        return new MessageEvent(type, message);
    }

    public static WindowEvent window(EventType type)
    {
        return new WindowEvent(type);
    }

    public static InventoryEvent inventory(EventType type)
    {
        return new InventoryEvent(type);
    }

    public static InputEvent input(EventType type)
    {
        return new InputEvent(type);
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
