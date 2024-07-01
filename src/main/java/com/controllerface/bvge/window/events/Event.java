package com.controllerface.bvge.window.events;

public sealed interface Event permits
    Event.Input,
    Event.Inventory,
    Event.Message,
    Event.Window
{
    EventType type();

    record Input(EventType type) implements Event {}
    record Inventory(EventType type) implements Event {}
    record Message(EventType type, String message) implements Event {}
    record Window(EventType type) implements Event {}

    static Input input(EventType type)
    {
        return new Input(type);
    }

    static Inventory inventory(EventType type)
    {
        return new Inventory(type);
    }

    static Message message(EventType type, String message)
    {
        return new Message(type, message);
    }

    static Window window(EventType type)
    {
        return new Window(type);
    }
}
