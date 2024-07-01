package com.controllerface.bvge.window.events;

public sealed interface Event permits
    Event.Input,
    Event.Inventory,
    Event.Message,
    Event.Window
{
    Type type();

    enum Type
    {
        WINDOW_RESIZE,
        ITEM_CHANGE,
        NEXT_ITEM,
        PREV_ITEM,
        ITEM_PLACING,
    }

    record Input(Type type)                           implements Event {}
    record Inventory(Type type)                       implements Event {}
    record Message(Type type, String message)         implements Event {}
    record Window(Type type)                          implements Event {}

    static Input input(Type type)                     { return new Input(type); }
    static Inventory inventory(Type type)             { return new Inventory(type); }
    static Message message(Type type, String message) { return new Message(type, message); }
    static Window window(Type type)                   { return new Window(type); }
}
