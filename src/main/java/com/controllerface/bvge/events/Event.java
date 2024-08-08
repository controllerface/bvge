package com.controllerface.bvge.events;

import com.controllerface.bvge.substances.Solid;

public interface Event
{
    Type type();

    enum Type
    {
        WINDOW_RESIZE,
        ITEM_CHANGE,
        NEXT_ITEM,
        PREV_ITEM,
        ITEM_PLACING,
        SELECT_BLOCK,
    }

    record Input(Type type)                    implements Event {}
    record Inventory(Type type)                implements Event {}
    record Message(Type type, String message)  implements Event {}
    record SelectBlock(Type type, Solid solid) implements Event {}
    record Window(Type type)                   implements Event {}

    static Input input(Type type)                         { return new Input(type); }
    static Inventory inventory(Type type)                 { return new Inventory(type); }
    static Message message(Type type, String message)     { return new Message(type, message); }
    static SelectBlock select_block(Solid solid)          { return new SelectBlock(Type.SELECT_BLOCK, solid); }
    static Window window(Type type)                       { return new Window(type); }
}
