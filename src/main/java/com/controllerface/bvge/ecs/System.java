package com.controllerface.bvge.ecs;

public interface System
{
    default void setup(ECS ecs){}

    void run(float dt);
}
