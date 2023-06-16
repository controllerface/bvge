package com.controllerface.bvge.ecs;

public interface SystemEX
{
    default void setup(ECS ecs){}

    void run(float dt);
}
