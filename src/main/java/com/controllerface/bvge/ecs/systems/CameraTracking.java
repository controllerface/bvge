package com.controllerface.bvge.ecs.systems;

import com.controllerface.bvge.cl.OpenCL;
import com.controllerface.bvge.data.HullIndex;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.Component;
import com.controllerface.bvge.ecs.systems.physics.SpatialPartition;
import com.controllerface.bvge.window.Window;

public class CameraTracking extends GameSystem
{
    private final SpatialPartition spatialPartition;
    public CameraTracking(ECS ecs, SpatialPartition spatialPartition)
    {
        super(ecs);
        this.spatialPartition = spatialPartition;
    }

    @Override
    public void run(float dt)
    {
        var focusTargets = ecs.getComponents(Component.CameraFocus);
        var focusTarget = focusTargets.entrySet().stream().findAny().orElseThrow();
        HullIndex hull = Component.RigidBody2D.forEntity(ecs, focusTarget.getKey());
        if (hull == null) return;

        // todo: change to armature
        float[] pos = OpenCL.read_position(hull.index());

        // will be null for the first few frames while the loop is primed
        if (pos == null) return;

        float pos_x = pos[0];
        float pos_y = pos[1];

        var camera = Window.get().camera();
        var width = (float)Window.get().getWidth() * camera.getZoom();
        var height = (float)Window.get().getHeight() * camera.getZoom();
        var new_x = (pos_x - width / 2);
        var new_y = (pos_y - height / 2);
        var new_origin_x = (pos_x - width / camera.getZoom()) + (spatialPartition.getWidth() / 2);
        var new_origin_y = (pos_y - height / camera.getZoom()) + (spatialPartition.getHeight() / 2);
        camera.position.x = new_x;
        camera.position.y = new_y;
        spatialPartition.updateOrigin(new_origin_x, new_origin_y);
    }

    @Override
    public void shutdown()
    {

    }
}
