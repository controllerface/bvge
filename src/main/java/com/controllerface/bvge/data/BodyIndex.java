package com.controllerface.bvge.data;

import com.controllerface.bvge.ecs.components.GameComponent;

/*
 * Memory layout: float16
 *  0: x position                 (transform)
 *  1: y position                 (transform)
 *  2: scale x                    (transform)
 *  3: scale y                    (transform)
 *  4: acceleration x component
 *  5: acceleration y component
 *  6: collision flags            (int cast)
 *  7: start point index          (int cast)
 *  8: end point index            (int cast)
 *  9: start edge index           (int cast)
 * 10: end edge index             (int cast)
 * 11:
 * 12:
 * 13:
 * 14:
 * 15:
 *  */

public record BodyIndex(int index) implements GameComponent { }
