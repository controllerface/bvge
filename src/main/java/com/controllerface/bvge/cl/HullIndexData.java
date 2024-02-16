package com.controllerface.bvge.cl;

import org.jocl.cl_mem;

/**
 * A container for results of a model id query. Typically used when CPU code queries the GPU
 * for the hull indices of all objects with a given model ID. This is useful for rendering
 * multiple instances of a single mode.
 *
 * @param indices a CL memory buffer that contains the indices of matching hulls
 * @param count the number of hull indices that are stored in the buffer
 */
public record HullIndexData(long indices, int count) { }
