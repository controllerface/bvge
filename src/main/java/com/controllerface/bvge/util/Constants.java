package com.controllerface.bvge.util;

public class Constants
{
    // todo: this class should probably contain more fields
    public static class Rendering
    {
        // should equate to 256k per edge batch
        // todo: experiment with this, maybe make it configurable
        public static final int MAX_BATCH_SIZE = 16_000;
    }
}
