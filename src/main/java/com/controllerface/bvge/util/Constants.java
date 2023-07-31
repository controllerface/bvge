package com.controllerface.bvge.util;

public class Constants
{
    // todo: this class should probably contain more fields
    public static class Rendering
    {
        // should equate to 256k per edge batch
        // todo: experiment with this, maybe make it configurable
        public static final int MAX_BATCH_SIZE = 16_000;

        public static final int VECTOR_2D_LENGTH = 2; // 2D vector; x,y
        public static final int VECTOR_FLOAT_2D_SIZE = VECTOR_2D_LENGTH * Float.BYTES;
        public static final int VECTOR_INT_2D_SIZE = VECTOR_2D_LENGTH * Integer.BYTES;

        public static final int VECTOR_4D_LENGTH = 2; // 3D vector; x,y,z,w
        public static final int VECTOR_FLOAT_4D_SIZE = VECTOR_4D_LENGTH * Float.BYTES;
        public static final int VECTOR_INT_4D_SIZE = VECTOR_4D_LENGTH * Integer.BYTES;
    }
}
