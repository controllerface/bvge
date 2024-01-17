package com.controllerface.bvge.util;

public class Constants
{
    // todo: this class should probably contain more fields
    public static class Rendering
    {
        // todo: experiment with this, maybe make it configurable
        public static final int MAX_BATCH_SIZE = 10_000;

        public static final int VECTOR_2D_LENGTH = 2; // 2D vector; x,y
        public static final int VECTOR_FLOAT_2D_SIZE = VECTOR_2D_LENGTH * Float.BYTES;
        public static final int VECTOR_INT_2D_SIZE = VECTOR_2D_LENGTH * Integer.BYTES;

        public static final int VECTOR_4D_LENGTH = 4; // 3D vector; x,y,z,w
        public static final int VECTOR_FLOAT_4D_SIZE = VECTOR_4D_LENGTH * Float.BYTES;
        public static final int VECTOR_INT_4D_SIZE = VECTOR_4D_LENGTH * Integer.BYTES;
    }

    public enum HullFlags
    {
        EMPTY         (0x0000),
        IS_STATIC     (0x0001),
        IS_CIRCLE     (0x0002),
        IS_POLYGON    (0x0004),
        NO_BONES      (0x0008),
        OUT_OF_BOUNDS (0x0010),

        ;

        public final int bits;

        HullFlags(int bits)
        {
            this.bits = bits;
        }
    }
}
