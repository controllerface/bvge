package com.controllerface.bvge.util;

public class Constants
{
    // todo: this class should probably contain more fields
    public static class Rendering
    {
        // should equate to 256k per edge batch
        // todo: experiment with this, maybe make it configurable
        public static final int MAX_BATCH_SIZE = 5_000;

        public static final int VECTOR_2D_LENGTH = 2; // 2D vector; x,y
        public static final int VECTOR_FLOAT_2D_SIZE = VECTOR_2D_LENGTH * Float.BYTES;
        public static final int VECTOR_INT_2D_SIZE = VECTOR_2D_LENGTH * Integer.BYTES;

        public static final int VECTOR_4D_LENGTH = 4; // 3D vector; x,y,z,w
        public static final int VECTOR_FLOAT_4D_SIZE = VECTOR_4D_LENGTH * Float.BYTES;
        public static final int VECTOR_INT_4D_SIZE = VECTOR_4D_LENGTH * Integer.BYTES;
    }

    public enum HullFlags
    {
        EMPTY(0x00),
        IS_STATIC(0x01),
        IS_CIRCLE(0x02),
        IS_POLYGON(0x04),
        NO_BONES(0x08),
        OUT_OF_BOUNDS(0x16),

        ;

        public final int bits;

        HullFlags(int bits)
        {
            this.bits = bits;
        }
    }
}
