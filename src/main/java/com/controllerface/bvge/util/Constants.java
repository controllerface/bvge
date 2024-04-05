package com.controllerface.bvge.util;

public class Constants
{
    // todo: this class should probably contain more fields
    public static class Rendering
    {
        // todo: experiment with this, maybe make it configurable
        public static final int MAX_BATCH_SIZE = 10_000;

        public static final int SCALAR_LENGTH = 1;
        public static final int SCALAR_FLOAT_SIZE = Float.BYTES;

        public static final int VECTOR_2D_LENGTH = 2; // 2D vector; x,y
        public static final int VECTOR_FLOAT_2D_SIZE = VECTOR_2D_LENGTH * Float.BYTES;
        public static final int VECTOR_INT_2D_SIZE = VECTOR_2D_LENGTH * Integer.BYTES;

        public static final int VECTOR_4D_LENGTH = 4; // 3D vector; x,y,z,w
        public static final int VECTOR_FLOAT_4D_SIZE = VECTOR_4D_LENGTH * Float.BYTES;
        public static final int VECTOR_INT_4D_SIZE = VECTOR_4D_LENGTH * Integer.BYTES;
    }

    public enum HullFlags
    {
        EMPTY         (0b0000_0000_0000_0000_0000_0000_0000_0000),
        IS_STATIC     (0b0000_0000_0000_0000_0000_0000_0000_0001),
        IS_CIRCLE     (0b0000_0000_0000_0000_0000_0000_0000_0010),
        IS_POLYGON    (0b0000_0000_0000_0000_0000_0000_0000_0100),
        NO_BONES      (0b0000_0000_0000_0000_0000_0000_0000_1000),
        OUT_OF_BOUNDS (0b0000_0000_0000_0000_0000_0000_0001_0000),
        NON_COLLIDING (0b0000_0000_0000_0000_0000_0000_0010_0000),

        ;

        public final int bits;

        HullFlags(int bits)
        {
            this.bits = bits;
        }
    }

    public enum ControlFlags
    {
        LEFT  (0b0000_0000_0000_0000_0000_0000_0000_0001),
        RIGHT (0b0000_0000_0000_0000_0000_0000_0000_0010),
        UP    (0b0000_0000_0000_0000_0000_0000_0000_0100),
        DOWN  (0b0000_0000_0000_0000_0000_0000_0000_1000),
        JUMP  (0b0000_0000_0000_0000_0000_0000_0001_0000),

        ;

        public final int bits;

        ControlFlags(int bits)
        {
            this.bits = bits;
        }
    }
}
