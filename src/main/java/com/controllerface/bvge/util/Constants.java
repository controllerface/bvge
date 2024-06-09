package com.controllerface.bvge.util;

public class Constants
{
    // todo: this class should probably contain more fields
    public static class Rendering
    {
        // todo: experiment with this, maybe make it configurable
        public static final int MAX_BATCH_SIZE = (256 * 256);
        public static final int SCALAR_LENGTH = 1;
        public static final int SCALAR_FLOAT_SIZE = Float.BYTES;
        public static final int SCALAR_INT_SIZE = Integer.BYTES;
        public static final int VECTOR_2D_LENGTH = 2; // x, y
        public static final int VECTOR_FLOAT_2D_SIZE = VECTOR_2D_LENGTH * Float.BYTES;
        public static final int VECTOR_4D_LENGTH = 4; // x, y, z, w
        public static final int VECTOR_FLOAT_4D_SIZE = VECTOR_4D_LENGTH * Float.BYTES;
    }

    public enum HullFlags
    {
        IS_STATIC     (0b0000_0000_0000_0000_0000_0000_0000_0001),
        IS_CIRCLE     (0b0000_0000_0000_0000_0000_0000_0000_0010),
        IS_POLYGON    (0b0000_0000_0000_0000_0000_0000_0000_0100),
        NO_BONES      (0b0000_0000_0000_0000_0000_0000_0000_1000),
        OUT_OF_BOUNDS (0b0000_0000_0000_0000_0000_0000_0001_0000),
        NON_COLLIDING (0b0000_0000_0000_0000_0000_0000_0010_0000),
        IS_FOOT       (0b0000_0000_0000_0000_0000_0000_0100_0000),
        SIDE_L        (0b0000_0000_0000_0000_0000_0000_1000_0000),
        SIDE_R        (0b0000_0000_0000_0000_0000_0001_0000_0000),
        IS_LIQUID     (0b0000_0000_0000_0000_0000_0010_0000_0000),
        IN_LIQUID     (0b0000_0000_0000_0000_0000_0100_0000_0000),
        TOUCH_ALIKE   (0b0000_0000_0000_0000_0000_1000_0000_0000),
        IS_BLOCK      (0b0000_0000_0000_0000_0001_0000_0000_0000),
        IS_ORGANIC    (0b0000_0000_0000_0000_0010_0000_0000_0000),
        IN_PERIMETER  (0b0000_0000_0000_0000_0100_0000_0000_0000),
        IS_CURSOR     (0b0000_0000_0000_0000_1000_0000_0000_0000),
        CURSOR_OVER   (0b0000_0000_0000_0001_0000_0000_0000_0000),
        IS_HAND       (0b0000_0000_0000_0010_0000_0000_0000_0000),
        IN_RANGE      (0b0000_0000_0000_0100_0000_0000_0000_0000),

        ;

        public final int _int;

        HullFlags(int _int)
        {
            this._int = _int;
        }
    }

    public enum EdgeFlags
    {
        IS_INTERIOR     (0b0000_0000_0000_0000_0000_0000_0000_0001),

        ;

        public final int bits;

        EdgeFlags(int bits)
        {
            this.bits = bits;
        }
    }

    public enum PointFlags
    {
        IS_INTERIOR     (0b0000_0000_0000_0000_0000_0000_0000_0001),
        HIT_FLOOR       (0b0000_0000_0000_0000_0000_0000_0000_0010),
        HIT_WALL        (0b0000_0000_0000_0000_0000_0000_0000_0100),
        FLOW_LEFT       (0b0000_0000_0000_0000_0000_0000_0000_1000),
        HIGH_DENSITY    (0b0000_0000_0000_0000_0000_0000_0001_0000),

        ;

        public final int bits;

        PointFlags(int bits)
        {
            this.bits = bits;
        }
    }

    public enum ControlFlags
    {
        LEFT   (0b0000_0000_0000_0000_0000_0000_0000_0001),
        RIGHT  (0b0000_0000_0000_0000_0000_0000_0000_0010),
        UP     (0b0000_0000_0000_0000_0000_0000_0000_0100),
        DOWN   (0b0000_0000_0000_0000_0000_0000_0000_1000),
        JUMP   (0b0000_0000_0000_0000_0000_0000_0001_0000),
        MOUSE1 (0b0000_0000_0000_0000_0000_0000_0010_0000),
        MOUSE2 (0b0000_0000_0000_0000_0000_0000_0100_0000),
        RUN    (0b0000_0000_0000_0000_0000_0000_1000_0000),

        ;

        public final int bits;

        ControlFlags(int bits)
        {
            this.bits = bits;
        }
    }
}
