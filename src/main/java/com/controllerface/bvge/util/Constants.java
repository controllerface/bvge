package com.controllerface.bvge.util;

import java.util.Arrays;
import java.util.stream.Collectors;

public class Constants
{
    public static final String PLAYER_ID = "player";


    // todo: experiment with this, maybe make it configurable
    private static final int BUFFER_BASE_SIZE = 256;
    private static final int BUFFER_MULTIPLIER = 4;

    public static class Rendering
    {
        public static final int MAX_BATCH_SIZE = (BUFFER_BASE_SIZE * BUFFER_BASE_SIZE) * BUFFER_MULTIPLIER;
        public static final int SCALAR_LENGTH = 1;
        public static final int SCALAR_FLOAT_SIZE = Float.BYTES;
        public static final int SCALAR_INT_SIZE = Integer.BYTES;
        public static final int VECTOR_2D_LENGTH = 2; // x, y
        public static final int VECTOR_FLOAT_2D_SIZE = VECTOR_2D_LENGTH * Float.BYTES;
        public static final int VECTOR_4D_LENGTH = 4; // x, y, z, w
        public static final int VECTOR_FLOAT_4D_SIZE = VECTOR_4D_LENGTH * Float.BYTES;
        public static final int MATRIX_4_LENGTH = 4; // 4x4 matrix
        public static final int MATRIX_FLOAT_4_SIZE = MATRIX_4_LENGTH * Float.BYTES;
    }

    public enum HullFlags
    {
        IS_STATIC     (0b00000000000000000000000000000001),
        IS_CIRCLE     (0b00000000000000000000000000000010),
        IS_POLYGON    (0b00000000000000000000000000000100),
        NO_BONES      (0b00000000000000000000000000001000),
        OUT_OF_BOUNDS (0b00000000000000000000000000010000),
        IS_SENSOR     (0b00000000000000000000000000100000),
        IS_FOOT       (0b00000000000000000000000001000000),
        SIDE_L        (0b00000000000000000000000010000000),
        SIDE_R        (0b00000000000000000000000100000000),
        IS_LIQUID     (0b00000000000000000000001000000000),
        IN_LIQUID     (0b00000000000000000000010000000000),
        TOUCH_ALIKE   (0b00000000000000000000100000000000),
        IS_BLOCK      (0b00000000000000000001000000000000),
        IS_ORGANIC    (0b00000000000000000010000000000000),
        IN_PERIMETER  (0b00000000000000000100000000000000),
        IS_CURSOR     (0b00000000000000001000000000000000),
        CURSOR_OVER   (0b00000000000000010000000000000000),
        IS_HAND       (0b00000000000000100000000000000000),
        IN_RANGE      (0b00000000000001000000000000000000),
        CURSOR_HIT    (0b00000000000010000000000000000000),
        GHOST_HULL    (0b00000000000100000000000000000000),
        IS_HEAD       (0b00000000001000000000000000000000),
        SENSOR_HIT    (0b00000000010000000000000000000000),

        ;

        public final int bits;

        HullFlags(int bits)
        {
            this.bits = bits;
        }
    }

    public enum EdgeFlags
    {
        E_INTERIOR  (0b00000000000000000000000000000001),
        SENSOR_EDGE(0b00000000000000000000000000000010),

        ;

        public final int bits;

        EdgeFlags(int bits)
        {
            this.bits = bits;
        }
    }

    public enum PointFlags
    {
        P_INTERIOR      (0b00000000000000000000000000000001),
        HIT_FLOOR       (0b00000000000000000000000000000010),
        HIT_WALL        (0b00000000000000000000000000000100),
        FLOW_LEFT       (0b00000000000000000000000000001000),
        HIGH_DENSITY    (0b00000000000000000000000000010000),

        ;

        public final int bits;

        PointFlags(int bits)
        {
            this.bits = bits;
        }
    }

    public enum EntityFlags
    {
        DELETED      (0b00000000000000000000000000000001),
        CAN_JUMP     (0b00000000000000000000000000000010),
        FACE_LEFT    (0b00000000000000000000000000000100),
        IS_WET       (0b00000000000000000000000000001000),
        SECTOR_OUT   (0b00000000000000000000000000010000),
        ATTACKING    (0b00000000000000000000000000100000),
        BROKEN       (0b00000000000000000000000001000000),
        CAN_COLLECT  (0b00000000000000000000000010000000),
        COLLECTED    (0b00000000000000000000000100000000),
        COLLECTABLE  (0b00000000000000000000001000000000),
        GHOST_ACTIVE (0b00000000000000000000010000000000),

        ;

        public final int bits;

        EntityFlags(int bits)
        {
            this.bits = bits;
        }
    }

    public static String hull_flags_src()
    {
        return Arrays.stream(Constants.HullFlags.values())
            .map(v->"#define " + v.name() + " 0b" + Integer.toBinaryString(v.bits))
            .collect(Collectors.joining("\n", "", "\n"));
    }

    public static String entity_flags_src()
    {
        return Arrays.stream(Constants.EntityFlags.values())
            .map(v->"#define " + v.name() + " 0b" + Integer.toBinaryString(v.bits))
            .collect(Collectors.joining("\n", "", "\n"));
    }

    public static String point_flags_src()
    {
        return Arrays.stream(Constants.PointFlags.values())
            .map(v->"#define " + v.name() + " 0b" + Integer.toBinaryString(v.bits))
            .collect(Collectors.joining("\n", "", "\n"));
    }

    public static String edge_flags_src()
    {
        return Arrays.stream(Constants.EdgeFlags.values())
            .map(v->"#define " + v.name() + " 0b" + Integer.toBinaryString(v.bits))
            .collect(Collectors.joining("\n", "", "\n"));
    }
}
