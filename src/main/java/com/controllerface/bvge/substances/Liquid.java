package com.controllerface.bvge.substances;

import org.joml.Vector4f;

import static com.controllerface.bvge.substances.Compound.*;

public enum Liquid
{
    WATER                 (new Vector4f(0.000f, 0.048f, 0.494f, 0.250f), H2O),
    BEER                  (new Vector4f(0.839f, 0.588f ,0.114f, 0.250f), H2O, HOP_MASH, ETHANOL),
    WINE                  (new Vector4f(0.251f, 0.012f, 0.012f, 0.250f), H2O, SUCROSE, FRUCTOSE, ETHANOL),
    SEAWATER              (new Vector4f(0.000f, 0.100f, 0.200f, 0.250f), H2O, Compound.OCEANIC_IMPURITIES),
    PEROXIDE_DISENFECTANT (new Vector4f(0.949f, 0.949f, 0.949f, 0.250f), H2O, H2O2),
    RUBBING_ALCOHOL       (new Vector4f(0.100f, 0.100f, 0.100f, 0.250f), H2O, ISOPROPANOL),
    POOL_CLEANER          (new Vector4f(0.294f, 0.294f, 0.394f, 0.250f), H2O, SODIUM_HYPOCHLORITE),
    MURIATIC_ACID         (new Vector4f(0.200f, 0.198f, 0.182f, 0.250f), H2O, HYDROCHLORIC_ACID),
    GASOLINE              (new Vector4f(0.500f, 0.406f, 0.000f, 0.250f), OCTANE, BENZENE, XYLENE, ETHANOL),

    ;

    public final Vector4f color;
    public final byte liquid_number;
    public final Compound[] compounds;

    Liquid(Vector4f color, Compound[] compounds)
    {
        this.color = color;
        this.liquid_number = (byte) this.ordinal();
        this.compounds = compounds;
    }

    Liquid(Vector4f color, Compound compound_1)
    {
        this(color, new Compound[]{compound_1, NOTHING, NOTHING, NOTHING});
    }

    Liquid(Vector4f color, Compound compound_1, Compound compound_2)
    {
        this(color, new Compound[]{compound_1, compound_2, NOTHING, NOTHING});
    }

    Liquid(Vector4f color, Compound compound_1, Compound compound_2, Compound compound_3)
    {
        this(color, new Compound[]{compound_1, compound_2, compound_3, NOTHING});
    }

    Liquid(Vector4f color, Compound compound_1, Compound compound_2, Compound compound_3, Compound compound_4)
    {
        this(color, new Compound[]{compound_1, compound_2, compound_3, compound_4});
    }

    private static String lookup_table = "";

    /**
     * Generates an Open CL C lookup table for the colors of all defined liquids. The generated table will look
     * similar to this example at runtime:
     *
     *     constant float4 liquid_lookup_table[] =
     *     {
     * 	     (float4)(0.000, 0.694, 0.078, 0.25),   // liquid 0 color
     * 	     // ... other colors go here ...
     * 	     (float4)(1.000, 0.812, 0.000, 0.65),   // liquid n color
     *     };
     *
     * Each ordinal of the enum is mapped to the color in this lookup table so an Open CL kernel can easily choose the
     * correct color value to pass to the renderer when rendering the liquid.
     * @return a String containing the generated lookup table.
     */
    public static String cl_lookup_table()
    {
        if (lookup_table.isEmpty())
        {
            var buffer = new StringBuilder();

            buffer.append("constant float4 liquid_lookup_table[] = \n{\n");
            for (var liquid : values())
            {
                var x = String.valueOf(liquid.color.x);
                var y = String.valueOf(liquid.color.y);
                var z = String.valueOf(liquid.color.z);
                var w = String.valueOf(liquid.color.w);
                buffer.append(STR."\t(float4)(\{x}, \{y}, \{z}, \{w}),\n");
            }
            buffer.append("};\n\n");
            lookup_table = buffer.toString();
        }
        return lookup_table;
    }
}
