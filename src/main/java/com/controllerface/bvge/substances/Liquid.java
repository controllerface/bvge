package com.controllerface.bvge.substances;

import static com.controllerface.bvge.substances.Compound.*;

public enum Liquid
{
    WATER                 (Compound.WATER),
    BEER                  (Compound.WATER, HOP_MASH, SUCROSE, ETHANOL),
    WINE                  (Compound.WATER, SUCROSE, FRUCTOSE, ETHANOL),
    SEAWATER              (Compound.WATER, Compound.OCEANIC_IMPURITIES),
    PEROXIDE_DISENFECTANT (Compound.WATER, Compound.PEROXIDE),
    RUBBING_ALCOHOL       (Compound.WATER, ISOPROPANOL),
    POOL_CHLORINE         (Compound.WATER, SODIUM_HYPOCHLORITE),
    MURIATIC_ACID         (Compound.WATER, HYDROCHLORIC_ACID),
    GASOLINE              (OCTANE, BENZENE, XYLENE, ETHANOL),

    ;

    public final byte liquid_number;
    public final Compound[] compounds;

    Liquid(Compound[] compounds)
    {
        this.liquid_number = (byte) this.ordinal();
        this.compounds = compounds;
    }

    Liquid(Compound compound_1)
    {
        this(new Compound[]{compound_1, NOTHING, NOTHING, NOTHING});
    }

    Liquid(Compound compound_1, Compound compound_2)
    {
        this(new Compound[]{compound_1, compound_2, NOTHING, NOTHING});
    }

    Liquid(Compound compound_1, Compound compound_2, Compound compound_3)
    {
        this(new Compound[]{compound_1, compound_2, compound_3, NOTHING});
    }

    Liquid(Compound compound_1, Compound compound_2, Compound compound_3, Compound compound_4)
    {
        this(new Compound[]{compound_1, compound_2, compound_3, compound_4});
    }
}
