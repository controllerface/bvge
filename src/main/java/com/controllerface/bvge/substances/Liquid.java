package com.controllerface.bvge.substances;

import static com.controllerface.bvge.substances.Compound.*;

public enum Liquid
{
    WATER                 (H2O),
    BEER                  (H2O, HOP_MASH, SUCROSE, ETHANOL),
    WINE                  (H2O, SUCROSE, FRUCTOSE, ETHANOL),
    SEAWATER              (H2O, Compound.OCEANIC_IMPURITIES),
    PEROXIDE_DISENFECTANT (H2O, H2O2),
    RUBBING_ALCOHOL       (H2O, ISOPROPANOL),
    POOL_CLEANER          (H2O, SODIUM_HYPOCHLORITE),
    MURIATIC_ACID         (H2O, HYDROCHLORIC_ACID),
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
