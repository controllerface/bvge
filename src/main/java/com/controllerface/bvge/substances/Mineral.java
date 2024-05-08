package com.controllerface.bvge.substances;

import static com.controllerface.bvge.substances.Compound.*;

public enum Mineral
{
    NOTHING          (Compound.NOTHING),
    ANDESITE         (PLAGIOCLASE_FELDSPAR, PYROXENE, AMPHIBOLE, BIOTITE),
    BASALT           (PLAGIOCLASE_FELDSPAR, PYROXENE, OLIVINE, MAGNETITE),
    BLUESCHIST       (GLAUCOPHANE, EPIDOTE, ACTINOLITE, CHLORITE),
    CARBONATITE      (CALCITE, Compound.DOLOMITE, MAGNETITE, APATITE),
    CHALK            (CALCITE),
    CLAYSTONE        (KAOLINITE, ILLITE),
    COAL             (CARBON, PYRITE, KAOLINITE),
    DIORITE          (PLAGIOCLASE_FELDSPAR, BIOTITE, AMPHIBOLE, QUARTZ),
    DOLOMITE         (Compound.DOLOMITE, CALCITE, QUARTZ),
    FLINT            (CHERT, QUARTZ, CHALCEDONY, CALCITE),
    GNEISS           (QUARTZ, FELDSPAR, BIOTITE, GARNET),
    GRANITE          (QUARTZ, FELDSPAR, MICA, HORNBLENDE),
    GREENSCHIST      (CHLORITE, ACTINOLITE, EPIDOTE, ALBITE),
    LIMESTONE        (CALCITE, Compound.DOLOMITE, QUARTZ),
    MARBLE           (CALCITE, Compound.DOLOMITE, QUARTZ, GRAPHITE),
    MUDSTONE         (ILLITE, SMECTITE),
    MUGEARITE        (NEPHELINE, AUGITE, APATITE, MAGNETITE),
    OBSIDIAN         (GLASS, FELDSPAR, MAGNETITE, BIOTITE),
    PERIDOTITE       (OLIVINE, PYROXENE, SERPENTINE, MAGNETITE),
    PUMICE           (GLASS, FELDSPAR, MAGNETITE, MICA),
    QUARTZ_DIORITE   (QUARTZ, PLAGIOCLASE_FELDSPAR, BIOTITE, AMPHIBOLE),
    QUARTZ_MONZONITE (QUARTZ, PLAGIOCLASE_FELDSPAR, POTASSIUM_FELDSPAR, BIOTITE),
    QUARTZITE        (QUARTZ, FELDSPAR, MICA, MAGNETITE),
    SANDSTONE        (QUARTZ, FELDSPAR, CALCITE),
    SCHIST           (BIOTITE, MUSCOVITE, QUARTZ, GARNET),
    SHALE            (ILLITE, KAOLINITE, FELDSPAR, CALCITE),
    SLATE            (CHLORITE, MUSCOVITE, FELDSPAR, GRAPHITE),
    SOAPSTONE        (TALC, CHLORITE, MAGNESITE, MAGNETITE),
    SYLVINITE        (SYLVITE, HALITE, CARNALLITE, ANHYDRITE),
    TALC_CARBONATE   (TALC, MAGNESITE, Compound.DOLOMITE, QUARTZ),
    TUFF             (QUARTZ, FELDSPAR, BIOTITE, HORNBLENDE),
    WHITESCHIST      (QUARTZ, FELDSPAR, MICA, CHLORITE),

    ;

    public final int mineral_number;
    public final Compound[] compounds;

    Mineral(Compound[] compounds)
    {
        this.mineral_number = this.ordinal();
        this.compounds = compounds;
    }

    Mineral(Compound compound_1)
    {
        this(new Compound[]{compound_1, Compound.NOTHING, Compound.NOTHING, Compound.NOTHING});
    }

    Mineral(Compound compound_1, Compound compound_2)
    {
        this(new Compound[]{compound_1, compound_2, Compound.NOTHING, Compound.NOTHING});
    }

    Mineral(Compound compound_1, Compound compound_2, Compound compound_3)
    {
        this(new Compound[]{compound_1, compound_2,compound_3, Compound.NOTHING});
    }

    Mineral(Compound compound_1, Compound compound_2, Compound compound_3, Compound compound_4)
    {
        this(new Compound[]{compound_1, compound_2, compound_3, compound_4});
    }
}
