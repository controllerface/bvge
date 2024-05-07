package com.controllerface.bvge.substances;

import static com.controllerface.bvge.substances.Compound.*;

public enum Mineral
{
    NOTHING          ((byte) 0, Compound.NOTHING),
    ANDESITE         ((byte) 1, PLAGIOCLASE_FELDSPAR, PYROXENE, AMPHIBOLE, BIOTITE),
    BASALT           ((byte) 2, PLAGIOCLASE_FELDSPAR, PYROXENE, OLIVINE, MAGNETITE),
    BLUESCHIST       ((byte) 3, GLAUCOPHANE, EPIDOTE, ACTINOLITE, CHLORITE),
    CARBONATITE      ((byte) 4, CALCITE, Compound.DOLOMITE, MAGNETITE, APATITE),
    CHALK            ((byte) 5, CALCITE),
    CLAYSTONE        ((byte) 6, KAOLINITE, ILLITE),
    COAL             ((byte) 7, CARBON, PYRITE, KAOLINITE),
    DIORITE          ((byte) 8, PLAGIOCLASE_FELDSPAR, BIOTITE, AMPHIBOLE, QUARTZ),
    DOLOMITE         ((byte) 9, Compound.DOLOMITE, CALCITE, QUARTZ),
    FLINT            ((byte) 10, CHERT, QUARTZ, CHALCEDONY, CALCITE),
    GNEISS           ((byte) 11, QUARTZ, FELDSPAR, BIOTITE, GARNET),
    GRANITE          ((byte) 12, QUARTZ, FELDSPAR, MICA, HORNBLENDE),
    GREENSCHIST      ((byte) 13, CHLORITE, ACTINOLITE, EPIDOTE, ALBITE),
    LIMESTONE        ((byte) 14, CALCITE, Compound.DOLOMITE, QUARTZ),
    MARBLE           ((byte) 15, CALCITE, Compound.DOLOMITE, QUARTZ, GRAPHITE),
    MUDSTONE         ((byte) 16, ILLITE, SMECTITE),
    MUGEARITE        ((byte) 17, NEPHELINE, AUGITE, APATITE, MAGNETITE),
    OBSIDIAN         ((byte) 18, GLASS, FELDSPAR, MAGNETITE, BIOTITE),
    PERIDOTITE       ((byte) 19, OLIVINE, PYROXENE, SERPENTINE, MAGNETITE),
    PUMICE           ((byte) 20, GLASS, FELDSPAR, MAGNETITE, MICA),
    QUARTZ_DIORITE   ((byte) 21, QUARTZ, PLAGIOCLASE_FELDSPAR, BIOTITE, AMPHIBOLE),
    QUARTZ_MONZONITE ((byte) 22, QUARTZ, PLAGIOCLASE_FELDSPAR, POTASSIUM_FELDSPAR, BIOTITE),
    QUARTZITE        ((byte) 23, QUARTZ, FELDSPAR, MICA, MAGNETITE),
    SANDSTONE        ((byte) 24, QUARTZ, FELDSPAR, CALCITE),
    SCHIST           ((byte) 25, BIOTITE, MUSCOVITE, QUARTZ, GARNET),
    SHALE            ((byte) 26, ILLITE, KAOLINITE, FELDSPAR, CALCITE),
    SLATE            ((byte) 27, CHLORITE, MUSCOVITE, FELDSPAR, GRAPHITE),
    SOAPSTONE        ((byte) 28, TALC, CHLORITE, MAGNESITE, MAGNETITE),
    SYLVINITE        ((byte) 29, SYLVITE, HALITE, CARNALLITE, ANHYDRITE),
    TALC_CARBONATE   ((byte) 30, TALC, MAGNESITE, Compound.DOLOMITE, QUARTZ),
    TUFF             ((byte) 31, QUARTZ, FELDSPAR, BIOTITE, HORNBLENDE),
    WHITESCHIST      ((byte) 32, QUARTZ, FELDSPAR, MICA, CHLORITE),

    ;

    public final byte mineral_number;
    public final Compound[] compounds;

    Mineral(byte mineral_number, Compound[] compounds)
    {
        this.mineral_number = mineral_number;
        this.compounds = compounds;
    }

    Mineral(byte mineral_number, Compound compound_1)
    {
        this(mineral_number, new Compound[]{compound_1, Compound.NOTHING, Compound.NOTHING, Compound.NOTHING});
    }

    Mineral(byte mineral_number, Compound compound_1, Compound compound_2)
    {
        this(mineral_number, new Compound[]{compound_1, compound_2, Compound.NOTHING, Compound.NOTHING});
    }

    Mineral(byte mineral_number, Compound compound_1, Compound compound_2, Compound compound_3)
    {
        this(mineral_number, new Compound[]{compound_1, compound_2,compound_3, Compound.NOTHING});
    }

    Mineral(byte mineral_number, Compound compound_1, Compound compound_2, Compound compound_3, Compound compound_4)
    {
        this(mineral_number, new Compound[]{compound_1, compound_2, compound_3, compound_4});
    }
}
