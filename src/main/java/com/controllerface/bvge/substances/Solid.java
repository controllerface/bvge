package com.controllerface.bvge.substances;

import static com.controllerface.bvge.substances.Compound.*;

public enum Solid
{
    ANDESITE         (PLAGIOCLASE_FELDSPAR, PYROXENE, AMPHIBOLE, BIOTITE),
    BASALT           (PLAGIOCLASE_FELDSPAR, PYROXENE, OLIVINE, MAGNETITE),
    BLUESCHIST       (GLAUCOPHANE, EPIDOTE, ACTINOLITE, SAPPHIRE),
    CARBONATITE      (CALCITE, DOLOMITE, MAGNETITE, APATITE),
    CHALK            (CALCITE),
    CLAYSTONE        (KAOLINITE, ILLITE),
    COAL_DEPOSIT     (COAL, PYRITE, KAOLINITE),
    DIORITE          (PLAGIOCLASE_FELDSPAR, BIOTITE, AMPHIBOLE, QUARTZ),
    DOLOMITE_DEPOSIT (DOLOMITE, CALCITE, QUARTZ),
    FLINT            (CHERT, QUARTZ, CHALCEDONY, CALCITE),
    GNEISS           (QUARTZ, TANZANITE, BIOTITE, GARNET),
    GRANITE          (PURPLE_SAPPHIRE, FELDSPAR, MICA, HORNBLENDE),
    GREENSCHIST      (EMERALD, ACTINOLITE, EPIDOTE, ALBITE),
    LIMESTONE        (CALCITE, DOLOMITE, QUARTZ),
    MARBLE           (CALCITE, YELLOW_SAPPHIRE, QUARTZ, GRAPHITE),
    MUDSTONE         (ILLITE, SMECTITE),
    MUGEARITE        (NEPHELINE, AUGITE, APATITE, MAGNETITE),
    OBSIDIAN         (GLASS, FELDSPAR, MAGNETITE, BIOTITE),
    PERIDOTITE       (OLIVINE, PERIDOT, SERPENTINE, MAGNETITE),
    PUMICE           (GLASS, FELDSPAR, MAGNETITE, MICA),
    QUARTZ_DIORITE   (QUARTZ, PLAGIOCLASE_FELDSPAR, CITRINE, AMPHIBOLE),
    QUARTZ_MONZONITE (QUARTZ, PLAGIOCLASE_FELDSPAR, POTASSIUM_FELDSPAR, BIOTITE),
    QUARTZITE        (QUARTZ, AMETHYST, MICA, MAGNETITE),
    SANDSTONE        (QUARTZ, FELDSPAR, CALCITE),
    SCHIST           (BIOTITE, MUSCOVITE, RUBY, GARNET),
    SHALE            (ILLITE, KAOLINITE, FELDSPAR, CALCITE),
    SLATE            (CHLORITE, MUSCOVITE, FELDSPAR, GRAPHITE),
    SOAPSTONE        (TALC, CHLORITE, MAGNESITE, MAGNETITE),
    SYLVINITE        (SYLVITE, HALITE, CARNALLITE, ANHYDRITE),
    TALC_CARBONATE   (TALC, MAGNESITE, DOLOMITE, QUARTZ),
    TUFF             (QUARTZ, FELDSPAR, BIOTITE, HORNBLENDE),
    WHITESCHIST      (QUARTZ, FELDSPAR, MICA, CHLORITE),
    KIMBERLITE       (DIAMOND, GARNET, CALCITE, DOLOMITE),

    ;

    private static final int slot_count = 4;
    public final int mineral_number;
    public final Compound[] compounds;

    Solid(Compound ... compounds)
    {
        assert compounds != null : "Null element list";
        assert compounds.length <= slot_count;
        var _compounds = new Compound[]{ Compound.NOTHING, Compound.NOTHING, Compound.NOTHING, Compound.NOTHING };
        System.arraycopy(compounds, 0, _compounds, 0, compounds.length);
        this.mineral_number = this.ordinal();
        this.compounds = _compounds;
    }
}
