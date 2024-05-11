package com.controllerface.bvge.substances;

import static com.controllerface.bvge.substances.Element.*;

public enum Compound
{
    NOTHING               (Element.NOTHING),

    // Minerals
    ACTINOLITE            (CALCIUM, MAGNESIUM, IRON, SILICON),
    ALBITE                (SODIUM, ALUMINUM, SILICON, OXYGEN),
    AMPHIBOLE             (CALCIUM, SODIUM, MAGNESIUM, IRON),
    AUGITE                (CALCIUM, MAGNESIUM, IRON, SILICON),
    BIOTITE               (POTASSIUM, MAGNESIUM, IRON, ALUMINUM),
    COAL                  (CARBON),
    CARNALLITE            (POTASSIUM, MAGNESIUM, CHLORINE),
    CHERT                 (SILICON, OXYGEN),
    CHLORITE              (MAGNESIUM, IRON, ALUMINUM, SILICON),
    FELDSPAR              (POTASSIUM, SODIUM, ALUMINUM, SILICON),
    GLASS                 (SILICON, OXYGEN),
    GLAUCOPHANE           (SODIUM, MAGNESIUM, ALUMINUM, SILICON),
    GRAPHITE              (CARBON),
    HALITE                (SODIUM, CHLORINE),
    HORNBLENDE            (CALCIUM, SODIUM, MAGNESIUM, IRON),
    ILLITE                (POTASSIUM, ALUMINUM, SILICON, OXYGEN),
    KAOLINITE             (ALUMINUM, SILICON, OXYGEN, HYDROGEN),
    MAGNESITE             (MAGNESIUM, CARBON, OXYGEN),
    MAGNETITE             (IRON, OXYGEN),
    MICA                  (POTASSIUM, ALUMINUM, SILICON, OXYGEN),
    MUSCOVITE             (POTASSIUM, ALUMINUM, SILICON, OXYGEN),
    NEPHELINE             (SODIUM, ALUMINUM, SILICON, OXYGEN),
    OLIVINE               (MAGNESIUM, IRON, SILICON, OXYGEN),
    PLAGIOCLASE_FELDSPAR  (SODIUM, CALCIUM, ALUMINUM, SILICON),
    POTASSIUM_FELDSPAR    (POTASSIUM, ALUMINUM, SILICON, OXYGEN),
    PYROXENE              (CALCIUM, MAGNESIUM, IRON, SILICON),
    QUARTZ                (SILICON, OXYGEN),
    SMECTITE              (SODIUM, MAGNESIUM, ALUMINUM, SILICON),
    SYLVITE               (POTASSIUM, CHLORINE),
    TALC                  (MAGNESIUM, SILICON, OXYGEN, HYDROGEN),

    //CALCIUM_HYPOCHLORITE  (CHLORINE, CALCIUM, OXYGEN),

    // Fluids,
    H2O                   (HYDROGEN, HYDROGEN, OXYGEN),
    H2O2                  (HYDROGEN, OXYGEN, HYDROGEN, OXYGEN),
    HOP_MASH              (POTASSIUM, PHOSPHORUS, SULFUR, CALCIUM),
    OCEANIC_IMPURITIES    (CHLORINE, SODIUM, MAGNESIUM, SULFUR),
    FRESHWATER_IMPURITIES (IRON, PHOSPHORUS, NITROGEN, POTASSIUM),
    ISOPROPANOL           (CARBON, HYDROGEN, OXYGEN),
    SODIUM_HYPOCHLORITE   (CHLORINE, SODIUM, OXYGEN),
    OCTANE                (CARBON, HYDROGEN, HYDROGEN, HYDROGEN),
    BENZENE               (CARBON, HYDROGEN, CARBON, HYDROGEN),
    XYLENE                (CARBON, HYDROGEN, HYDROGEN),
    ETHANOL               (CARBON, HYDROGEN, OXYGEN),
    HYDROCHLORIC_ACID     (HYDROGEN, CHLORINE),
    SUCROSE               (CARBON, HYDROGEN, OXYGEN),
    FRUCTOSE              (CARBON, HYDROGEN, OXYGEN),
    CHLOROPHYLL           (CARBON, HYDROGEN, NITROGEN, MAGNESIUM),


    // Gemstones,
    AMETHYST              (SILICON, OXYGEN, IRON, ALUMINUM),
    ANHYDRITE             (CALCIUM, SULFUR, OXYGEN),
    APATITE               (CALCIUM, PHOSPHORUS, OXYGEN, FLUORINE),
    CALCITE               (CALCIUM, CARBON, OXYGEN),
    CHALCEDONY            (SILICON, OXYGEN),
    CITRINE               (SILICON, OXYGEN, IRON),
    DIAMOND               (CARBON),
    DOLOMITE              (CALCIUM, MAGNESIUM, CARBON, OXYGEN),
    EMERALD               (ALUMINUM, OXYGEN, CHROMIUM, VANADIUM),
    EPIDOTE               (CALCIUM, ALUMINUM, IRON, SILICON),
    GARNET                (ALUMINUM, IRON, SILICON, OXYGEN),
    PERIDOT               (MAGNESIUM, IRON, SILICON, OXYGEN),
    PURPLE_SAPPHIRE       (ALUMINUM, OXYGEN, IRON, CHROMIUM),
    PYRITE                (IRON, SULFUR),
    RUBY                  (ALUMINUM, OXYGEN, CHROMIUM),
    SAPPHIRE              (ALUMINUM, OXYGEN, TITANIUM),
    SERPENTINE            (MAGNESIUM, SILICON, OXYGEN, HYDROGEN),
    TANZANITE             (CALCIUM, ALUMINUM, SILICON, OXYGEN),
    YELLOW_SAPPHIRE       (ALUMINUM, OXYGEN, IRON, TITANIUM),

    ;

    public final short compound_number;

    public final Element[] elements;

    Compound(Element[] elements)
    {
        this.compound_number = (short) this.ordinal();
        this.elements = elements;
    }

    Compound(Element compound_1)
    {
        this(new Element[]{compound_1, Element.NOTHING, Element.NOTHING, Element.NOTHING});
    }

    Compound(Element compound_1, Element compound_2)
    {
        this(new Element[]{compound_1, compound_2, Element.NOTHING, Element.NOTHING});
    }

    Compound(Element compound_1, Element compound_2, Element compound_3)
    {
        this(new Element[]{compound_1, compound_2, compound_3, Element.NOTHING});
    }

    Compound(Element compound_1, Element compound_2, Element compound_3, Element compound_4)
    {
        this(new Element[]{compound_1, compound_2, compound_3, compound_4});
    }
}
