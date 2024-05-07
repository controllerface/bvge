package com.controllerface.bvge.substances;

import static com.controllerface.bvge.substances.Element.*;

public enum Compound
{
    NOTHING              (Element.NOTHING),
    ACTINOLITE           (CALCIUM, MAGNESIUM, IRON, SILICON),
    ALBITE               (SODIUM, ALUMINUM, SILICON, OXYGEN),
    AMPHIBOLE            (CALCIUM, SODIUM, MAGNESIUM, IRON),
    ANHYDRITE            (CALCIUM, SULFUR, OXYGEN),
    APATITE              (CALCIUM, PHOSPHORUS, OXYGEN, FLUORINE),
    AUGITE               (CALCIUM, MAGNESIUM, IRON, SILICON),
    BIOTITE              (POTASSIUM, MAGNESIUM, IRON, ALUMINUM),
    CALCITE              (CALCIUM, Element.CARBON, OXYGEN),
    CARBON               (Element.CARBON),
    CARNALLITE           (POTASSIUM, MAGNESIUM, Element.CHLORINE),
    CHALCEDONY           (SILICON, OXYGEN),
    CHERT                (SILICON, OXYGEN),
    CHLORITE             (MAGNESIUM, IRON, ALUMINUM, SILICON),
    DOLOMITE             (CALCIUM, MAGNESIUM, Element.CARBON, OXYGEN),
    EPIDOTE              (CALCIUM, ALUMINUM, IRON, SILICON),
    FELDSPAR             (POTASSIUM, SODIUM, ALUMINUM, SILICON),
    GARNET               (ALUMINUM, IRON, SILICON, OXYGEN),
    GLASS                (SILICON, OXYGEN),
    GLAUCOPHANE          (SODIUM, MAGNESIUM, ALUMINUM, SILICON),
    GRAPHITE             (Element.CARBON),
    HALITE               (SODIUM, Element.CHLORINE),
    HORNBLENDE           (CALCIUM, SODIUM, MAGNESIUM, IRON),
    ILLITE               (POTASSIUM, ALUMINUM, SILICON, OXYGEN),
    KAOLINITE            (ALUMINUM, SILICON, OXYGEN, HYDROGEN),
    MAGNESITE            (MAGNESIUM, Element.CARBON, OXYGEN),
    MAGNETITE            (IRON, OXYGEN),
    MICA                 (POTASSIUM, ALUMINUM, SILICON, OXYGEN),
    MUSCOVITE            (POTASSIUM, ALUMINUM, SILICON, OXYGEN),
    NEPHELINE            (SODIUM, ALUMINUM, SILICON, OXYGEN),
    OLIVINE              (MAGNESIUM, IRON, SILICON, OXYGEN),
    PLAGIOCLASE_FELDSPAR (SODIUM, CALCIUM, ALUMINUM, SILICON),
    POTASSIUM_FELDSPAR   (POTASSIUM, ALUMINUM, SILICON, OXYGEN),
    PYRITE               (IRON, SULFUR),
    PYROXENE             (CALCIUM, MAGNESIUM, IRON, SILICON),
    QUARTZ               (SILICON, OXYGEN),
    SERPENTINE           (MAGNESIUM, SILICON, OXYGEN, HYDROGEN),
    SMECTITE             (SODIUM, MAGNESIUM, ALUMINUM, SILICON),
    SYLVITE              (POTASSIUM, Element.CHLORINE),
    TALC                 (MAGNESIUM, SILICON, OXYGEN, HYDROGEN),


    CALCIUM_HYPOCHLORITE (Element.CHLORINE, CALCIUM, OXYGEN),


    WATER                (HYDROGEN, OXYGEN, OXYGEN),
    HOP_MASH             (POTASSIUM, PHOSPHORUS, SULFUR, CALCIUM),
    OCEANIC_IMPURITIES   (Element.CHLORINE, SODIUM, MAGNESIUM, SULFUR),
    PEROXIDE             (HYDROGEN, OXYGEN, HYDROGEN, OXYGEN),
    ISOPROPANOL          (Element.CARBON, HYDROGEN, OXYGEN),
    CHLORINE             (Element.CHLORINE),
    SODIUM_HYPOCHLORITE  (Element.CHLORINE, SODIUM, OXYGEN),
    OCTANE               (Element.CARBON, HYDROGEN, HYDROGEN, HYDROGEN),
    BENZENE              (Element.CARBON, HYDROGEN, Element.CARBON, HYDROGEN),
    XYLENE               (Element.CARBON, HYDROGEN, HYDROGEN),
    ETHANOL              (Element.CARBON, HYDROGEN, OXYGEN),
    HYDROCHLORIC_ACID    (HYDROGEN, Element.CHLORINE),
    SUCROSE              (Element.CARBON, HYDROGEN, OXYGEN),
    FRUCTOSE             (Element.CARBON, HYDROGEN, OXYGEN),

    ;

    public final byte compound_number;

    public final Element[] elements;

    Compound(Element[] elements)
    {
        this.compound_number = (byte) this.ordinal();
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
