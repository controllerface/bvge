package com.controllerface.bvge.substances;

import static com.controllerface.bvge.substances.Element.*;

public enum Compound
{
    NOTHING              ((byte) 0, Element.NOTHING),
    ACTINOLITE           ((byte) 1, CALCIUM, MAGNESIUM, IRON, SILICON),
    ALBITE               ((byte) 2, SODIUM, ALUMINUM, SILICON, OXYGEN),
    AMPHIBOLE            ((byte) 3, CALCIUM, SODIUM, MAGNESIUM, IRON),
    ANHYDRITE            ((byte) 4, CALCIUM, SULFUR, OXYGEN),
    APATITE              ((byte) 5, CALCIUM, PHOSPHORUS, OXYGEN, FLUORINE),
    AUGITE               ((byte) 6, CALCIUM, MAGNESIUM, IRON, SILICON),
    BIOTITE              ((byte) 7, POTASSIUM, MAGNESIUM, IRON, ALUMINUM),
    CALCITE              ((byte) 8, CALCIUM, Element.CARBON, OXYGEN),
    CARBON               ((byte) 9, Element.CARBON),
    CARNALLITE           ((byte) 10, POTASSIUM, MAGNESIUM, CHLORINE),
    CHALCEDONY           ((byte) 11, SILICON, OXYGEN),
    CHERT                ((byte) 12, SILICON, OXYGEN),
    CHLORITE             ((byte) 13, MAGNESIUM, IRON, ALUMINUM, SILICON),
    DOLOMITE             ((byte) 14, CALCIUM, MAGNESIUM, Element.CARBON, OXYGEN),
    EPIDOTE              ((byte) 15, CALCIUM, ALUMINUM, IRON, SILICON),
    FELDSPAR             ((byte) 16, POTASSIUM, SODIUM, ALUMINUM, SILICON),
    GARNET               ((byte) 17, ALUMINUM, IRON, SILICON, OXYGEN),
    GLASS                ((byte) 18, SILICON, OXYGEN),
    GLAUCOPHANE          ((byte) 19, SODIUM, MAGNESIUM, ALUMINUM, SILICON),
    GRAPHITE             ((byte) 20, Element.CARBON),
    HALITE               ((byte) 21, SODIUM, CHLORINE),
    HORNBLENDE           ((byte) 22, CALCIUM, SODIUM, MAGNESIUM, IRON),
    ILLITE               ((byte) 23, POTASSIUM, ALUMINUM, SILICON, OXYGEN),
    KAOLINITE            ((byte) 24, ALUMINUM, SILICON, OXYGEN, HYDROGEN),
    MAGNESITE            ((byte) 25, MAGNESIUM, Element.CARBON, OXYGEN),
    MAGNETITE            ((byte) 26, IRON, OXYGEN),
    MICA                 ((byte) 27, POTASSIUM, ALUMINUM, SILICON, OXYGEN),
    MUSCOVITE            ((byte) 28, POTASSIUM, ALUMINUM, SILICON, OXYGEN),
    NEPHELINE            ((byte) 29, SODIUM, ALUMINUM, SILICON, OXYGEN),
    OLIVINE              ((byte) 30, MAGNESIUM, IRON, SILICON, OXYGEN),
    PLAGIOCLASE_FELDSPAR ((byte) 31, SODIUM, CALCIUM, ALUMINUM, SILICON),
    POTASSIUM_FELDSPAR   ((byte) 32, POTASSIUM, ALUMINUM, SILICON, OXYGEN),
    PYRITE               ((byte) 33, IRON, SULFUR),
    PYROXENE             ((byte) 34, CALCIUM, MAGNESIUM, IRON, SILICON),
    QUARTZ               ((byte) 35, SILICON, OXYGEN),
    SERPENTINE           ((byte) 36, MAGNESIUM, SILICON, OXYGEN, HYDROGEN),
    SMECTITE             ((byte) 37, SODIUM, MAGNESIUM, ALUMINUM, SILICON),
    SYLVITE              ((byte) 38, POTASSIUM, CHLORINE),
    TALC                 ((byte) 39, MAGNESIUM, SILICON, OXYGEN, HYDROGEN),

    ;

    public final byte compound_number;

    public final Element[] elements;

    Compound(byte compound_number, Element[] elements)
    {
        this.compound_number = compound_number;
        this.elements = elements;
    }

    Compound(byte compound_number, Element compound_1)
    {
        this(compound_number, new Element[]{compound_1, Element.NOTHING, Element.NOTHING, Element.NOTHING});
    }

    Compound(byte compound_number, Element compound_1, Element compound_2)
    {
        this(compound_number, new Element[]{compound_1, compound_2, Element.NOTHING, Element.NOTHING});
    }

    Compound(byte compound_number, Element compound_1, Element compound_2, Element compound_3)
    {
        this(compound_number, new Element[]{compound_1, compound_2, compound_3, Element.NOTHING});
    }

    Compound(byte compound_number, Element compound_1, Element compound_2, Element compound_3, Element compound_4)
    {
        this(compound_number, new Element[]{compound_1, compound_2, compound_3, compound_4});
    }
}
