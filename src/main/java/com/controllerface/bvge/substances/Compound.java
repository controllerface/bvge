package com.controllerface.bvge.substances;

import static com.controllerface.bvge.substances.Element.*;

public enum Compound
{
    NOTHING               ( Element.NOTHING ),

    // Minerals
    ACTINOLITE            ( Ca, Mg, Fe, Si ),
    ALBITE                ( Na, Al, Si,  O ),
    AMPHIBOLE             ( Ca, Na, Mg, Fe ),
    AUGITE                ( Ca, Mg, Fe, Si ),
    BIOTITE               (  K, Mg, Fe, Al ),
    COAL                  (  C  ),
    CARNALLITE            (  K, Mg, Cl ),
    CHERT                 ( Si,  O ),
    CHLORITE              ( Mg, Fe, Al, Si ),
    FELDSPAR              (  K, Na, Al, Si ),
    GLASS                 ( Si,  O ),
    GLAUCOPHANE           ( Na, Mg, Al, Si ),
    GRAPHITE              (  C  ),
    HALITE                ( Na, Cl ),
    HORNBLENDE            ( Ca, Na, Mg, Fe ),
    ILLITE                (  K, Al, Si,  O ),
    KAOLINITE             ( Al, Si,  O,  H ),
    MAGNESITE             ( Mg,  C,  O ),
    MAGNETITE             ( Fe,  O ),
    MICA                  (  K, Al, Si,  O ),
    MUSCOVITE             (  K, Al, Si,  O ),
    NEPHELINE             ( Na, Al, Si,  O ),
    OLIVINE               ( Mg, Fe, Si,  O ),
    PLAGIOCLASE_FELDSPAR  ( Na, Ca, Al, Si ),
    POTASSIUM_FELDSPAR    (  K, Al, Si,  O ),
    PYROXENE              ( Ca, Mg, Fe, Si ),
    QUARTZ                ( Si,  O ),
    SMECTITE              ( Na, Mg, Al, Si ),
    SYLVITE               (  K, Cl ),
    TALC                  ( Mg, Si,  O,  H ),
    // Fluids
    CHLOROPHYLL           (  C,  H,  N, Mg ),
    H2O                   (  H,  H,  O ),
    H2O2                  (  H,  H,  O,  O ),
    HOP_MASH              (  K,  P,  S, Ca ),
    OCEANIC_IMPURITIES    ( Cl, Na, Mg,  S ),
    FRESHWATER_IMPURITIES ( Fe,  P,  N,  K ),
    ISOPROPANOL           (  C,  H,  O ),
    SODIUM_HYPOCHLORITE   ( Cl, Na,  O ),
    OCTANE                (  C,  H,  H,  H ),
    BENZENE               (  C,  H,  C,  H ),
    XYLENE                (  C,  H,  H ),
    ETHANOL               (  C,  H,  O ),
    HYDROCHLORIC_ACID     (  H, Cl ),
    SUCROSE               (  C,  H,  O ),
    FRUCTOSE              (  C,  H,  O ),
    // Gemstones,
    AMETHYST              ( Si,  O, Fe, Al ),
    ANHYDRITE             ( Ca,  S,  O ),
    APATITE               ( Ca,  P,  O,  F ),
    CALCITE               ( Ca,  C,  O ),
    CHALCEDONY            ( Si,  O ),
    CITRINE               ( Si,  O, Fe ),
    DIAMOND               (  C ),
    DOLOMITE              ( Ca, Mg,  C,  O ),
    EMERALD               ( Al,  O, Cr,  V ),
    EPIDOTE               ( Ca, Al, Fe, Si ),
    GARNET                ( Al, Fe, Si,  O ),
    PERIDOT               ( Mg, Fe, Si,  O ),
    PURPLE_SAPPHIRE       ( Al,  O, Fe, Cr ),
    PYRITE                ( Fe,  S ),
    RUBY                  ( Al,  O, Cr ),
    SAPPHIRE              ( Al,  O, Ti ),
    SERPENTINE            ( Mg, Si,  O,  H ),
    TANZANITE             ( Ca, Al, Si,  O ),
    YELLOW_SAPPHIRE       ( Al,  O, Fe, Ti ),

    //CALCIUM_HYPOCHLORITE  ( CHLORINE, CALCIUM, OXYGEN ),
    ;


    public final short compound_number;
    public final Element[] elements;
    
    Compound(Element ... elements)
    {
        Element[] _e = new Element[]
        {
            Element.NOTHING,
            Element.NOTHING,
            Element.NOTHING,
            Element.NOTHING,
        };

        if (elements != null)
        {
            System.arraycopy(elements, 0, _e, 0, Math.min(4, elements.length));
        }

        this.compound_number = ( short) this.ordinal();
        this.elements = _e;
    }

//    Compound(Element compound_1)
//    {
//        this(new Element[]{compound_1, Element.NOTHING, Element.NOTHING, Element.NOTHING});
//    }
//
//    Compound(Element compound_1, Element compound_2)
//    {
//        this(new Element[]{compound_1, compound_2, Element.NOTHING, Element.NOTHING});
//    }
//
//    Compound(Element compound_1, Element compound_2, Element compound_3)
//    {
//        this(new Element[]{compound_1, compound_2, compound_3, Element.NOTHING});
//    }
//
//    Compound(Element compound_1, Element compound_2, Element compound_3, Element compound_4)
//    {
//        this(new Element[]{compound_1, compound_2, compound_3, compound_4});
//    }
}
