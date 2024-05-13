package com.controllerface.bvge.substances;

import static com.controllerface.bvge.substances.Element.*;

public enum Compound
{
    NOTHING               ( __, __, __, __ ),

    // Minerals
    ACTINOLITE            ( Ca, Mg, Fe, Si ),
    ALBITE                ( Na, Al, Si,  O ),
    AMPHIBOLE             ( Ca, Na, Mg, Fe ),
    AUGITE                ( Ca, Mg, Fe, Si ),
    BIOTITE               (  K, Mg, Fe, Al ),
    COAL                  (  C, __, __, __ ),
    CARNALLITE            (  K, Mg, Cl, __ ),
    CHERT                 ( Si,  O, __, __ ),
    CHLORITE              ( Mg, Fe, Al, Si ),
    FELDSPAR              (  K, Na, Al, Si ),
    GLASS                 ( Si,  O, __, __ ),
    GLAUCOPHANE           ( Na, Mg, Al, Si ),
    GRAPHITE              (  C, __, __, __ ),
    HALITE                ( Na, Cl, __, __ ),
    HORNBLENDE            ( Ca, Na, Mg, Fe ),
    ILLITE                (  K, Al, Si,  O ),
    KAOLINITE             ( Al, Si,  O,  H ),
    MAGNESITE             ( Mg,  C,  O, __ ),
    MAGNETITE             ( Fe,  O, __, __ ),
    MICA                  (  K, Al, Si,  O ),
    MUSCOVITE             (  K, Al, Si,  O ),
    NEPHELINE             ( Na, Al, Si,  O ),
    OLIVINE               ( Mg, Fe, Si,  O ),
    PLAGIOCLASE_FELDSPAR  ( Na, Ca, Al, Si ),
    POTASSIUM_FELDSPAR    (  K, Al, Si,  O ),
    PYROXENE              ( Ca, Mg, Fe, Si ),
    QUARTZ                ( Si,  O, __, __ ),
    SMECTITE              ( Na, Mg, Al, Si ),
    SYLVITE               (  K, Cl, __, __ ),
    TALC                  ( Mg, Si,  O,  H ),

    // Fluids
    CHLOROPHYLL           (  C,  H,  N, Mg ),
    H2O                   (  H,  H,  O, __ ),
    H2O2                  (  H,  H,  O,  O ),
    HOP_MASH              (  K,  P,  S, Ca ),
    OCEANIC_IMPURITIES    ( Cl, Na, Mg,  S ),
    FRESHWATER_IMPURITIES ( Fe,  P,  N,  K ),
    ISOPROPANOL           (  C,  H,  O, __ ),
    SODIUM_HYPOCHLORITE   ( Cl, Na,  O, __ ),
    OCTANE                (  C,  H,  H,  H ),
    BENZENE               (  C,  H,  C,  H ),
    XYLENE                (  C,  H,  H, __ ),
    ETHANOL               (  C,  H,  O, __ ),
    HYDROCHLORIC_ACID     (  H, Cl, __, __ ),
    SUCROSE               (  C,  H,  O, __ ),
    FRUCTOSE              (  C,  H,  O, __ ),

    // Gemstones,
    AMETHYST              ( Si,  O, Fe, Al ),
    ANHYDRITE             ( Ca,  S,  O, __ ),
    APATITE               ( Ca,  P,  O,  F ),
    CALCITE               ( Ca,  C,  O, __ ),
    CHALCEDONY            ( Si,  O, __, __ ),
    CITRINE               ( Si,  O, Fe, __ ),
    DIAMOND               (  C, __, __, __ ),
    DOLOMITE              ( Ca, Mg,  C,  O ),
    EMERALD               ( Al,  O, Cr,  V ),
    EPIDOTE               ( Ca, Al, Fe, Si ),
    GARNET                ( Al, Fe, Si,  O ),
    PERIDOT               ( Mg, Fe, Si,  O ),
    PURPLE_SAPPHIRE       ( Al,  O, Fe, Cr ),
    PYRITE                ( Fe,  S, __, __ ),
    RUBY                  ( Al,  O, Cr, __ ),
    SAPPHIRE              ( Al,  O, Ti, __ ),
    SERPENTINE            ( Mg, Si,  O,  H ),
    TANZANITE             ( Ca, Al, Si,  O ),
    YELLOW_SAPPHIRE       ( Al,  O, Fe, Ti ),

    ;

    private static final int slot_count = 4;
    public final short compound_number;
    public final Element[] elements;
    
    Compound(Element ... elements)
    {
        assert elements != null : "Null element list";
        assert elements.length == slot_count;
        this.compound_number = ( short) this.ordinal();
        this.elements = elements;
    }
}
