package com.controllerface.bvge.substances;

public enum Element
{
    NOTHING       ((byte)0),
    HYDROGEN      ((byte)1),
    HELIUM        ((byte)2),
    LITHIUM       ((byte)3),
    BERYLLIUM     ((byte)4),
    BORON         ((byte)5),
    CARBON        ((byte)6),
    NITROGEN      ((byte)7),
    OXYGEN        ((byte)8),
    FLUORINE      ((byte)9),
    NEON          ((byte)10),
    SODIUM        ((byte)11),
    MAGNESIUM     ((byte)12),
    ALUMINUM      ((byte)13),
    SILICON       ((byte)14),
    PHOSPHORUS    ((byte)15),
    SULFUR        ((byte)16),
    CHLORINE      ((byte)17),
    ARGON         ((byte)18),
    POTASSIUM     ((byte)19),
    CALCIUM       ((byte)20),
    SCANDIUM      ((byte)21),
    TITANIUM      ((byte)22),
    VANADIUM      ((byte)23),
    CHROMIUM      ((byte)24),
    MANGANESE     ((byte)25),
    IRON          ((byte)26),
    COBALT        ((byte)27),
    NICKEL        ((byte)28),
    COPPER        ((byte)29),
    ZINC          ((byte)30),
    GALLIUM       ((byte)31),
    GERMANIUM     ((byte)32),
    ARSENIC       ((byte)33),
    SELENIUM      ((byte)34),
    BROMINE       ((byte)35),
    KRYPTON       ((byte)36),
    RUBIDIUM      ((byte)37),
    STRONTIUM     ((byte)38),
    YTTRIUM       ((byte)39),
    ZIRCONIUM     ((byte)40),
    NIOBIUM       ((byte)41),
    MOLYBDENUM    ((byte)42),
    TECHNETIUM    ((byte)43),
    RUTHENIUM     ((byte)44),
    RHODIUM       ((byte)45),
    PALLADIUM     ((byte)46),
    SILVER        ((byte)47),
    CADMIUM       ((byte)48),
    INDIUM        ((byte)49),
    TIN           ((byte)50),
    ANTIMONY      ((byte)51),
    TELLURIUM     ((byte)52),
    IODINE        ((byte)53),
    XENON         ((byte)54),
    CESIUM        ((byte)55),
    BARIUM        ((byte)56),
    LANTHANUM     ((byte)57),
    CERIUM        ((byte)58),
    PRASEODYMIUM  ((byte)59),
    NEODYMIUM     ((byte)60),
    PROMETHIUM    ((byte)61),
    SAMARIUM      ((byte)62),
    EUROPIUM      ((byte)63),
    GADOLINIUM    ((byte)64),
    TERBIUM       ((byte)65),
    DYSPROSIUM    ((byte)66),
    HOLMIUM       ((byte)67),
    ERBIUM        ((byte)68),
    THULIUM       ((byte)69),
    YTTERBIUM     ((byte)70),
    LUTETIUM      ((byte)71),
    HAFNIUM       ((byte)72),
    TANTALUM      ((byte)73),
    TUNGSTEN      ((byte)74),
    RHENIUM       ((byte)75),
    OSMIUM        ((byte)76),
    IRIDIUM       ((byte)77),
    PLATINUM      ((byte)78),
    GOLD          ((byte)79),
    MERCURY       ((byte)80),
    THALLIUM      ((byte)81),
    LEAD          ((byte)82),
    BISMUTH       ((byte)83),
    POLONIUM      ((byte)84),
    ASTATINE      ((byte)85),
    RADON         ((byte)86),
    FRANCIUM      ((byte)87),
    RADIUM        ((byte)88),
    ACTINIUM      ((byte)89),
    THORIUM       ((byte)90),
    PROTACTINIUM  ((byte)91),
    URANIUM       ((byte)92),
    NEPTUNIUM     ((byte)93),
    PLUTONIUM     ((byte)94),
    AMERICIUM     ((byte)95),
    CURIUM        ((byte)96),
    BERKELIUM     ((byte)97),
    CALIFORNIUM   ((byte)98),
    EINSTEINIUM   ((byte)99),
    FERMIUM       ((byte)100),
    MENDELEVIUM   ((byte)101),
    NOBELIUM      ((byte)102),
    LAWRENCIUM    ((byte)103),
    RUTHERFORDIUM ((byte)104),
    DUBNIUM       ((byte)105),
    SEABORGIUM    ((byte)106),
    BOHRIUM       ((byte)107),
    HASSIUM       ((byte)108),
    MEITNERIUM    ((byte)109),
    DARMSTADTIUM  ((byte)110),
    ROENTGENIUM   ((byte)111),
    COPERNICIUM   ((byte)112),
    NIHONIUM      ((byte)113),
    FLEROVIUM     ((byte)114),
    MOSCOVIUM     ((byte)115),
    LIVERMORIUM   ((byte)116),
    TENNESSINE    ((byte)117),
    OGANESSON     ((byte)118);

    public final byte atomic_number;

    Element(byte atomic_number)
    {
        this.atomic_number = atomic_number;
    }
}
