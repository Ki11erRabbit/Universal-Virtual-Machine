


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opcode {
    Halt,
    NoOp,
    /* Instruction for loading values */
    Load,
    /* Instructions for signed integers */
    AddI,
    SubI,
    MulI,
    DivI,
    EqI,
    NeqI,
    LtI,
    GtI,
    LeqI,
    GeqI,
    /* Instructions for unsigned integers */
    AddU,
    SubU,
    MulU,
    DivU,
    EqU,
    NeqU,
    LtU,
    GtU,
    LeqU,
    GeqU,
    /* Instructions of adding 32-bit floating point numbers */
    AddF32,
    SubF32,
    MulF32,
    DivF32,
    EqF32,
    NeqF32,
    LtF32,
    GtF32,
    LeqF32,
    GeqF32,
    /* Instructions of adding 64-bit floating point numbers */
    AddF64,
    SubF64,
    MulF64,
    DivF64,
    EqF64,
    NeqF64,
    LtF64,
    GtF64,
    LeqF64,
    GeqF64,
    /* Instructions for Atomic Integers */
    LoadA,
    /* Instructions for signed Atomic Integers */
    AddAI,
    SubAI,
    MulAI,
    DivAI,
    EqAI,
    NeqAI,
    LtAI,
    GtAI,
    LeqAI,
    GeqAI,
    /* Instructions for unsigned Atomic Integers */
    AddAU,
    SubAU,
    MulAU,
    DivAU,
    EqAU,
    NeqAU,
    LtAU,
    GtAU,
    LeqAU,
    GeqAU,
    /* Instructions for bitwise operations */
    And,
    Or,
    Xor,
    Not,
    ShiftLeft,
    ShiftRight,
    /* Instructions for jumping */
    Jump,
    JumpEq,
    JumpNeq,
    JumpLt,
    JumpGt,
    JumpLeq,
    JumpGeq,

    /* Instructions for function calls */
    Call,
    Return,
    /* Instructions for stack management */
    Pop,
    Push,
    /* Instructions for memory management */
    Malloc,
    Free,
    Move,
    

    /* Instructions illegal instruction */
    Illegal,
}

impl From<u16> for Opcode {
    fn from(v: u16) -> Self {
        use Opcode::*;
        match v {
            0 => Halt,
            1 => NoOp,
            /* Instructions for values */
            2 => Load,
            /* Instructions for signed integers */
            3 => AddI,
            4 => SubI,
            5 => MulI,
            6 => DivI,
            7 => EqI,
            8 => NeqI,
            9 => LtI,
            10 => GtI,
            11 => LeqI,
            12 => GeqI,
            /* Instructions for unsigned integers */
            13 => AddU,
            14 => SubU,
            15 => MulU,
            16 => DivU,
            17 => EqU,
            18 => NeqU,
            19 => LtU,
            20 => GtU,
            21 => LeqU,
            22 => GeqU,
            /* Instructions for 32-bit floating point numbers */
            23 => AddF32,
            24 => SubF32,
            25 => MulF32,
            26 => DivF32,
            27 => EqF32,
            28 => NeqF32,
            29 => LtF32,
            30 => GtF32,
            31 => LeqF32,
            32 => GeqF32,
            /* Instructions for 64-bit floating point numbers */
            33 => AddF64,
            34 => SubF64,
            35 => MulF64,
            36 => DivF64,
            37 => EqF64,
            38 => NeqF64,
            39 => LtF64,
            40 => GtF64,
            41 => LeqF64,
            42 => GeqF64,
            /* Instructions for Atomic Integers */
            43 => LoadA,
            /* Instructions for signed Atomic Integers */
            44 => AddAI,
            45 => SubAI,
            46 => MulAI,
            47 => DivAI,
            48 => EqAI,
            49 => NeqAI,
            50 => LtAI,
            51 => GtAI,
            52 => LeqAI,
            53 => GeqAI,
            /* Instructions for unsigned Atomic Integers */
            54 => AddAU,
            55 => SubAU,
            56 => MulAU,
            57 => DivAU,
            58 => EqAU,
            59 => NeqAU,
            60 => LtAU,
            61 => GtAU,
            62 => LeqAU,
            63 => GeqAU,
            /* Instructions for bitwise operations */
            64 => And,
            65 => Or,
            66 => Xor,
            67 => Not,
            68 => ShiftLeft,
            69 => ShiftRight,
            /* Instructions for jumping */
            70 => Jump,
            71 => JumpEq,
            72 => JumpNeq,
            73 => JumpLt,
            74 => JumpGt,
            75 => JumpLeq,
            76 => JumpGeq,
            // Gap for future instructions
            /* Instructions for function calls */
            100 => Call,
            101 => Return,
            /* Instructions for stack management */
            102 => Pop,
            103 => Push,
            /* Instructions for memory management */
            104 => Malloc,
            105 => Free,
            106 => Move,
            /* Instructions illegal instruction */
            _ => Illegal,
        }
    }
}



#[derive(Debug, Clone, PartialEq)]
pub struct Instruction {
    pub opcode: Opcode,
}

impl Instruction {
    pub fn new(opcode: Opcode) -> Self {
        Self { opcode }
    }
}
