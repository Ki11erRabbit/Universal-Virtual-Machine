


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opcode {
    Halt,
    NoOp,
    /* Instruction for loading values */
    Load,
    DeRef,
    Move,
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
    /* Instructions for floating point numbers */
    LoadF,
    DeRefF,
    MoveF,
    AddF,
    SubF,
    MulF,
    DivF,
    EqF,
    NeqF,
    LtF,
    GtF,
    LeqF,
    GeqF,
    /* Instructions for Atomic Integers */
    LoadA,
    DeRefA,
    MoveA,
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
    JumpZero,
    JumpNotZero,
    JumpNeg,
    JumpNotNeg,
    JumpEven,
    JumpOdd,
    JumpBack,
    JumpForward,

    /* Instructions for function calls */
    Call,
    Return,
    /* Instructions for stack management */
    Pop,
    Push,
    /* Instructions for memory management */
    Malloc,
    Free,
    

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
            3 => DeRef,
            4 => Move,
            /* Instructions for signed integers */
            5 => AddI,
            6 => SubI,
            7 => MulI,
            8 => DivI,
            9 => EqI,
            10 => NeqI,
            11 => LtI,
            12 => GtI,
            13 => LeqI,
            14 => GeqI,
            /* Instructions for unsigned integers */
            15 => AddU,
            16 => SubU,
            17 => MulU,
            18 => DivU,
            19 => EqU,
            20 => NeqU,
            21 => LtU,
            22 => GtU,
            23 => LeqU,
            24 => GeqU,
            /* Instructions for floating point numbers */
            25 => LoadF,
            26 => DeRefF,
            27 => MoveF,
            28 => AddF,
            29 => SubF,
            30 => MulF,
            31 => DivF,
            32 => EqF,
            33 => NeqF,
            34 => LtF,
            35 => GtF,
            36 => LeqF,
            37 => GeqF,
            /* Instructions for Atomic Integers */
            38 => LoadA,
            39 => DeRefA,
            40 => MoveA,
            /* Instructions for signed Atomic Integers */
            41 => AddAI,
            42 => SubAI,
            43 => MulAI,
            44 => DivAI,
            45 => EqAI,
            46 => NeqAI,
            47 => LtAI,
            48 => GtAI,
            49 => LeqAI,
            50 => GeqAI,
            /* Instructions for unsigned Atomic Integers */
            51 => AddAU,
            52 => SubAU,
            53 => MulAU,
            54 => DivAU,
            55 => EqAU,
            56 => NeqAU,
            57 => LtAU,
            58 => GtAU,
            59 => LeqAU,
            60 => GeqAU,
            /* Instructions for bitwise operations */
            61 => And,
            62 => Or,
            63 => Xor,
            64 => Not,
            65 => ShiftLeft,
            66 => ShiftRight,
            /* Instructions for jumping */
            67 => Jump,
            68 => JumpEq,
            69 => JumpNeq,
            70 => JumpLt,
            71 => JumpGt,
            72 => JumpLeq,
            73 => JumpGeq,
            74 => JumpZero,
            75 => JumpNotZero,
            76 => JumpNeg,
            77 => JumpNotNeg,
            78 => JumpEven,
            79 => JumpOdd,
            80 => JumpBack,
            81 => JumpForward,
            // Gap to allow for future jump instructions
            /* Instructions for function calls */
            106 => Call,
            107 => Return,
            /* Instructions for stack management */
            108 => Pop,
            109 => Push,
            /* Instructions for memory management */
            110 => Malloc,
            111 => Free,
            
            
            
            
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
