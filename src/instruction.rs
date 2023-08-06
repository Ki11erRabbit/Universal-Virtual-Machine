


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opcode {
    Halt,
    NoOp,
    /* Instruction for loading values */
    DeRefReg,
    DeRef,
    Move,
    Set,
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
    DeRefRegF,
    DeRefF,
    MoveF,
    SetF,
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
    DeRefRegA,
    DeRefA,
    MoveA,
    SetA,
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
    JumpPos,
    JumpEven,
    JumpOdd,
    JumpBack,
    JumpForward,
    JumpInfinity,
    JumpNotInfinity,
    JumpOverflow,
    JumpUnderflow,
    JumpNotOverflow,
    JumpNotUnderflow,
    JumpNaN,
    JumpNotNaN,
    JumpRemainder,
    JumpNotRemainder,
    

    /* Instructions for function calls */
    Call,
    Return,
    /* Instructions for stack management */
    Pop,
    Push,
    PopF,
    PushF,
    PopA,
    PushA,
    /* Instructions for accessing the stack */
    DeRefRegStack,
    DeRefStack,
    MoveStack,
    DeRefRegStackF,
    DeRefStackF,
    MoveStackF,
    DeRefRegStackA,
    DeRefStackA,
    MoveStackA,
    /* Instructions for accessing the stack accross cores */
    DeRefRegStackC,
    DeRefStackC,
    MoveStackC,
    DeRefRegStackCF,
    DeRefStackCF,
    MoveStackCF,
    DeRefRegStackCA,
    DeRefStackCA,
    MoveStackCA,
    /* Instructions for memory management */
    Malloc,
    Free,
    /* IO instructions */
    ReadByte,
    /* Takes a string pointer and a length */
    Read,
    WriteByte,
    /* Takes a string pointer and a length */
    Write,
    /* Takes a string pointer and a length */
    Open,
    Close,
    /* Takes a file descriptor */
    Flush,
    /* Instructions for threads */
    ThreadSpawn,
    Remainder,
    /* Instruction for clearing flags */
    Clear,
    /* Instructions operation on integers and floats */
    AddFI,
    SubFI,
    MulFI,
    DivFI,
    AddIF,
    SubIF,
    MulIF,
    DivIF,
    AddUF,
    SubUF,
    MulUF,
    DivUF,
    RegMove,
    RegMoveF,
    RegMoveA,
    /* Special Return for threads */
    ThreadReturn,
    /* Other Thread Instructions */
    ThreadJoin,
    ThreadDetach,

    StackPointer,
    
    
    
    

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
            2 => DeRefReg,
            3 => DeRef,
            4 => Move,
            5 => Set,
            /* Instructions for signed integers */
            6 => AddI,
            7 => SubI,
            8 => MulI,
            9 => DivI,
            10 => EqI,
            11 => NeqI,
            12 => LtI,
            13 => GtI,
            14 => LeqI,
            15 => GeqI,
            /* Instructions for unsigned integers */
            16 => AddU,
            17 => SubU,
            18 => MulU,
            19 => DivU,
            20 => EqU,
            21 => NeqU,
            22 => LtU,
            23 => GtU,
            24 => LeqU,
            25 => GeqU,
            /* Instructions for floating point numbers */
            26 => DeRefRegF,
            27 => DeRefF,
            28 => MoveF,
            29 => SetF,
            30 => AddF,
            31 => SubF,
            32 => MulF,
            33 => DivF,
            34 => EqF,
            35 => NeqF,
            36 => LtF,
            37 => GtF,
            38 => LeqF,
            39 => GeqF,
            /* Instructions for Atomic Integers */
            40 => DeRefRegA,
            41 => DeRefA,
            42 => MoveA,
            43 => SetA,
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
            77 => JumpZero,
            78 => JumpNotZero,
            79 => JumpNeg,
            80 => JumpPos,
            81 => JumpEven,
            82 => JumpOdd,
            83 => JumpBack,
            84 => JumpForward,
            85 => JumpInfinity,
            86 => JumpNotInfinity,
            87 => JumpOverflow,
            88 => JumpUnderflow,
            89 => JumpNotOverflow,
            90 => JumpNotUnderflow,
            91 => JumpNaN,
            92 => JumpNotNaN,
            93 => JumpRemainder,
            94 => JumpNotRemainder,
            
            // Block of reserved opcodes for future use
            /* Instructions for function calls */
            109 => Call,
            110 => Return,
            /* Instructions for stack management */
            111 => Pop,
            112 => Push,
            113 => PopF,
            114 => PushF,
            115 => PopA,
            116 => PushA,
            /* Instructions for accessing the stack */
            117 => DeRefRegStack,
            118 => DeRefStack,
            119 => MoveStack,
            120 => DeRefRegStackF,
            121 => DeRefStackF,
            122 => MoveStackF,
            123 => DeRefRegStackA,
            124 => DeRefStackA,
            125 => MoveStackA,
            /* Instructions for accessing the stack accross cores */
            126 => DeRefRegStackC,
            127 => DeRefStackC,
            128 => MoveStackC,
            129 => DeRefRegStackCF,
            130 => DeRefStackCF,
            131 => MoveStackCF,
            132 => DeRefRegStackCA,
            133 => DeRefStackCA,
            134 => MoveStackCA,
            /* Instructions for memory management */
            135 => Malloc,
            136 => Free,
            /* IO instructions */
            137 => ReadByte,
            138 => Read,
            139 => WriteByte,
            140 => Write,
            141 => Open,
            142 => Close,
            143 => Flush,
            /* Instructions for threads */
            144 => ThreadSpawn,
            145 => Remainder,
            /* Instruction for clearing flags */
            146 => Clear,
            /* Instructions operation on integers and floats */
            147 => AddFI,
            148 => SubFI,
            149 => MulFI,
            150 => DivFI,
            151 => AddIF,
            152 => SubIF,
            153 => MulIF,
            154 => DivIF,
            155 => AddUF,
            156 => SubUF,
            157 => MulUF,
            158 => DivUF,
            159 => RegMove,
            160 => RegMoveF,
            161 => RegMoveA,
            /* Special Return for Threads */
            162 => ThreadReturn,
            163 => ThreadJoin,
            164 => ThreadDetach,
            165 => StackPointer,
            
            
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
