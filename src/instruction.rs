


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
    SetStack,
    DeRefRegStackF,
    DeRefStackF,
    MoveStackF,
    SetStackF,
    DeRefRegStackA,
    DeRefStackA,
    MoveStackA,
    SetStackA,
    /* Instructions for accessing the stack accross cores */
    DeRefRegStackC,
    DeRefStackC,
    MoveStackC,
    SetStackC,
    DeRefRegStackCF,
    DeRefStackCF,
    MoveStackCF,
    SetStackCF,
    DeRefRegStackCA,
    DeRefStackCA,
    MoveStackCA,
    SetStackCA,
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
            120 => SetStack,
            121 => DeRefRegStackF,
            122 => DeRefStackF,
            123 => MoveStackF,
            124 => SetStackF,
            125 => DeRefRegStackA,
            126 => DeRefStackA,
            127 => MoveStackA,
            128 => SetStackA,
            /* Instructions for accessing the stack accross cores */
            129 => DeRefRegStackC,
            130 => DeRefStackC,
            131 => MoveStackC,
            132 => SetStackC,
            133 => DeRefRegStackCF,
            134 => DeRefStackCF,
            135 => MoveStackCF,
            136 => SetStackCF,
            137 => DeRefRegStackCA,
            138 => DeRefStackCA,
            139 => MoveStackCA,
            140 => SetStackCA,
            /* Instructions for memory management */
            141 => Malloc,
            142 => Free,
            /* IO instructions */
            143 => ReadByte,
            144 => Read,
            145 => WriteByte,
            146 => Write,
            147 => Open,
            148 => Close,
            149 => Flush,
            /* Instructions for threads */
            150 => ThreadSpawn,
            151 => Remainder,
            /* Instruction for clearing flags */
            152 => Clear,
            /* Instructions operation on integers and floats */
            153 => AddFI,
            154 => SubFI,
            155 => MulFI,
            156 => DivFI,
            157 => AddIF,
            158 => SubIF,
            159 => MulIF,
            160 => DivIF,
            161 => AddUF,
            162 => SubUF,
            163 => MulUF,
            164 => DivUF,
            166 => RegMove,
            167 => RegMoveF,
            168 => RegMoveA,
            
            
            /* Instructions illegal instruction */
            _ => Illegal,
        }
    }
}


impl From<Opcode> for u16 {
    fn from(v: Opcode) -> Self {
        use Opcode::*;
        match v {
            Halt => 0,
            NoOp => 1,
            /* Instruction for loading values */
            DeRefReg => 2,
            DeRef => 3,
            Move => 4,
            Set => 5,
            /* Instructions for signed integers */
            AddI => 6,
            SubI => 7,
            MulI => 8,
            DivI => 9,
            EqI => 10,
            NeqI => 11,
            LtI => 12,
            GtI => 13,
            LeqI => 14,
            GeqI => 15,
            /* Instructions for unsigned integers */
            AddU => 16,
            SubU => 17,
            MulU => 18,
            DivU => 19,
            EqU => 20,
            NeqU => 21,
            LtU => 22,
            GtU => 23,
            LeqU => 24,
            GeqU => 25,
            /* Instructions for floating point numbers */
            DeRefRegF => 26,
            DeRefF => 27,
            MoveF => 28,
            SetF => 29,
            AddF => 30,
            SubF => 31,
            MulF => 32,
            DivF => 33,
            EqF => 34,
            NeqF => 35,
            LtF => 36,
            GtF => 37,
            LeqF => 38,
            GeqF => 39,
            /* Instructions for Atomic Integers */
            DeRefRegA => 40,
            DeRefA => 41,
            MoveA => 42,
            SetA => 43,
            /* Instructions for signed Atomic Integers */
            AddAI => 44,
            SubAI => 45,
            MulAI => 46,
            DivAI => 47,
            EqAI => 48,
            NeqAI => 49,
            LtAI => 50,
            GtAI => 51,
            LeqAI => 52,
            GeqAI => 53,
            /* Instructions for unsigned Atomic Integers */
            AddAU => 54,
            SubAU => 55,
            MulAU => 56,
            DivAU => 57,
            EqAU => 58,
            NeqAU => 59,
            LtAU => 60,
            GtAU => 61,
            LeqAU => 62,
            GeqAU => 63,
            /* Instructions for bitwise operations */
            And => 64,
            Or => 65,
            Xor => 66,
            Not => 67,
            ShiftLeft => 68,
            ShiftRight => 69,
            /* Instructions for jumping */
            Jump => 70,
            JumpEq => 71,
            JumpNeq => 72,
            JumpLt => 73,
            JumpGt => 74,
            JumpLeq => 75,
            JumpGeq => 76,
            JumpZero => 77,
            JumpNotZero => 78,
            JumpNeg => 79,
            JumpPos => 80,
            JumpEven => 81,
            JumpOdd => 82,
            JumpBack => 83,
            JumpForward => 84,
            JumpInfinity => 85,
            JumpNotInfinity => 86,
            JumpOverflow => 87,
            JumpUnderflow => 88,
            JumpNotOverflow => 89,
            JumpNotUnderflow => 90,
            JumpNaN => 91,
            JumpNotNaN => 92,
            JumpRemainder => 93,
            JumpNotRemainder => 94,
            // Gap for future instructions

            /* Instructions for function calls */
            Call => 109,
            Return => 110,
            /* Instructions for stack management */
            Pop => 112,
            Push => 113,
            PopF => 114,
            PushF => 115,
            PopA => 116,
            PushA => 117,
            /* Instructions for accessing the stack */
            DeRefRegStack => 118,
            DeRefStack => 119,
            MoveStack => 120,
            SetStack => 121,
            DeRefRegStackF => 122,
            DeRefStackF => 123,
            MoveStackF => 124,
            SetStackF => 125,
            DeRefRegStackA => 126,
            DeRefStackA => 127,
            MoveStackA => 128,
            SetStackA => 129,
            /* Instructions for accessing the stack accross cores */
            DeRefRegStackC => 130,
            DeRefStackC => 131,
            MoveStackC => 132,
            SetStackC => 133,
            DeRefRegStackCF => 134,
            DeRefStackCF => 135,
            MoveStackCF => 136,
            SetStackCF => 137,
            DeRefRegStackCA => 138,
            DeRefStackCA => 139,
            MoveStackCA => 140,
            SetStackCA => 141,
            /* Instructions for memory management */
            Malloc => 142,
            Free => 143,
            /* IO instructions */
            ReadByte => 144,
            /* Takes a string pointer and a length */
            Read => 145,
            WriteByte => 146,
            /* Takes a string pointer and a length */
            Write => 147,
            /* Takes a string pointer and a length */
            Open => 148,
            Close => 149,
            /* Takes a file descriptor */
            Flush => 150,
            /* Instructions for threads */
            ThreadSpawn => 151,
            Remainder => 152,
            /* Instruction for clearing flags */
            Clear => 153,
            /* Instructions operation on integers and floats */
            AddFI => 154,
            SubFI => 155,
            MulFI => 156,
            DivFI => 157,
            AddIF => 158,
            SubIF => 159,
            MulIF => 160,
            DivIF => 161,
            AddUF => 162,
            SubUF => 163,
            MulUF => 164,
            DivUF => 165,
            RegMove => 166,
            RegMoveF => 167,
            RegMoveA => 168,
            





            /* Instructions illegal instruction */
            Illegal => 65535,
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
