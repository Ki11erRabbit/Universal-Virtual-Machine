


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// The different types of instructions
pub enum Opcode {
    /// The halt instruction
    /// This currently just causes the core to stop executing
    Halt,
    /// The no-op instruction
    /// This currently does the same thing as halt
    NoOp,
    /* Instruction for loading values */
    /// This instruction loads a value into a register from a pointer in another register with an offset in another register
    DeRefReg,
    /// This instruction loads a value into a register from a pointer in the instruction
    DeRef,
    /// This instruction moves a value from a register into memory
    Move,
    /// This instruction loads a constant into a register
    Set,
    /* Instructions for signed integers */
    /// This instruction does signed integer addition on two registers and stores it in the first
    AddI,
    /// This instruction does signed integer subtraction on two registers and stores it in the first
    SubI,
    /// This instruction does signed integer multiplication on two registers and stores it in the first
    MulI,
    /// This instruction does signed integer division on two registers and stores it in the first
    DivI,
    /// This instruction does signed integer equality on two registers and flips the right flags
    EqI,
    /// This instruction does signed integer inequality on two registers and flips the right flags
    NeqI,
    /// This instruction does signed integer less than on two registers and flips the right flags
    LtI,
    /// This instruction does signed integer greater than on two registers and flips the right flags
    GtI,
    /// This instruction does signed integer less than or equal on two registers and flips the right flags
    LeqI,
    /// This instruction does signed integer greater than or equal on two registers and flips the right flags
    GeqI,
    /* Instructions for unsigned integers */
    /// This instruction does unsigned integer addition on two registers and stores it in the first
    AddU,
    /// This instruction does unsigned integer subtraction on two registers and stores it in the first
    SubU,
    /// This instruction does unsigned integer multiplication on two registers and stores it in the first
    MulU,
    /// This instruction does unsigned integer division on two registers and stores it in the first
    DivU,
    /// This instruction does unsigned integer equality on two registers and flips the right flags
    EqU,
    /// This instruction does unsigned integer inequality on two registers and flips the right flags
    NeqU,
    /// This instruction does unsigned integer less than on two registers and flips the right flags
    LtU,
    /// This instruction does unsigned integer greater than on two registers and flips the right flags
    GtU,
    /// This instruction does unsigned integer less than or equal on two registers and flips the right flags
    LeqU,
    /// This instruction does unsigned integer greater than or equal on two registers and flips the right flags
    GeqU,
    /* Instructions for floating point numbers */
    /// This instruction is the same as DeRefReg but for floats
    DeRefRegF,
    /// This instruction is the same as DeRef but for floats
    DeRefF,
    /// This instruction is the same as Move but for floats
    MoveF,
    /// This instruction is the same as Set but for floats
    SetF,
    /// This instruction does floating point addition on two registers and stores it in the first
    AddF,
    /// This instruction does floating point subtraction on two registers and stores it in the first
    SubF,
    /// This instruction does floating point multiplication on two registers and stores it in the first
    MulF,
    /// This instruction does floating point division on two registers and stores it in the first
    DivF,
    /// This instruction does floating point equality on two registers and flips the right flags
    EqF,
    /// This instruction does floating point inequality on two registers and flips the right flags
    NeqF,
    /// This instruction does floating point less than on two registers and flips the right flags
    LtF,
    /// This instruction does floating point greater than on two registers and flips the right flags
    GtF,
    /// This instruction does floating point less than or equal on two registers and flips the right flags
    LeqF,
    /// This instruction does floating point greater than or equal on two registers and flips the right flags
    GeqF,
    /* Instructions for Atomic Integers */
    /// This instruction is the same as DeRefReg but for Atomic Integers
    DeRefRegA,
    /// This instruction is the same as DeRef but for Atomic Integers
    DeRefA,
    /// This instruction is the same as Move but for Atomic Integers
    MoveA,
    /// This instruction is the same as Set but for Atomic Integers
    SetA,
    /* Instructions for signed Atomic Integers */
    /// This instruction does signed integer addition on two registers and stores it in the first
    AddAI,
    /// This instruction does signed integer subtraction on two registers and stores it in the first
    SubAI,
    /// This instruction does signed integer multiplication on two registers and stores it in the first
    MulAI,
    /// This instruction does signed integer division on two registers and stores it in the first
    DivAI,
    /// This instruction does signed integer equality on two registers and flips the right flags
    EqAI,
    /// This instruction does signed integer inequality on two registers and flips the right flags
    NeqAI,
    /// This instruction does signed integer less than on two registers and flips the right flags
    LtAI,
    /// This instruction does signed integer greater than on two registers and flips the right flags
    GtAI,
    /// This instruction does signed integer less than or equal on two registers and flips the right flags
    LeqAI,
    /// This instruction does signed integer greater than or equal on two registers and flips the right flags
    GeqAI,
    /* Instructions for unsigned Atomic Integers */
    /// This instruction does unsigned integer addition on two registers and stores it in the first
    AddAU,
    /// This instruction does unsigned integer subtraction on two registers and stores it in the first
    SubAU,
    /// This instruction does unsigned integer multiplication on two registers and stores it in the first
    MulAU,
    /// This instruction does unsigned integer division on two registers and stores it in the first
    DivAU,
    /// This instruction does unsigned integer equality on two registers and flips the right flags
    EqAU,
    /// This instruction does unsigned integer inequality on two registers and flips the right flags
    NeqAU,
    /// This instruction does unsigned integer less than on two registers and flips the right flags
    LtAU,
    /// This instruction does unsigned integer greater than on two registers and flips the right flags
    GtAU,
    /// This instruction does unsigned integer less than or equal on two registers and flips the right flags
    LeqAU,
    /// This instruction does unsigned integer greater than or equal on two registers and flips the right flags
    GeqAU,
    /* Instructions for bitwise operations */
    /// This instruction does bitwise AND on two registers and stores it in the first
    And,
    /// This instruction does bitwise OR on two registers and stores it in the first
    Or,
    /// This instruction does bitwise XOR on two registers and stores it in the first
    Xor,
    /// This instruction does bitwise NOT on a register
    Not,
    /// This instruction does bitwise left shift a register
    ShiftLeft,
    /// This instruction does bitwise right shift a register
    ShiftRight,
    /* Instructions for jumping */
    /// This instruction is an unconditional jump
    Jump,
    /// This Jump occurs if the last comparison was equal
    JumpEq,
    /// This Jump occurs if the last comparison was not equal
    JumpNeq,
    /// This Jump occurs if the last comparison was less than
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
    Realloc,
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
    ForeignCall,
    
    
    
    

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
            166 => ForeignCall,
            167 => Realloc,
            
            
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
