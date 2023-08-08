


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
    /// This Jump occurs if the last comparison was greater than
    JumpGt,
    /// This Jump occurs if the last comparison was less than or equal
    JumpLeq,
    /// This Jump occurs if the last comparison was greater than or equal
    JumpGeq,
    /// This Jump occurs if the last operation resulted in a zero
    JumpZero,
    /// This Jump occurs if the last operation resulted in a non-zero
    JumpNotZero,
    /// This Jump occurs if the last operation resulted in a negative number
    JumpNeg,
    /// This Jump occurs if the last operation resulted in a positive number
    JumpPos,
    /// This Jump occurs if the last operation resulted in an even number
    JumpEven,
    /// This Jump occurs if the last operation resulted in an odd number
    JumpOdd,
    /// This is an unconditional jump backwards n spaces in the program
    JumpBack,
    /// This is an unconditional jump forwards n spaces in the program
    JumpForward,
    /// This Jump occurs if a floating point calculation resulted in infinity
    JumpInfinity,
    /// This Jump occurs if a floating point calculation did not result in infinity
    JumpNotInfinity,
    /// This Jump occurs if we overflow an integer
    JumpOverflow,
    /// This Jump occurs if we underflow an integer
    JumpUnderflow,
    /// This Jump occurs if we don't overflow an integer
    JumpNotOverflow,
    /// This Jump occurs if we don't underflow an integer
    JumpNotUnderflow,
    /// This Jump occurs if a floating point calculation resulted in NaN
    JumpNaN,
    /// This Jump occurs if a floating point calculation did not result in NaN
    JumpNotNaN,
    /// This jump occurs if an integer division resulted in a remainder
    JumpRemainder,
    /// This jump occurs if we don't have a remainder
    JumpNotRemainder,
    

    /* Instructions for function calls */
    /// This instruction calls a function
    /// This instruction will load the current program counter value onto the stack
    Call,
    /// This instruction will return from a function
    /// This instruction will pop the current program counter value off the stack
    Return,
    /* Instructions for stack management */
    /// This instruction will pop an integer off the stack with a given size
    Pop,
    /// This instruction will push an integer onto the stack with a given size
    Push,
    /// This instruction will pop a float off the stack
    PopF,
    /// This instruction will push a float onto the stack
    PushF,
    /// This instruction will pop an atomic integer off the stack
    PopA,
    /// This instruction will push an atomic integer onto the stack
    PushA,
    /* Instructions for accessing the stack */
    /// This is the DeRefReg instruction for the stack
    DeRefRegStack,
    /// This is the DeRef instruction for the stack
    DeRefStack,
    /// This is the Move instruction for the stack
    MoveStack,
    /// This is the DeRefReg instruction for the stack but with floats
    DeRefRegStackF,
    /// This is the DeRef instruction for the stack but with floats
    DeRefStackF,
    /// This is the Move instruction for the stack but with floats
    MoveStackF,
    /// This is the DeRefReg instruction for the stack but with atomic integers
    DeRefRegStackA,
    /// This is the DeRef instruction for the stack but with atomic integers
    DeRefStackA,
    /// This is the Move instruction for the stack but with atomic integers
    MoveStackA,
    /* Instructions for accessing the stack accross cores */
    /// This is the DeRefReg instruction for the stack but for integers accross cores
    DeRefRegStackC,
    /// This is the DeRef instruction for the stack but for integers accross cores
    DeRefStackC,
    /// This is the Move instruction for the stack but for integers accross cores
    MoveStackC,
    /// This is the DeRefReg instruction for the stack but for floats accross cores
    DeRefRegStackCF,
    /// This is the DeRef instruction for the stack but for floats accross cores
    DeRefStackCF,
    /// This is the Move instruction for the stack but for floats accross cores
    MoveStackCF,
    /// This is the DeRefReg instruction for the stack but for atomic integers accross cores
    DeRefRegStackCA,
    /// This is the DeRef instruction for the stack but for atomic integers accross cores
    DeRefStackCA,
    /// This is the Move instruction for the stack but for atomic integers accross cores
    MoveStackCA,
    /* Instructions for memory management */
    /// This instruction will allocate a block of memory and return a pointer to it
    Malloc,
    /// This instruction will free a block of memory
    Free,
    /// This instruction will reallocate a block of memory
    Realloc,
    /* IO instructions */
    /// This instruction will read a byte from a file descriptor
    ReadByte,
    /* Takes a string pointer and a length */
    /// This instruction will read a string from a file descriptor
    Read,
    /// This instruction will write a byte to a file descriptor
    WriteByte,
    /* Takes a string pointer and a length */
    /// This instruction will write a string to a file descriptor
    Write,
    /* Takes a string pointer and a length */
    /// This instruction will open a file
    Open,
    /// This instruction will close a file
    Close,
    /* Takes a file descriptor */
    /// This instruction will flush a file
    Flush,
    /* Instructions for threads */
    /// This instruction will spawn a thread
    ThreadSpawn,
    /// This function will get the remainder of an operation an reset the flag
    Remainder,
    /* Instruction for clearing flags */
    /// This instruction will clear the flags
    Clear,
    /* Instructions operation on integers and floats */
    /// This instruction will add a signed integer to a float
    AddFI,
    /// This instruction will subtract a signed integer from a float
    SubFI,
    /// This instruction will multiply a signed integer with a float
    MulFI,
    /// This instruction will divide a float by a signed integer
    DivFI,
    /// This instruction will add float to a signed integer
    AddIF,
    /// This instruction will subtract a float from a signed integer
    SubIF,
    /// This instruction will multiply a float with a signed integer
    MulIF,
    /// This instruction will divide a signed integer by a float
    DivIF,
    /// This instruction will add a float to an unsigned integer
    AddUF,
    /// This instruction will subtract a float from an unsigned integer
    SubUF,
    /// This instruction will multiply a float with an unsigned integer
    MulUF,
    /// This instruction will divide an unsigned integer by a float
    DivUF,
    /// This instruction will move the value of a register to another register
    RegMove,
    /// This instruction will move the value of a register to another register but for floats
    RegMoveF,
    /// This instruction will move the value of a register to another register but for atomic integers
    RegMoveA,
    /* Special Return for threads */
    /// This instruction will notify the machine that a thread has finished
    ThreadReturn,
    /* Other Thread Instructions */
    /// This instruction will join a thread and block until it is finished
    ThreadJoin,
    /// This instruction will detach a thread
    ThreadDetach,

    /// This instruction will copy the stack pointer into a register
    StackPointer,
    /// This instruction will call a specified foreign function
    ForeignCall,
    
    
    
    

    /* Instructions illegal instruction */
    /// This represents an illegal instruction
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
