/// # Assembly Instructions
/// All integer constants are unsigned 8 bit integers as the default
/// All floating point constants are 32 bit floats as the default
/// To specify the type the constant should be, use Rust constant syntax
/// ## Example
/// ```
///0.0f64 // This is a 64 bit float
///0.0f32 // This is a 32 bit float
///0u8 // This is an 8 bit unsigned integer
///0u16 // This is a 16 bit unsigned integer
///```
/// To specify a register, use the `$` symbol followed by a number to specify the register
/// ## Example
/// ```
///$0 // This is register 0
///$1 // This is register 1
///```
/// To specify an address, use the `#` symbol followed by a number to specify the address.
/// ## Example
/// ```
///#0 // This is address 0
///#1 // This is address 1
///```
/// Labels can be used in the place of constants in order to make the assembler figure out an address.
/// ## Example
/// ```
/// move 64, $0, ref_table
///```
///
/// This is a table of all the register types and their sizes
///
/// | Register       | Size |
/// |----------------|------|
/// | 64 bit int     | 16   |
/// |----------------|------|
/// | 128 bit int    | 8    |
/// |----------------|------|
/// | 32 bit float   | 8    |
/// |----------------|------|
/// | 64 bit float   | 8    |
///
/// This the value in the angle brackets represents the name of the value in the instruction.
/// ## Example
///`add <size>u8, $<dest>, $<src>`
///
///`<size>` is the size of the operation. The trailing `u8` is used to specify the size of the constant.
///
///`<dest>` is the destination register and in this case one of the operands.
///
///`<src>` is the source register and in this case the other operand.
///
/// ## Halt
/// `halt`
/// ## No Op
/// `noop`
/// ## Move
/// These instructions are for integers.
/// ### Dereferencing from a register
/// `move <size>u8, $<value>, $<pointer>, <offset>i64`
/// ### Dereferencing a hard coded address
/// `move <size>u8, $<value>, <address>`
/// ### Moving a value from a register to memory
/// `move <size>u8, $<pointer>, $<value>`
/// ### Loading a constant into a register
/// `move <size>u8, $<value>, <constant>`
/// ### Moving between registers
/// `move $<dest>, $<src>, <size>u8`
/// ## MoveF
/// These instructions are for floating point numbers.
/// ### Dereferencing from a register
/// `movef <size>u8, $<value>, $<pointer>, <offset>i64`
/// ### Dereferencing a hard coded address
/// `movef <size>u8, $<value>, <address>`
/// ### Moving a value from a register to memory
/// `movef <size>u8, $<pointer>, $<value>`
/// ### Loading a constant into a register
/// `movef <size>u8, $<value>, <constant>`
/// ### Moving between registers
/// `movef $<dest>, $<src>, <size>u8`
/// ## Math and Comparison Instructions
/// These instructions are for integers.
/// ### Addition
/// #### With Registers
/// `add <size>u8, $<dest>, $<src>`
/// #### With Constants
/// `add <size>u8, $<dest>, <constant>`
/// ### Subtraction
/// #### With Registers
/// `sub <size>u8, $<dest>, $<src>`
/// #### With Constants
/// `sub <size>u8, $<dest>, <constant>`
/// ### Multiplication
/// #### With Registers
/// `mul <size>u8, $<dest>, $<src>`
/// #### With Constants
/// `mul <size>u8, $<dest>, <constant>`
/// ### Division
/// #### With Registers
/// `div <size>u8, $<dest>, $<src>`
/// #### With Constants
/// `div <size>u8, $<dest>, <constant>`
/// ### Equality
/// #### With Registers
/// `eq <size>u8, $<a>, $<b>`
/// #### With Constants
/// `eq <size>u8, $<a>, <constant>`
/// ### Inequality
/// #### With Registers
/// `neq <size>u8, $<a>, $<b>`
/// #### With Constants
/// `neq <size>u8, $<a>, <constant>`
/// ### Less Than
/// #### With Registers
/// `lt <size>u8, $<a>, $<b>`
/// #### With Constants
/// `lt <size>u8, $<a>, <constant>`
/// ### Greater Than
/// #### With Registers
/// `gt <size>u8, $<a>, $<b>`
/// #### With Constants
/// `gt <size>u8, $<a>, <constant>`
/// ### Less Than or Equal To
/// #### With Registers
/// `leq <size>u8, $<a>, $<b>`
/// #### With Constants
/// `leq <size>u8, $<a>, <constant>`
/// ### Greater Than or Equal To
/// #### With Registers
/// `geq <size>u8, $<a>, $<b>`
/// #### With Constants
/// `geq <size>u8, $<a>, <constant>`
/// ## Math and Comparison Instructions
/// These instructions are for floating point numbers.
/// ### Addition
/// #### With Registers
/// `addf <size>u8, $<dest>, $<src>`
/// #### With Constants
/// `addf <size>u8, $<dest>, <constant>`
/// ### Subtraction
/// #### With Registers
/// `subf <size>u8, $<dest>, $<src>`
/// #### With Constants
/// `subf <size>u8, $<dest>, <constant>`
/// ### Multiplication
/// #### With Registers
/// `mulf <size>u8, $<dest>, $<src>`
/// #### With Constants
/// `mulf <size>u8, $<dest>, <constant>`
/// ### Division
/// #### With Registers
/// `divf <size>u8, $<dest>, $<src>`
/// #### With Constants
/// `divf <size>u8, $<dest>, <constant>`
/// ### Equality
/// #### With Registers
/// `eqf <size>u8, $<a>, $<b>`
/// #### With Constants
/// `eqf <size>u8, $<a>, <constant>`
/// ### Inequality
/// #### With Registers
/// `neqf <size>u8, $<a>, $<b>`
/// #### With Constants
/// `neqf <size>u8, $<a>, <constant>`
/// ### Less Than
/// #### With Registers
/// `ltf <size>u8, $<a>, $<b>`
/// #### With Constants
/// `ltf <size>u8, $<a>, <constant>`
/// ### Greater Than
/// #### With Registers
/// `gtf <size>u8, $<a>, $<b>`
/// #### With Constants
/// `gtf <size>u8, $<a>, <constant>`
/// ### Less Than or Equal To
/// #### With Registers
/// `leqf <size>u8, $<a>, $<b>`
/// #### With Constants
/// `leqf <size>u8, $<a>, <constant>`
/// ### Greater Than or Equal To
/// #### With Registers
/// `geqf <size>u8, $<a>, $<b>`
/// #### With Constants
/// `geqf <size>u8, $<a>, <constant>`
/// ## Bitwise Instructions
/// These instructions are for integers.
/// ### And
/// #### With Registers
/// `and <size>u8, $<dest>, $<src>`
/// #### With Constants
/// `and <size>u8, $<dest>, <constant>`
/// ### Or
/// #### With Registers
/// `or <size>u8, $<dest>, $<src>`
/// #### With Constants
/// `or <size>u8, $<dest>, <constant>`
/// ### Xor
/// #### With Registers
/// `xor <size>u8, $<dest>, $<src>`
/// #### With Constants
/// `xor <size>u8, $<dest>, <constant>`
/// ### Not
/// `not <size>u8, $<dest>`
/// ### Shift Left
/// #### With Registers
/// `shl <size>u8, $<value>, $<amount>`
/// #### With Constants
/// `shl <size>u8, $<value>, <amount>u8`
/// ### Shift Right
/// #### With Registers
/// `shr <size>u8, $<value>, $<amount>`
/// #### With Constants
/// `shr <size>u8, $<value>, <amount>u8`
/// ## Math and Comparison Instructions
/// These instructions are for floating point numbers.
/// ### Addition
/// #### With Registers
/// `addf <size>u8, $<dest>, $<src>`
/// #### With Constants
/// `addf <size>u8, $<dest>, <constant>`
/// ### Subtraction
/// #### With Registers
/// `subf <size>u8, $<dest>, $<src>`
/// #### With Constants
/// `subf <size>u8, $<dest>, <constant>`
/// ### Multiplication
/// #### With Registers
/// `mulf <size>u8, $<dest>, $<src>`
/// #### With Constants
/// `mulf <size>u8, $<dest>, <constant>`
/// ### Division
/// #### With Registers
/// `divf <size>u8, $<dest>, $<src>`
/// #### With Constants
/// `divf <size>u8, $<dest>, <constant>`
/// ### Equality
/// #### With Registers
/// `eqf <size>u8, $<a>, $<b>`
/// #### With Constants
/// `eqf <size>u8, $<a>, <constant>`
/// ### Inequality
/// #### With Registers
/// `neqf <size>u8, $<a>, $<b>`
/// #### With Constants
/// `neqf <size>u8, $<a>, <constant>`
/// ### Less Than
/// #### With Registers
/// `ltf <size>u8, $<a>, $<b>`
/// #### With Constants
/// `ltf <size>u8, $<a>, <constant>`
/// ### Greater Than
/// #### With Registers
/// `gtf <size>u8, $<a>, $<b>`
/// #### With Constants
/// `gtf <size>u8, $<a>, <constant>`
/// ### Less Than or Equal To
/// #### With Registers
/// `leqf <size>u8, $<a>, $<b>`
/// #### With Constants
/// `leqf <size>u8, $<a>, <constant>`
/// ### Greater Than or Equal To
/// #### With Registers
/// `geqf <size>u8, $<a>, $<b>`
/// #### With Constants
/// `geqf <size>u8, $<a>, <constant>`
/// ## Math instructions between floating point and integer
/// ### Integer to Floating Point
/// #### Addition
/// ##### With Registers
/// `addfi <size>u8, $<dest>, $<src>`
/// ##### With Constants
/// `addfi <size>u8, $<dest>, <constant>`
/// #### Subtraction
/// ##### With Registers
/// `subfi <size>u8, $<dest>, $<src>`
/// ##### With Constants
/// `subfi <size>u8, $<dest>, <constant>`
/// #### Multiplication
/// ##### With Registers
/// `mulf <size>u8, $<dest>, $<src>`
/// ##### With Constants
/// `mulf <size>u8, $<dest>, <constant>`
/// #### Division
/// ##### With Registers
/// `divfi <size>u8, $<dest>, $<src>`
/// ##### With Constants
/// `divfi <size>u8, $<dest>, <constant>`
/// ### Floating Point to Integer
/// #### Addition
/// ##### With Registers
/// `addif <size>u8, $<dest>, $<src>`
/// ##### With Constants
/// `addif <size>u8, $<dest>, <constant>`
/// #### Subtraction
/// ##### With Registers
/// `subif <size>u8, $<dest>, $<src>`
/// ##### With Constants
/// `subif <size>u8, $<dest>, <constant>`
/// #### Multiplication
/// ##### With Registers
/// `mulif <size>u8, $<dest>, $<src>`
/// ##### With Constants
/// `mulif <size>u8, $<dest>, <constant>`
/// #### Division
/// ##### With Registers
/// `divif <size>u8, $<dest>, $<src>`
/// ##### With Constants
/// `divif <size>u8, $<dest>, <constant>`
/// ## Jump Instructions
/// ### Jump
/// `jump <constant>u64`
/// ### Jump If Equal
/// `jumpeq <constant>u64`
/// ### Jump If Not Equal
/// `jumpneq <constant>u64`
/// ### Jump If Less Than
/// `jumplt <constant>u64`
/// ### Jump If Greater Than
/// `jumpgt <constant>u64`
/// ### Jump If Less Than or Equal To
/// `jumpleq <constant>u64`
/// ### Jump If Greater Than or Equal To
/// `jumpgeq <constant>u64`
/// ### Jump If Zero
/// `jumpzero <constant>u64`
/// ### Jump If Not Zero
/// `jumpnzero <constant>u64`
/// ### Jump If Negative
/// `jumpneg <constant>u64`
/// ### Jump If Positive
/// `jumppos <constant>u64`
/// ### Jump If Even
/// `jumpeven <constant>u64`
/// ### Jump If Odd
/// `jumpodd <constant>u64`
/// ### Jump If Infinity
/// `jumpinf <constant>u64`
/// ### Jump If Not Infinity
/// `jumpninf <constant>u64`
/// ### Jump If NaN
/// `jumpnan <constant>u64`
/// ### Jump If Not NaN
/// `jumpnnan <constant>u64`
/// ### Jump If Overflow
/// `jumpoverflow <constant>u64`
/// ### Jump If Not Overflow
/// `jumpnoverflow <constant>u64`
/// ### Jump If Underflow
/// `jumpunderflow <constant>u64`
/// ### Jump If Not Underflow
/// `jumpnunderflow <constant>u64`
/// ### Jump If there is a Remainder
/// `jumprmndr <constant>u64`
/// ### Jump If there is No Remainder
/// `jumpnrmndr <constant>u64`
/// ### Jump Back n Instructions
/// `jumpback <constant>u64`
/// ### Jump Forward n Instructions
/// `jumpforward <constant>u64`
/// ## Call Instructions
/// ### Call
/// `call <constant>u64`
/// ### Call an Arbitrary Address
/// `callarb $<address>`
/// ### Return
/// `ret`
/// ### Call a Foreign Function
/// `foreign $<function id>`
/// ## Stack Instructions
/// ### Push
/// #### For integers
/// `push <size>u8, $<value>`
/// #### For floats
/// `pushf <size>u8, $<value>`
/// ### Pop
/// #### For integers
/// `pop <size>u8, $<dest>`
/// #### For floats
/// `popf <size>u8, $<dest>`
/// ## Heap Instructions
/// ### Malloc
/// `malloc $<dest ptr>, $<size>`
/// ### Realloc
/// `realloc $<dest/src ptr>, $<size>`
/// ### Free
/// `free $<ptr>`
/// ## IO Instructions
/// ### Read
/// #### Read Byte
/// `readbyte $<dest>, $<fd>`
/// #### Read
/// `read $<fd>, $<dest ptr>, $<amount>`
/// ### Write
/// #### Write Byte
/// `writebyte $<fd>, $<src>`
/// #### Write
/// `write $<fd>, $<src ptr>, $<length>`
/// ### Open
/// `open $<dest fd>, $<path ptr>, $<size> ,$<flags>`
/// ### Close
/// `close $<fd>`
/// ### Flush
/// `flush $<fd>`
/// ## Thread Instructions
/// ### Thread Spawn
/// `threadspawn $<core id dest>, $<function addr>`
/// ### Thread Join
/// `threadjoin $<core id>`
/// ### Thread Detach
/// `threaddetach $<core id>`
/// ## Random Instructions
/// ### Random Integer
/// `rand <size>u8, $<dest>`
/// ### Random Float
/// `randf <size>u8, $<dest>`
/// ## Sleep Instructions
/// ### With Registers
/// `sleep $<time sec>, $<scale>`
/// ### With Constants
/// `sleep <time sec>u64, <scale>u64`
/// ## Misc Instructions
/// ### Clear
/// resets all the core flags to their default values
///
/// `clear`
/// ### Reset
/// resets all the registers to zero
///
/// `reset`
/// ### String Length
/// Gets the length of a string that is null terminated
///
/// `strlen $<dest>, $<src ptr>`
/// ### Remainder
/// Gets the remainder of a division operation
///
/// `rmndr $<dest>, $<src>`







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
    /// This instruction does integer addition on two registers and stores it in the first
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Neq,
    Lt,
    Gt,
    Leq,
    Geq,
    /* Instructions for unsigned integers */
    /// This instruction does integer addition with a register and a constant and stores it in the register
    AddC,
    /// This instruction does integer subtraction with a register and a constant and stores it in the register
    SubC,
    MulC,
    DivC,
    EqC,
    NeqC,
    LtC,
    GtC,
    LeqC,
    GeqC,
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
    /* Instructions for floating point numbers with constants */
    /// This instruction does floating point addition on a register and a constant and stores it in the register
    AddFC,
    /// This instruction does floating point subtraction on a register and a constant and stores it in the register
    SubFC,
    /// This instruction does floating point multiplication on a register and a constant and stores it in the register
    MulFC,
    /// This instruction does floating point division on a register and a constant and stores it in the register
    DivFC,
    /// This instruction does floating point equality on a register and a constant and flips the right flags
    EqFC,
    /// This instruction does floating point inequality on a register and a constant and flips the right flags
    NeqFC,
    /// This instruction does floating point less than on a register and a constant and flips the right flags
    LtFC,
    /// This instruction does floating point greater than on a register and a constant and flips the right flags
    GtFC,
    /// This instruction does floating point less than or equal on a register and a constant and flips the right flags
    LeqFC,
    /// This instruction does floating point greater than or equal on a register and a constant and flips the right flags
    GeqFC,
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
    /// This instruction calls an arbitrary function from memory specified by a register
    CallArb,
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
    RegMove,
    /// This instruction will move the value of a register to another register but for floats
    RegMoveF,
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
    /// This instruction will sleep for a specified amount of time in seconds with a scale factor
    Sleep,
    /// This instruction will sleep for a specified amount of time seconds from the register
    SleepReg,
    /// This function generates a random integer
    Random,
    /// This function generates a random float
    RandomF,
    AndC,
    OrC,
    XorC,
    ShiftLeftC,
    ShiftRightC,
    AddFIC,
    SubFIC,
    MulFIC,
    DivFIC,
    AddIFC,
    SubIFC,
    MulIFC,
    DivIFC,
    /// This instruction will clear all the registers
    Reset,
    StrLen,
    
    
    
    

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
            /* Instructions for integers */
            6 => Add,
            7 => Sub,
            8 => Mul,
            9 => Div,
            10 => Eq,
            11 => Neq,
            12 => Lt,
            13 => Gt,
            14 => Leq,
            15 => Geq,
            /* Instructions for integers with constants */
            16 => AddC,
            17 => SubC,
            18 => MulC,
            19 => DivC,
            20 => EqC,
            21 => NeqC,
            22 => LtC,
            23 => GtC,
            24 => LeqC,
            25 => GeqC,
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
            /* Instruction for floating point numbers with constants */
            40 => AddFC,
            41 => SubFC,
            42 => MulFC,
            43 => DivFC,
            44 => EqFC,
            45 => NeqFC,
            46 => LtFC,
            47 => GtFC,
            48 => LeqFC,
            49 => GeqFC,
            /* Instructions for Bitwise operations and constants */
            50 => AndC,
            51 => OrC,
            52 => XorC,
            53 => ShiftLeftC,
            54 => ShiftRightC,
            /* Gap starts at 55 */
            /* Gap ends at 63 */
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
            108 => CallArb,
            109 => Call,
            110 => Return,
            /* Instructions for stack management */
            111 => Pop,
            112 => Push,
            113 => PopF,
            114 => PushF,
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
            /* Instructions 155, 156, 157, 158 are reserved for future use */
            159 => RegMove,
            160 => RegMoveF,
            161 => Reset,
            /* Special Return for Threads */
            162 => ThreadReturn,
            163 => ThreadJoin,
            164 => ThreadDetach,
            165 => StackPointer,
            166 => ForeignCall,
            167 => Realloc,
            168 => Sleep,
            169 => SleepReg,
            170 => Random,
            171 => RandomF,

            172 => AddFIC,
            173 => SubFIC,
            174 => MulFIC,
            175 => DivFIC,
            176 => AddIFC,
            177 => SubIFC,
            178 => MulIFC,
            179 => DivIFC,
            180 => StrLen,
            
            
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
