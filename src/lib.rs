//pub mod assembler;
pub mod instruction;
pub mod virtual_machine;
pub mod core;
pub mod assembler;
pub mod binary;

use crate::core::Core;
use std::fmt;

pub type CoreId = u8;
pub type Byte = u8;
pub type Pointer = u64;


#[derive(Debug,PartialEq)]
pub enum RegisterType {
    Register64,
    Register128,
    RegisterF32,
    RegisterF64,
    RegisterAtomic64,
}

impl fmt::Display for RegisterType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            RegisterType::Register64 => write!(f, "Register64"),
            RegisterType::Register128 => write!(f, "Register128"),
            RegisterType::RegisterF32 => write!(f, "RegisterF32"),
            RegisterType::RegisterF64 => write!(f, "RegisterF64"),
            RegisterType::RegisterAtomic64 => write!(f, "RegisterAtomic64"),
        }
    }
}

//pub type FFun = Box<dyn FnMut(&mut Core) -> Result<(),Fault> + Send + Sync + 'static>;

/// Messages that a core and the machine can send to each other
#[derive(Clone)]
pub enum Message {
    Malloc(u64),                               // Takes size
    MemoryPointer(Pointer),                    // Returns pointer
    DeallocateMemory(Pointer),                 // Takes pointer
    DereferenceStackPointer(CoreId,Pointer),   // Takes core and pointer
    DereferencedMemory(Vec<Byte>),             // Returns dereferenced memory
    OpenFile(Vec<Byte>,u8),                    // Takes filename, and flags
    FileDescriptor(i64),                       // Returns file descriptor
    ReadFile(i64,u64),                         // Takes file descriptor and amount to read
    FileData(Vec<Byte>, u64),                  // Returns read data, and amount read
    WriteFile(i64,Vec<u8>),                    // Takes file descriptor and data to write
    CloseFile(i64),                            // Takes file descriptor
    Flush(i64),                                // Takes file descriptor
    FileClosed,                                // Returns file closed
    SpawnThread(Pointer),                      // Takes address of function to call
    ThreadSpawned(CoreId),                     // Returns core id of spawned thread
    ThreadDone(CoreId),                        // Returns thread done, then with return value as bytes, then the type of return value
    JoinThread(CoreId),                        // Takes core id of thread to join
    DetachThread(CoreId),                      // Takes core id of thread to detach

    Error(Fault),                              // Returns error
    Success,                                   // Returns success
    GetForeignFunction(u64),                   // Takes function name
    ForeignFunction(Box<dyn FnOnce(&mut Core) -> Result<(), Fault>>),// Returns function
}


#[derive(Debug,PartialEq)]
pub enum Fault {
    ProgramLock,
    InvalidOperation,
    InvalidSize,
    InvalidRegister(usize,RegisterType),// (register, type)
    InvalidMemory,
    InvalidAddress(Pointer),
    DivideByZero,
    CorruptedMemory,
    InvalidFileDescriptor,
    InvalidJump,
    StackOverflow,
    StackUnderflow,
    MemoryOutOfBounds,
    MachineCrash(&'static str),
    FileOpenError,
    InvalidMessage,
    FileWriteError,

}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Fault::ProgramLock => write!(f, "Program Lock"),
            Fault::InvalidOperation => write!(f, "Invalid Operation"),
            Fault::InvalidSize => write!(f, "Invalid Size"),
            Fault::InvalidRegister(register, register_type) => write!(f, "Invalid Register {} of type {}", register, register_type),
            Fault::InvalidMemory => write!(f, "Invalid Memory"),
            Fault::InvalidAddress(address) => write!(f, "Invalid Address {}", address),
            Fault::DivideByZero => write!(f, "Divide By Zero"),
            Fault::CorruptedMemory => write!(f, "Corrupted Memory"),
            Fault::InvalidFileDescriptor => write!(f, "Invalid File Descriptor"),
            Fault::InvalidJump => write!(f, "Invalid Jump"),
            Fault::StackOverflow => write!(f, "Stack Overflow"),
            Fault::StackUnderflow => write!(f, "Stack Underflow"),
            Fault::MemoryOutOfBounds => write!(f, "Memory Out Of Bounds"),
            Fault::MachineCrash(msg) => write!(f, "Machine Crash: {}", msg),
            Fault::FileOpenError => write!(f, "File Open Error"),
            Fault::InvalidMessage => write!(f, "Invalid Message"),
            Fault::FileWriteError => write!(f, "File Write Error"),
        }
    }
}
