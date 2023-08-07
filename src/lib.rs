//pub mod assembler;
pub mod instruction;
pub mod virtual_machine;
pub mod core;
pub mod assembler;
pub mod binary;

use crate::core::Core;
use std::sync::Arc;
use std::sync::RwLock;
use std::any::Any;
use std::fmt;

pub type CoreId = u8;
pub type Byte = u8;
pub type Pointer = u64;
pub type FileDescriptor = u64;


#[derive(Debug,PartialEq, Clone)]
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

pub type ForeignFunction = Arc<fn(&mut Core, Option<Arc<RwLock<dyn Any + Send + Sync>>>)-> Result<(),Fault>>;
pub type ForeignFunctionArg = Option<Arc<RwLock<dyn Any + Send + Sync>>>;

//pub type FFun = Box<dyn FnMut(&mut Core) -> Result<(),Fault> + Send + Sync + 'static>;
//fn(&mut Core) -> Result<(), Fault>
/// Messages that a core and the machine can send to each other
pub enum Message {
    Malloc(u64),                               // Takes size
    Realloc(Pointer, u64),                     // Takes pointer and size
    MemoryPointer(Pointer),                    // Returns pointer
    Free(Pointer),                             // Takes pointer
    DereferenceStackPointer(CoreId,Pointer),   // Takes core and pointer
    DereferencedMemory(Vec<Byte>),             // Returns dereferenced memory
    OpenFile(Vec<Byte>,u8),                    // Takes filename, and flags
    FileDescriptor(FileDescriptor),            // Returns file descriptor
    ReadFile(FileDescriptor,u64),              // Takes file descriptor and amount to read
    FileData(Vec<Byte>, u64),                  // Returns read data, and amount read
    WriteFile(FileDescriptor,Vec<u8>),                    // Takes file descriptor and data to write
    CloseFile(FileDescriptor),                            // Takes file descriptor
    Flush(FileDescriptor),                                // Takes file descriptor
    FileClosed,                                // Returns file closed
    SpawnThread(Pointer),                      // Takes address of function to call
    ThreadSpawned(CoreId),                     // Returns core id of spawned thread
    ThreadDone(CoreId),                        // Returns thread done, then with return value as bytes, then the type of return value
    JoinThread(CoreId),                        // Takes core id of thread to join
    DetachThread(CoreId),                      // Takes core id of thread to detach

    Error(Fault),                              // Returns error
    Success,                                   // Returns success
    GetForeignFunction(u64),                   // Takes function name
    ForeignFunction(ForeignFunctionArg, ForeignFunction),// Returns function and its arguments
}

impl fmt::Debug for Message {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Message::Malloc(size) => write!(f, "Malloc({})", size),
            Message::Realloc(pointer, size) => write!(f, "Realloc({}, {})", pointer, size),
            Message::MemoryPointer(pointer) => write!(f, "MemoryPointer({})", pointer),
            Message::Free(pointer) => write!(f, "DeallocateMemory({})", pointer),
            Message::DereferenceStackPointer(core_id,pointer) => write!(f, "DereferenceStackPointer({}, {})", core_id, pointer),
            Message::DereferencedMemory(data) => write!(f, "DereferencedMemory({:?})", data),
            Message::OpenFile(filename,flags) => write!(f, "OpenFile({:?}, {})", filename, flags),
            Message::FileDescriptor(fd) => write!(f, "FileDescriptor({})", fd),
            Message::ReadFile(fd,amount) => write!(f, "ReadFile({}, {})", fd, amount),
            Message::FileData(data,amount) => write!(f, "FileData({:?}, {})", data, amount),
            Message::WriteFile(fd,data) => write!(f, "WriteFile({}, {:?})", fd, data),
            Message::CloseFile(fd) => write!(f, "CloseFile({})", fd),
            Message::Flush(fd) => write!(f, "Flush({})", fd),
            Message::FileClosed => write!(f, "FileClosed"),
            Message::SpawnThread(address) => write!(f, "SpawnThread({})", address),
            Message::ThreadSpawned(core_id) => write!(f, "ThreadSpawned({})", core_id),
            Message::ThreadDone(core_id) => write!(f, "ThreadDone({})", core_id),
            Message::JoinThread(core_id) => write!(f, "JoinThread({})", core_id),
            Message::DetachThread(core_id) => write!(f, "DetachThread({})", core_id),
            Message::Error(fault) => write!(f, "Error({})", fault),
            Message::Success => write!(f, "Success"),
            Message::GetForeignFunction(address) => write!(f, "GetForeignFunction({})", address),
            Message::ForeignFunction(_, _) => write!(f, "ForeignFunction"),
        }
    }
}

#[derive(Debug,PartialEq, Clone)]
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
    InvalidFree,
    InvalidRealloc,

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
            Fault::InvalidFree => write!(f, "Invalid Free"),
            Fault::InvalidRealloc => write!(f, "Invalid Realloc"),
        }
    }
}
