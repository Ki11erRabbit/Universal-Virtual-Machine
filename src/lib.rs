//pub mod assembler;
pub mod instruction;
pub mod virtual_machine;
pub mod core;
pub mod assembler;
pub mod binary;
pub mod log;
mod garbage_collector;

use crate::core::REGISTER_128_COUNT;
use crate::core::REGISTER_64_COUNT;
use crate::core::REGISTER_F32_COUNT;
use crate::core::REGISTER_F64_COUNT;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::RwLock;
use std::any::Any;
use std::fmt;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::Sender;
use std::io::{Read, Write};

use instruction::Opcode;
use virtual_machine::Memory;

/// The id to identify a core
pub type CoreId = u8;
/// A byte
pub type Byte = u8;
/// A pointer is an index to a memory location
pub type Pointer = u64;
/// A file descriptor is an index to a file
/// The reason why this isn't signed or 32 bits is because we have more flavorful errors than C.
pub type FileDescriptor = u64;


/// A foreign function that can be called from the virtual machine
pub type ForeignFunction = Arc<fn(&mut dyn Core, Option<Arc<RwLock<dyn Any + Send + Sync>>>)-> SimpleResult>;
/// The argument to a foreign function
pub type ForeignFunctionArg = Option<Arc<RwLock<dyn Any + Send + Sync>>>;

pub type Registers = ([u64; REGISTER_64_COUNT],[u128; REGISTER_128_COUNT],[f32;REGISTER_F32_COUNT],[f64;REGISTER_F64_COUNT]);

//pub type FFun = Box<dyn FnMut(&mut Core) -> Result<(),Fault> + Send + Sync + 'static>;
//fn(&mut Core) -> Result<(), Fault>
/// Messages that a core and the machine can send to each other
#[derive(Clone)]
pub enum Message {
    /// Takes a size of memory to allocate
    Malloc(u64),                               // Takes size
    /// Takes a pointer and a size to reallocate
    Realloc(Pointer, u64),                     // Takes pointer and size
    /// Return message for (re)allocation
    MemoryPointer(Pointer),                    // Returns pointer
    /// Takes a pointer to free
    Free(Pointer),                             // Takes pointer
    /// Takes a core id and a pointer to dereference
    DereferenceStackPointer(CoreId,Pointer),   // Takes core and pointer
    /// Returns dereferenced memory
    DereferencedMemory(Vec<Byte>),             // Returns dereferenced memory
    /// Takes a filename in bytes and a flag to open a file
    OpenFile(Vec<Byte>,u8),                    // Takes filename, and flags
    /// Return message for open file
    FileDescriptor(FileDescriptor),            // Returns file descriptor
    /// Takes a file descriptor and amount to read
    ReadFile(FileDescriptor,u64),              // Takes file descriptor and amount to read
    /// Returns read data and amount read
    FileData(Vec<Byte>, u64),                  // Returns read data, and amount read
    /// Takes a file descriptor and data to write
    WriteFile(FileDescriptor,Vec<u8>),                    // Takes file descriptor and data to write
    /// Takes a file descriptor to close
    CloseFile(FileDescriptor),                            // Takes file descriptor
    /// Takes a file descriptor to flush
    Flush(FileDescriptor),                                // Takes file descriptor
    /// Return message for close file
    FileClosed,                                // Returns file closed
    /// Takes a program address to spawn a thread
    SpawnThread(Pointer, Registers),                      // Takes address of function to call
    /// Return message for spawn thread
    ThreadSpawned(CoreId),                     // Returns core id of spawned thread
    /// Takes a core id to mark as done
    ThreadDone(CoreId),                        // Returns thread done, then with return value as bytes, then the type of return value
    /// Takes a core id to join
    JoinThread(CoreId),                        // Takes core id of thread to join
    /// Takes a core id to detach
    DetachThread(CoreId),                      // Takes core id of thread to detach
    /// Message for when an error occurs
    Error(Fault),                              // Returns error
    /// Message for when an action succeeds
    Success,                                   // Returns success
    /// Takes an index to a foreign function
    GetForeignFunction(u64),                   // Takes function name
    /// Return message for foreign function
    ForeignFunction(ForeignFunctionArg, ForeignFunction),// Returns function and its arguments
    /// Signal to initiate garbage collection
    CollectGarbage,
    StackPointer(Pointer),
    StackPointers(Vec<Pointer>),
    RetryMessage(Box<Message>),
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
            Message::SpawnThread(address,_) => write!(f, "SpawnThread({})", address),
            Message::ThreadSpawned(core_id) => write!(f, "ThreadSpawned({})", core_id),
            Message::ThreadDone(core_id) => write!(f, "ThreadDone({})", core_id),
            Message::JoinThread(core_id) => write!(f, "JoinThread({})", core_id),
            Message::DetachThread(core_id) => write!(f, "DetachThread({})", core_id),
            Message::Error(fault) => write!(f, "Error({})", fault),
            Message::Success => write!(f, "Success"),
            Message::GetForeignFunction(address) => write!(f, "GetForeignFunction({})", address),
            Message::ForeignFunction(_, _) => write!(f, "ForeignFunction"),
            Message::CollectGarbage => write!(f, "CollectGarbage"),
            Message::StackPointer(pointer) => write!(f, "StackPointer({})", pointer),
            Message::StackPointers(pointers) => write!(f, "StackPointers({:?})", pointers),
            Message::RetryMessage(message) => write!(f, "RetryMessage({:?})", message),
        }
    }
}


#[derive(Debug,PartialEq, Clone)]
/// The type of register, this is used with the Fault enum
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



#[derive(Debug,PartialEq, Clone)]
/// The faults that a core can run into
/// Returning these is so that we don't have to cause the core to panic
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
    FileReadError,
    InvalidFree,
    InvalidRealloc,
    SegmentationFault,
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
            Fault::FileReadError => write!(f, "File Read Error"),
            Fault::InvalidFree => write!(f, "Invalid Free"),
            Fault::InvalidRealloc => write!(f, "Invalid Realloc"),
            Fault::SegmentationFault => write!(f, "Segmentation Fault"),
        }
    }
}

pub type CoreResult<T> = Result<T, Fault>;
pub type SimpleResult = CoreResult<()>;

pub trait Core {
    /// Runs the core
    fn run(&mut self, program_counter: usize) -> SimpleResult;
    /// Runs the core once
    fn run_once(&mut self) -> SimpleResult;

    /// Takes a program which is just a vector of bytes
    fn add_program(&mut self, program: Arc<Vec<Byte>>);

    fn add_channels(&mut self, machine_send: Sender<Message>, core_receive: Receiver<Message>);

    fn send_message(&self, message: Message) -> SimpleResult;

    fn recv_message(&mut self) -> CoreResult<Message>;

    fn check_messages(&mut self) -> SimpleResult;

    fn check_program_counter(&self) -> CoreResult<bool>;

    fn decode_opcode(&mut self) -> Opcode;

    fn get_register_64<'input>(&'input mut self, register: usize) -> CoreResult<&'input mut u64>;

    fn get_register_128<'input>(&'input mut self, register: usize) -> CoreResult<&'input mut u128>;

    fn get_register_f32<'input>(&'input mut self, register: usize) -> CoreResult<&'input mut f32>;

    fn get_register_f64<'input>(&'input mut self, register: usize) -> CoreResult<&'input mut f64>;
    
}

pub trait RegCore: Core {
    fn add_data_segment(&mut self, data: Arc<Vec<Byte>>);

    fn add_heap(&mut self, memory: Arc<RwLock<Vec<Byte>>>);

    fn add_stack(&mut self, stack: WholeStack, index: usize);

    fn set_registers(&mut self, registers: Registers);

    fn set_core_id(&mut self, core_id: usize);
}

pub trait GarbageCollectorCore: Core + Collector {}

pub trait Collector {
    fn add_stack(&mut self, stack: WholeStack);

    fn add_heap(&mut self, memory: Arc<RwLock<Memory>>);

    fn add_data_segment(&mut self, data: Arc<Vec<Byte>>);
}

pub trait ReadWrite: Read + Write {}

impl<T: Read + Write> ReadWrite for T {}

pub type Stack = Arc<RwLock<Box<[Byte]>>>;
pub type WholeStack = Arc<Vec<Stack>>;


#[macro_export]
macro_rules! access_heap {
    ($method:expr, $id:ident ,$block:block, $err:block) => {
        loop {
            match $method {
                Ok($id) => {$block; break;},
                Err(std::sync::TryLockError::WouldBlock) => {
                    std::thread::yield_now();
                },
                Err(_) => $err,
            }
        }
    };
}

#[macro_export]
macro_rules! get_heap_len_err {
    ($method:expr, $var:ident) => {
        crate::access_heap!($method.try_read(), heap, {
            $var = heap.len();
        }, {
            return Err(Fault::CorruptedMemory);
        })
    };
}

#[macro_export]
macro_rules! get_heap_len_panic {
    ($method:expr, $var:ident) => {
        crate::access_heap!($method.try_read(), heap, {
            $var = heap.len();
        }, {
            panic!("Corrupted Memory");
        })
    };
}

#[macro_export]
macro_rules! unsigned_t_signed {
    ($int:expr, $start:ty, $end:ty) => {
        <$end>::from_le_bytes(($int as $start).to_le_bytes())
    };
}
