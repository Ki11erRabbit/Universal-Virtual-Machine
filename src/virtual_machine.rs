#![allow(arithmetic_overflow)]

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, RwLock};
use std::thread::{self,JoinHandle};
use std::fmt;
use std::sync::mpsc::{Sender,Receiver, channel};
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Write;

use crate::binary::Binary;
use crate::core::Core;


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
/// Messages that a core and the machine can send to each other
#[derive(Debug,PartialEq)]
pub enum Message {
    Malloc(u64),                     // Takes size
    MemoryPointer(u64),              // Returns pointer
    DeallocateMemory(u64),           // Takes pointer
    DereferenceStackPointer(u8,u64), // Takes core and pointer
    DereferencedMemory(Vec<u8>),     // Returns dereferenced memory
    OpenFile(Vec<u8>,u8),            // Takes filename, and flags
    FileDescriptor(i64),             // Returns file descriptor
    ReadFile(i64,u64),               // Takes file descriptor and amount to read
    FileData(Vec<u8>, u64),          // Returns read data, and amount read
    WriteFile(i64,Vec<u8>),          // Takes file descriptor and data to write
    CloseFile(i64),                  // Takes file descriptor
    Flush(i64),                      // Takes file descriptor
    FileClosed,                      // Returns file closed
    SpawnThread(u64),                // Takes address of function to call
    ThreadSpawned(u8),               // Returns core id of spawned thread
    ThreadDone,                      // Returns thread done
    Error(Fault),                    // Returns error
    Success,                         // Returns success
}


#[derive(Debug,PartialEq)]
pub enum Fault {
    ProgramLock,
    InvalidOperation,
    InvalidSize,
    InvalidRegister(usize,RegisterType),// (register, type)
    InvalidMemory,
    InvalidAddress(u64),
    DivideByZero,
    CorruptedMemory,
    InvalidFileDescriptor,
    InvalidJump,
    StackOverflow,
    StackUnderflow,
    MemoryOutOfBounds,
    MachineCrash,
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
            Fault::MachineCrash => write!(f, "Machine Crash"),
            Fault::FileOpenError => write!(f, "File Open Error"),
            Fault::InvalidMessage => write!(f, "Invalid Message"),
            Fault::FileWriteError => write!(f, "File Write Error"),
        }
    }
}


pub struct Machine {
    memory: Arc<RwLock<Vec<u8>>>,
    cores: Vec<Core>,
    core_threads: Vec<JoinHandle<Result<(),Fault>>>,
    program: Option<Arc<Vec<u8>>>,
    entry_point: Option<usize>,
    channels: Rc<RefCell<Vec<(Sender<Message>, Receiver<Message>)>>>,
    files: Vec<Box<dyn Write>>,
}

impl Machine {

    pub fn new() -> Machine {
        Machine {
            memory: Arc::new(RwLock::new(Vec::new())),
            cores: Vec::new(),
            core_threads: Vec::new(),
            program: None,
            entry_point: None,
            channels: Rc::new(RefCell::new(Vec::new())),
            files: vec![Box::new(std::io::stdout()), Box::new(std::io::stderr())],
        }
    }

    pub fn new_with_cores(core_count: usize) -> Machine {
        let mut machine = Machine::new();
        for _ in 0..core_count {
            machine.add_core();
        }
        machine
    }

    pub fn run_at(&mut self, program_counter: usize) {
        self.entry_point = Some(program_counter);
        self.run();
    }
    
    pub fn run(&mut self) {
        //TODO: change print to log
        let program_counter = self.entry_point.expect("Entry point not set");
        self.run_core(0, program_counter);
        loop {
            self.check_finished_cores();
            self.check_messages();

            //TODO: check for commands from the threads to do various things like allocate more memory, etc.
            if self.core_threads.len() == 0 {
                break;
            }
        }
        
    }

    fn check_finished_cores(&mut self) {
        let mut finished_cores = Vec::new();
        for (core_num, core_thread) in self.core_threads.iter().enumerate() {
            if core_thread.is_finished() {
                finished_cores.push(core_num);

            }
        }

        for core_num in finished_cores.iter().rev() {
            let core = self.core_threads.remove(*core_num);
            let _ = self.channels.borrow_mut().remove(*core_num);

            match core.join() {
                Ok(result) => {
                    match result {
                        Ok(_) => eprintln!("Core {} finished", core_num),
                        Err(fault) => eprintln!("Core {} faulted with: {}", core_num, fault),
                    }
                },
                Err(_) => eprintln!("Core {} panicked", core_num),
            }
        }
        finished_cores.clear();

    }

    fn check_messages(&mut self) {
        for (send, recv) in self.channels.clone().borrow().iter() {
            match recv.try_recv() {
                Ok(message) => {
                    match message {
                        Message::OpenFile(filename, flag) => {
                            let message = self.open_file(filename, flag);
                            send.send(message).unwrap();
                        },
                        Message::WriteFile(fd, data) => {
                            let message = self.write_file(fd, data);
                            send.send(message).unwrap();
                        },
                        Message::CloseFile(fd) => {
                            let message = self.close_file(fd);
                            send.send(message).unwrap();
                        },
                        Message::Flush(fd) => {
                            let message = self.flush(fd);
                            send.send(message).unwrap();
                        },
                        _ => unimplemented!(),

                    }

                },
                Err(_) => continue,
            }

            
        }

    }

    fn write_file(&mut self, fd: i64, data: Vec<u8>) -> Message {
        let fd = fd as usize - 1;
        if fd >= self.files.len() {
            return Message::Error(Fault::InvalidFileDescriptor);
        }
        let file = &mut self.files[fd];
        match file.write(&data) {
            Ok(_) => Message::Success,
            Err(_) => Message::Error(Fault::FileWriteError),
        }
    }

    fn flush(&mut self, fd: i64) -> Message {
        let fd = fd as usize - 1;
        if fd >= self.files.len() {
            return Message::Error(Fault::InvalidFileDescriptor);
        }
        let file = &mut self.files[fd];
        match file.flush() {
            Ok(_) => Message::Success,
            Err(_) => Message::Error(Fault::FileWriteError),
        }
    }

    fn close_file(&mut self, fd: i64) -> Message {
        let fd = fd as usize - 1;
        if fd >= self.files.len() {
            return Message::Error(Fault::InvalidFileDescriptor);
        }

        self.files.remove(fd);
        Message::Success
    }

    fn open_file(&mut self, filename: Vec<u8>, flag: u8) -> Message {
        let mut file = OpenOptions::new();
        let file = file.create(true);
        let file = match flag as char {
            't' => file.truncate(true),
            'a' => file.append(true),
            'w' => file.write(true),
            'r' | _ => file.read(true),
        };

        let filename = String::from_utf8(filename);
        match filename {
            Ok(filename) => {
                match file.open(filename) {
                    Ok(file) => {
                        self.files.push(Box::new(file));
                        Message::FileDescriptor(self.files.len() as i64)
                    },
                    Err(_) => Message::Error(Fault::FileOpenError),
                }
            },
            Err(_) => Message::Error(Fault::InvalidMessage),
        }
    }

    pub fn add_core(&mut self) {
        let (core_sender, core_receiver) = channel();
        let (machine_sender, machine_receiver) = channel();
        self.channels.borrow_mut().push((core_sender, machine_receiver));
        let core = Core::new(self.memory.clone(), machine_sender, core_receiver,);
        self.cores.push(core);
    }

    pub fn run_core(&mut self, core: usize, program_counter: usize) {
        let mut core = self.cores.remove(core);
        core.add_program(self.program.as_ref().expect("Program Not set").clone());
        let core_thread = {
            thread::spawn(move || {
                core.run(program_counter)
            })
        };
        self.core_threads.push(core_thread);
    }

    pub fn add_program(&mut self, program: Vec<u8>) {
        self.program = Some(Arc::new(program));

        self.memory.write().unwrap().extend_from_slice(&self.program.as_ref().unwrap()[0..]);
        
    }

    pub fn core_count(&self) -> usize {
        self.core_threads.len()
    }

    pub fn load_binary(&mut self, binary: &Binary) {
        self.program = Some(Arc::new(binary.program().clone()));
        self.memory.write().unwrap().extend_from_slice(&binary.data_segment());
        self.entry_point = Some(binary.entry_address());
    }



}

#[cfg(test)]
mod tests {
    use crate::assembler::generate_binary;
    use std::io::Read;

    use super::*;

    #[test]
    fn test_file_hello_world() {
        let input = "File{
.string \"hello.txt\"}
Msg{
.string \"Hello World!\"}
main{
move 64, $0, File
move 64, $1, 9u64
move 8, $2, 119
move 64, $3, Msg
move 64, $4, 12u64
open $0, $1, $2, $5
write $5, $3, $4
flush $5
close $5
ret
}
";
        let binary = generate_binary(input, "test").unwrap();

        let mut machine = Machine::new();

        machine.load_binary(&binary);

        machine.add_core();

        machine.run();

        assert_eq!(machine.files.len(), 2);
        let mut file = File::open("hello.txt").unwrap();

        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();

        assert_eq!(contents, "Hello World!");
    }

    #[test]
    fn test_dp_fibonacci() {
        let input = "array{
.u32 0u32
.u32 1u32
.u32 0u32
.u32 0u32
.u32 0u32
.u32 0u32
.u32 0u32
.u32 0u32
.u32 0u32
.u32 0u32
.u32 0u32
.u32 0u32
.u32 0u32
.u32 0u32
.u32 0u32
.u32 0u32}
main{
move 64, $0, array ; pointer to first element
move 64, $1, 4u64
addu 64, $0, $1
addu 64, $0, $1 ; pointer is now in 3rd element
move 32, $3, 10u32; end condition 
move 32, $4, 2u32; i variable
move 32, $10, 1u32
}
loop{
equ 32, $3, $4
jumpeq end
move 32, $5, $0, -4i64
move 32, $6, $0, -8i64
addu 32, $5, $6
move 32, $0, $5
addu 64, $0, $1
addu 32, $4, $10
jump loop
}
end{
ret}
";
        
        let binary = generate_binary(input, "test").unwrap();

        let mut machine = Machine::new();

        machine.load_binary(&binary);

        machine.add_core();

        machine.run();

        assert_eq!(machine.memory.read().unwrap()[..], [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 5, 0, 0, 0, 8, 0, 0, 0, 13, 0, 0, 0, 21, 0, 0, 0, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_recursive_fibonacci() {
        /*let input = "number{
.u32 0u32}
fibonacci{
move 32, $5, 1u32
lequ 32, $5, $1
jumpeq before-end
push 32, $1
subu 32, $1, $5
call fibonacci
pop 32, $1
subu 32, $1, $5
subu 32, $1, $5
push 32, $0
call fibonacci
pop 32, $1
addu 32, $0, $1
jump end}
before-end{
move $0, $1, 32}
end{
ret}
main{
move 32, $1, 5u32
call fibonacci
move 64, $3, number
move 32, $3, $0
ret
}
";*/
        let input = "number{
.u32 0u32}
fibonacci{
move 32, $2, 1u32
lequ 32, $1, $2
jumpgt rec
move $0, $1, 32
jump end
}
rec{
push 32, $1
subu 32, $1, $2
call fibonacci
pop 32, $1
push 32, $0
move 32, $2, 2u32
subu 32, $1, $2
call fibonacci
pop 32, $1
addu 32, $0, $1}
end{
ret}
main{
move 32, $1, 9u32
call fibonacci
move 64, $3, number
move 32, $3, $0
ret
}
";
            
        let binary = generate_binary(input, "test").unwrap();

        let mut machine = Machine::new();

        machine.load_binary(&binary);

        println!("{}", binary.assembly());
        println!("{}", binary.program_with_count());

        machine.add_core();

        machine.run();

        assert_eq!(machine.memory.read().unwrap()[..], [0, 0, 0, 34]);
        
    }

}
