#![allow(arithmetic_overflow)]

use std::sync::{Arc, RwLock};
use std::thread::{self,JoinHandle};
use std::fmt;

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
        }
    }
}


pub struct Machine {
    memory: Arc<RwLock<Vec<u8>>>,
    cores: Vec<Core>,
    core_threads: Vec<JoinHandle<Result<(),Fault>>>,
    program: Option<Arc<Vec<u8>>>,
}

impl Machine {

    pub fn new() -> Machine {
        Machine {
            memory: Arc::new(RwLock::new(Vec::new())),
            cores: Vec::new(),
            core_threads: Vec::new(),
            program: None,
        }
    }

    pub fn new_with_cores(core_count: usize) -> Machine {
        let mut machine = Machine::new();
        for _ in 0..core_count {
            machine.add_core();
        }
        machine
    }

    
    pub fn run(&mut self, program_counter: usize) {
        //TODO: change print to log
        self.run_core(0, program_counter);
        let mut finished_cores = Vec::new();
        loop {
            for (core_num, core_thread) in self.core_threads.iter().enumerate() {
                if core_thread.is_finished() {
                    finished_cores.push(core_num);
                    
                }
            }

            for core_num in finished_cores.iter().rev() {
                let core = self.core_threads.remove(*core_num);

                match core.join() {
                    Ok(result) => {
                        match result {
                            Ok(_) => println!("Core {} finished", core_num),
                            Err(fault) => println!("Core {} faulted with: {}", core_num, fault),
                        }
                    },
                    Err(_) => println!("Core {} panicked", core_num),
                }
            }
            finished_cores.clear();

            //TODO: check for commands from the threads to do various things like allocate more memory, etc.
            if self.core_threads.len() == 0 {
                break;
            }
        }
        
    }

    pub fn add_core(&mut self) {
        let core = Core::new(self.memory.clone());
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




}

