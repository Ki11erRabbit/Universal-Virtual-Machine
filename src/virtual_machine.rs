#![allow(arithmetic_overflow)]

use std::sync::{Arc, RwLock};
use std::thread::{self,JoinHandle};

use crate::core::Core;

#[derive(Debug,PartialEq)]
pub enum RegisterType {
    Register64,
    Register128,
    RegisterF32,
    RegisterF64,
    RegisterAtomic64,
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

}


pub struct Machine {
    heap: Arc<RwLock<Vec<u8>>>,
    //cores: Vec<Arc<RwLock<Core>>>,
    core_threads: Vec<JoinHandle<Result<(),Fault>>>,
    program: Option<Arc<Vec<u8>>>,
}

impl Machine {

    pub fn new() -> Machine {
        Machine {
            heap: Arc::new(RwLock::new(Vec::new())),
     //       cores: Vec::new(),
            core_threads: Vec::new(),
            program: None,
        }
    }

    pub fn add_core(&mut self, core: Core, program_counter: usize) {
        let core_thread = {
            thread::spawn(move || {
                let mut core = core;
                core.run(program_counter)
            })
        };
        //self.cores.push(core);
        self.core_threads.push(core_thread);
    }

    pub fn add_program(&mut self, program: Vec<u8>) {
        self.program = Some(Arc::new(program));
    }

    pub fn core_count(&self) -> usize {
        self.core_threads.len()
    }

    pub fn run_single(&mut self) -> Result<(),Fault> {
        let mut core = Core::new(self.heap.clone(),self.program.clone().unwrap());
        core.run(0)
    }


}

