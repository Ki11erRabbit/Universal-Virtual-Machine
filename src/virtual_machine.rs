
use std::sync::{Arc, RwLock};
use std::sync::TryLockResult;
use std::sync::TryLockError;
use std::thread::{self,JoinHandle};
use std::sync::atomic::AtomicU64;
use std::array::from_fn;

use crate::instruction::Opcode;


pub enum RegisterType {
    Register64,
    Register128,
    RegisterF32,
    RegisterF64,
    RegisterAtomic64,
}

pub enum Fault {
    ProgramLock,
    InvalidOperation,
    InvalidSize,
    InvalidRegister(usize,RegisterType),// (register, type)
    InvalidMemory,
    InvalidAddress(u64),
    DivideByZero,

}


pub struct Machine {
    heap: Arc<RwLock<Vec<u8>>>,
    cores: Vec<Arc<RwLock<Core>>>,
    core_threads: Vec<JoinHandle<Result<(),Fault>>>,
    program: Arc<RwLock<Vec<u8>>>,
}



const REGISTER_64_COUNT: usize = 16;
const REGISTER_128_COUNT: usize = 8;
const REGISTER_F32_COUNT: usize = 8;
const REGISTER_F64_COUNT: usize = 8;
const REGISTER_ATOMIC_64_COUNT: usize = 8;


pub struct Core {
    /* 64-bit registers */
    registers_64: [u64; REGISTER_64_COUNT],
    /* 128-bit registers */
    registers_128: [u128; REGISTER_128_COUNT],
    /* floating point registers */
    registers_f32: [f32; REGISTER_F32_COUNT],
    registers_f64: [f64; REGISTER_F64_COUNT],
    /* Atomic registers */
    registers_atomic_64: [AtomicU64; REGISTER_ATOMIC_64_COUNT],
    /* flags */
    parity_flag: bool,
    zero_flag: bool,
    sign_flag: bool,
    /* other */
    program_counter: usize,
    pipeline_counter: usize,
    stack: Vec<u8>,
    program: Arc<RwLock<Vec<u8>>>,
    pipeline: Vec<u8>,
    memory: Arc<RwLock<Vec<u8>>>,
}
    
impl Core {
    pub fn new(memory: Arc<RwLock<Vec<u8>>>, program: Arc<RwLock<Vec<u8>>>,program_counter: usize) -> Core {
        Core {
            registers_64: [0; 16],
            registers_128: [0; 8],
            registers_f32: [0.0; 8],
            registers_f64: [0.0; 8],
            registers_atomic_64: from_fn(|_| AtomicU64::new(0)),
            parity_flag: false,
            zero_flag: false,
            sign_flag: false,
            program_counter,
            pipeline_counter: 0,
            stack: Vec::new(),
            pipeline: Vec::new(),
            program,
            memory,
        }
    }

    pub fn run(&mut self) -> Result<(),Fault> {
        let mut is_done = false;
        while !is_done {
            is_done = self.execute_instruction()?;
        }
        Ok(())
    }

    pub fn run_once(&mut self) -> Result<(),Fault> {
        self.execute_instruction()?;
        Ok(())
    }

    #[inline]
    fn check_program_counter(&self) -> Result<bool,Fault> {
        loop {
            match self.program.try_read() {
                Ok(program) => {
                    if self.program_counter >= program.len() {
                        return Ok(true);
                    }
                },
                TryLockResult::Err(TryLockError::WouldBlock) => {
                    thread::yield_now();
                    continue;
                },
                Err(_) => {
                    return Err(Fault::ProgramLock);
                },
            }
        }
    }

    fn decode_opcode(&mut self) -> Opcode {
        let opcode = Opcode::from(self.pipeline[self.pipeline_counter] as u16);
        self.program_counter += 2;
        self.pipeline_counter += 2;
        return opcode;

    }

    fn execute_instruction(&mut self) -> Result<bool, Fault> {
        if self.check_program_counter()? {
            return Ok(true);
        }
        use Opcode::*;
        match self.decode_opcode() {
            Halt | NoOp => return Ok(true),
            Load => {
                let size = self.pipeline[self.pipeline_counter] as usize;
                self.pipeline_counter += 1;
                self.program_counter += 1;
                let register = self.pipeline[self.pipeline_counter] as usize;
                self.pipeline_counter += 1;
                self.program_counter += 1;
                match size {
                    8 => {
                        let value = self.pipeline[self.pipeline_counter] as u8;
                        if register >= REGISTER_64_COUNT {
                            return Err(Fault::InvalidRegister(register, RegisterType::Register64));
                        }
                        self.registers_64[register] = value as u64;
                        self.pipeline_counter += 1;
                        self.program_counter += 1;
                    },
                    16 => {
                        let value = self.pipeline[self.pipeline_counter] as u16;
                        if register >= REGISTER_64_COUNT {
                            return Err(Fault::InvalidRegister(register, RegisterType::Register64));
                        }
                        self.registers_64[register] = value as u64;
                        self.pipeline_counter += 2;
                        self.program_counter += 2;
                    },
                    32 => {
                        let value = self.pipeline[self.pipeline_counter] as u32;
                        if register >= REGISTER_64_COUNT {
                            return Err(Fault::InvalidRegister(register, RegisterType::Register64));
                        }
                        self.registers_64[register] = value as u64;
                        self.pipeline_counter += 4;
                        self.program_counter += 4;
                    },
                    64 => {
                        let value = self.pipeline[self.pipeline_counter] as u64;
                        if register >= REGISTER_64_COUNT {
                            return Err(Fault::InvalidRegister(register, RegisterType::Register64));
                        }
                        self.registers_64[register] = value as u64;
                        self.pipeline_counter += 8;
                        self.program_counter += 8;
                    },
                    128 => {
                        let value = self.pipeline[self.pipeline_counter] as u128;
                        if register >= REGISTER_128_COUNT {
                            return Err(Fault::InvalidRegister(register, RegisterType::Register128));
                        }
                        self.registers_128[register] = value as u128;
                        self.pipeline_counter += 16;
                        self.program_counter += 16;
                    },
                    _ => return Err(Fault::InvalidSize),
                }
                
            },
            DeRef => {
                let size = self.pipeline[self.pipeline_counter] as usize;
                self.pipeline_counter += 1;
                self.program_counter += 1;
                let register = self.pipeline[self.pipeline_counter] as usize;
                self.pipeline_counter += 1;
                self.program_counter += 1;
                let address = self.registers_64[register];
                let memory = self.memory.read().unwrap();
                match size {
                    8 => {
                        if address >= memory.len() as u64 && address >= self.stack.len() as u64 {
                            return Err(Fault::InvalidAddress(address));
                        }
                        self.registers_64[register] = (memory[address as usize] as u8) as u64;
                    },
                    16 => {
                        if address >= memory.len() as u64 - 1 {
                            return Err(Fault::InvalidAddress(address));
                        }
                        self.registers_64[register] = (memory[address as usize] as u16) as u64;
                    },
                    32 => {
                        if address >= memory.len() as u64 - 3 {
                            return Err(Fault::InvalidAddress(address));
                        }
                        self.registers_64[register] = (memory[address as usize] as u32) as u64;
                    },
                    64 => {
                        if address >= memory.len() as u64 - 7 {
                            return Err(Fault::InvalidAddress(address));
                        }
                        self.registers_64[register] = (memory[address as usize] as u64) as u64;
                    },
                    128 => {
                        if address >= memory.len() as u64 - 15 {
                            return Err(Fault::InvalidAddress(address));
                        }
                        self.registers_128[register] = (memory[address as usize] as u128) as u128;
                    },
                    _ => return Err(Fault::InvalidSize),
                }
            },

            _ => return Err(Fault::InvalidOperation),
            

        }

        Ok(false)
            
    }
}
