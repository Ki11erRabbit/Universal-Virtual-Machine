
use std::sync::{Arc, RwLock};
use std::sync::TryLockResult;
use std::sync::TryLockError;
use std::thread::{self,JoinHandle};
use std::sync::atomic::AtomicU64;
use std::array::from_fn;

use crate::instruction::Opcode;

#[derive(Debug)]
pub enum RegisterType {
    Register64,
    Register128,
    RegisterF32,
    RegisterF64,
    RegisterAtomic64,
}

#[derive(Debug)]
pub enum Fault {
    ProgramLock,
    InvalidOperation,
    InvalidSize,
    InvalidRegister(usize,RegisterType),// (register, type)
    InvalidMemory,
    InvalidAddress(u64),
    DivideByZero,
    CorruptedMemory,

}


pub struct Machine {
    heap: Arc<RwLock<Vec<u8>>>,
    //cores: Vec<Arc<RwLock<Core>>>,
    core_threads: Vec<JoinHandle<Result<(),Fault>>>,
    program: Option<Arc<RwLock<Vec<u8>>>>,
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
        self.program = Some(Arc::new(RwLock::new(program)));
    }

    pub fn core_count(&self) -> usize {
        self.core_threads.len()
    }

    pub fn run_single(&mut self) -> Result<(),Fault> {
        let mut core = Core::new(self.heap.clone(),self.program.clone().unwrap());
        core.run(0)
    }


}


const REGISTER_64_COUNT: usize = 16;
const REGISTER_128_COUNT: usize = 8;
const REGISTER_F32_COUNT: usize = 8;
const REGISTER_F64_COUNT: usize = 8;
const REGISTER_ATOMIC_64_COUNT: usize = 8;

#[derive(Debug,PartialEq)]
pub enum Sign {
    Positive,
    Negative,
}

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
    odd_flag: bool,
    zero_flag: bool,
    sign_flag: Sign,
    overflow_flag: bool,
    /* other */
    remainder_64: usize,
    remainder_128: u128,
    program_counter: usize,
    pipeline_counter: usize,
    stack: Vec<u8>,
    program: Arc<RwLock<Vec<u8>>>,
    pipeline: Vec<u8>,
    memory: Arc<RwLock<Vec<u8>>>,
}
    
impl Core {
    pub fn new(memory: Arc<RwLock<Vec<u8>>>, program: Arc<RwLock<Vec<u8>>>) -> Core {
        Core {
            registers_64: [0; 16],
            registers_128: [0; 8],
            registers_f32: [0.0; 8],
            registers_f64: [0.0; 8],
            registers_atomic_64: from_fn(|_| AtomicU64::new(0)),
            remainder_64: 0,
            remainder_128: 0,
            odd_flag: false,
            zero_flag: false,
            sign_flag: Sign::Positive,
            overflow_flag: false,
            program_counter: 0,
            pipeline_counter: 0,
            stack: Vec::new(),
            pipeline: Vec::new(),
            program,
            memory,
        }
    }

    #[inline]
    fn advance_by_size(&mut self, size: usize) {
        self.program_counter += size;
        self.pipeline_counter += size;
    }
    fn advance_by_1_byte(&mut self) {
        self.advance_by_size(1);
    }
    fn advance_by_2_bytes(&mut self) {
        self.advance_by_size(2);
    }
    fn advance_by_4_bytes(&mut self) {
        self.advance_by_size(4);
    }
    fn advance_by_8_bytes(&mut self) {
        self.advance_by_size(8);
    }
    fn advance_by_16_bytes(&mut self) {
        self.advance_by_size(16);
    }

    pub fn run(&mut self, program_counter: usize) -> Result<(),Fault> {
        self.program_counter = program_counter;

        let pipeline = {
            let program = self.program.read().unwrap();
            program[..].to_vec().clone()
        };
        println!("Pipeline: {:?}",pipeline);
        println!("Program {:?}",self.program);
        self.pipeline = pipeline;
        
        let mut is_done = false;
        println!("Running core");
        while !is_done {
            println!("Running instruction");
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
            //println!("Checking program counter");
            //println!("Program {:?}",self.program);
            match self.program.try_read() {
                Ok(program) => {
                    if self.program_counter >= program.len() {
                        return Ok(true);
                    }
                    return Ok(false);
                },
                TryLockResult::Err(TryLockError::WouldBlock) => {
                    //thread::yield_now();
                    continue;
                },
                Err(_) => {
                    return Err(Fault::CorruptedMemory);
                },
            }
        }
    }

    fn decode_opcode(&mut self) -> Opcode {
        let opcode = Opcode::from(self.pipeline[self.pipeline_counter] as u16);
        self.advance_by_2_bytes();
        return opcode;

    }

    fn execute_instruction(&mut self) -> Result<bool, Fault> {
        if self.check_program_counter()? {
            println!("Program counter out of bounds");
            return Ok(true);
        }
        use Opcode::*;

        let opcode = self.decode_opcode();

        println!("Executing opcode: {:?}", opcode);
        
        match opcode {
            Halt | NoOp => return Ok(true),
            Set => self.set_opcode()?,
            DeRef => self.deref_opcode()?,
            Move => self.move_opcode()?,
            DeRefReg => self.derefreg_opcode()?,
            AddI => self.addi_opcode()?,
            SubI => self.subi_opcode()?,
            MulI => self.muli_opcode()?,
            DivI => self.divi_opcode()?,

            _ => return Err(Fault::InvalidOperation),
            

        }

        Ok(false)
            
    }


    fn set_opcode(&mut self) -> Result<(), Fault> {
        let size = self.pipeline[self.pipeline_counter] as usize;
        self.advance_by_1_byte();
        let register = self.pipeline[self.pipeline_counter] as usize;
        self.advance_by_1_byte();
        match size {
            8 => {
                let value = self.pipeline[self.pipeline_counter] as u8;
                if register >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register, RegisterType::Register64));
                }
                self.registers_64[register] = value as u64;
                self.advance_by_1_byte();
            },
            16 => {
                let value = self.pipeline[self.pipeline_counter] as u16;
                if register >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register, RegisterType::Register64));
                }
                self.registers_64[register] = value as u64;
                self.advance_by_2_bytes();
            },
            32 => {
                let value = self.pipeline[self.pipeline_counter] as u32;
                if register >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register, RegisterType::Register64));
                }
                self.registers_64[register] = value as u64;
                self.advance_by_4_bytes();
            },
            64 => {
                let value = self.pipeline[self.pipeline_counter] as u64;
                if register >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register, RegisterType::Register64));
                }
                self.registers_64[register] = value as u64;
                self.advance_by_8_bytes();
            },
            128 => {
                let value = self.pipeline[self.pipeline_counter] as u128;
                if register >= REGISTER_128_COUNT {
                    return Err(Fault::InvalidRegister(register, RegisterType::Register128));
                }
                self.registers_128[register] = value as u128;
                self.advance_by_16_bytes();
            },
            _ => return Err(Fault::InvalidSize),
        }
        Ok(())
    }

    fn deref_opcode(&mut self) -> Result<(), Fault> {
        let size = self.pipeline[self.pipeline_counter] as usize;
        self.advance_by_1_byte();
        let register = self.pipeline[self.pipeline_counter] as usize;
        self.advance_by_1_byte();
        let address = self.registers_64[register];
        match size {
            8 => {
                let memory = self.memory.read().unwrap();
                if address >= memory.len() as u64 {
                    return Err(Fault::InvalidAddress(address));
                }
                self.registers_64[register] = (memory[address as usize] as u8) as u64;
                drop(memory);
                self.advance_by_1_byte();
            },
            16 => {
                let memory = self.memory.read().unwrap();
                if address >= memory.len() as u64 - 1 {
                    return Err(Fault::InvalidAddress(address));
                }
                self.registers_64[register] = (memory[address as usize] as u16) as u64;
                drop(memory);
                self.advance_by_2_bytes();
            },
            32 => {
                let memory = self.memory.read().unwrap();
                if address >= memory.len() as u64 - 3 {
                    return Err(Fault::InvalidAddress(address));
                }
                self.registers_64[register] = (memory[address as usize] as u32) as u64;
                drop(memory);
                self.advance_by_4_bytes();
            },
            64 => {
                let memory = self.memory.read().unwrap();
                if address >= memory.len() as u64 - 7 {
                    return Err(Fault::InvalidAddress(address));
                }
                self.registers_64[register] = (memory[address as usize] as u64) as u64;
                drop(memory);
                self.advance_by_8_bytes();
            },
            128 => {
                let memory = self.memory.read().unwrap();
                if address >= memory.len() as u64 - 15 {
                    return Err(Fault::InvalidAddress(address));
                }
                self.registers_128[register] = (memory[address as usize] as u128) as u128;
                drop(memory);
                self.advance_by_16_bytes();
            },
            _ => return Err(Fault::InvalidSize),
        }
        Ok(())
    }

    fn move_opcode(&mut self) -> Result<(), Fault> {
        let size = self.pipeline[self.pipeline_counter] as usize;
        self.advance_by_1_byte();
        let register = self.pipeline[self.pipeline_counter] as usize;
        self.advance_by_1_byte();
        match size {
            8 => {
                if register >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register, RegisterType::Register64));
                }
                let address = self.pipeline[self.pipeline_counter] as u64;
                self.advance_by_8_bytes();
                loop {
                    match self.memory.try_write() {
                        Ok(mut memory) => {
                            if address >= memory.len() as u64 {
                                return Err(Fault::InvalidAddress(address));
                            }
                            memory[address as usize] = self.registers_64[register] as u8;
                            break;
                        },
                        Err(TryLockError::WouldBlock) => {
                            thread::yield_now();
                            continue;
                        },
                        Err(_) => {
                            return Err(Fault::CorruptedMemory);
                        },
                    }
                }
            },
            16 => {
                if register >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register, RegisterType::Register64));
                }
                let address = self.pipeline[self.pipeline_counter] as u64;
                self.advance_by_8_bytes();
                loop {
                    match self.memory.try_write() {
                        Ok(mut memory) => {
                            if address >= memory.len() as u64 - 1 {
                                return Err(Fault::InvalidAddress(address));
                            }
                            memory[address as usize] = (self.registers_64[register] >> 8) as u8;
                            memory[address as usize + 1] = self.registers_64[register] as u8;
                            break;
                        },
                        Err(TryLockError::WouldBlock) => {
                            thread::yield_now();
                            continue;
                        },
                        Err(_) => {
                            return Err(Fault::CorruptedMemory);
                        },
                    }
                }
            },
            32 => {
                if register >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register, RegisterType::Register64));
                }
                let address = self.pipeline[self.pipeline_counter] as u64;
                self.advance_by_8_bytes();
                loop {
                    match self.memory.try_write() {
                        Ok(mut memory) => {
                            if address >= memory.len() as u64 - 3 {
                                return Err(Fault::InvalidAddress(address));
                            }
                            memory[address as usize] = (self.registers_64[register] >> 24) as u8;
                            memory[address as usize + 1] = (self.registers_64[register] >> 16) as u8;
                            memory[address as usize + 2] = (self.registers_64[register] >> 8) as u8;
                            memory[address as usize + 3] = self.registers_64[register] as u8;
                            break;
                        },
                        Err(TryLockError::WouldBlock) => {
                            thread::yield_now();
                            continue;
                        },
                        Err(_) => {
                            return Err(Fault::CorruptedMemory);
                        },
                    }
                }
            },
            64 => {
                if register >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register, RegisterType::Register64));
                }
                let address = self.pipeline[self.pipeline_counter] as u64;
                self.advance_by_8_bytes();
                loop {
                    match self.memory.try_write() {
                        Ok(mut memory) => {
                            if address >= memory.len() as u64 - 7 {
                                return Err(Fault::InvalidAddress(address));
                            }
                            memory[address as usize] = (self.registers_64[register] >> 56) as u8;
                            memory[address as usize + 1] = (self.registers_64[register] >> 48) as u8;
                            memory[address as usize + 2] = (self.registers_64[register] >> 40) as u8;
                            memory[address as usize + 3] = (self.registers_64[register] >> 32) as u8;
                            memory[address as usize + 4] = (self.registers_64[register] >> 24) as u8;
                            memory[address as usize + 5] = (self.registers_64[register] >> 16) as u8;
                            memory[address as usize + 6] = (self.registers_64[register] >> 8) as u8;
                            memory[address as usize + 7] = self.registers_64[register] as u8;
                            break;
                        },
                        Err(TryLockError::WouldBlock) => {
                            thread::yield_now();
                            continue;
                        },
                        Err(_) => {
                            return Err(Fault::CorruptedMemory);
                        },
                    }
                }
            },
            128 => {
                if register >= REGISTER_128_COUNT {
                    return Err(Fault::InvalidRegister(register, RegisterType::Register128));
                }
                let address = self.pipeline[self.pipeline_counter] as u64;
                self.advance_by_8_bytes();
                loop {
                    match self.memory.try_write() {
                        Ok(mut memory) => {
                            if address >= memory.len() as u64 - 15 {
                                return Err(Fault::InvalidAddress(address));
                            }
                            memory[address as usize] = (self.registers_128[register] >> 120) as u8;
                            memory[address as usize + 1] = (self.registers_128[register] >> 112) as u8;
                            memory[address as usize + 2] = (self.registers_128[register] >> 104) as u8;
                            memory[address as usize + 3] = (self.registers_128[register] >> 96) as u8;
                            memory[address as usize + 4] = (self.registers_128[register] >> 88) as u8;
                            memory[address as usize + 5] = (self.registers_128[register] >> 80) as u8;
                            memory[address as usize + 6] = (self.registers_128[register] >> 72) as u8;
                            memory[address as usize + 7] = (self.registers_128[register] >> 64) as u8;
                            memory[address as usize + 8] = (self.registers_128[register] >> 56) as u8;
                            memory[address as usize + 9] = (self.registers_128[register] >> 48) as u8;
                            memory[address as usize + 10] = (self.registers_128[register] >> 40) as u8;
                            memory[address as usize + 11] = (self.registers_128[register] >> 32) as u8;
                            memory[address as usize + 12] = (self.registers_128[register] >> 24) as u8;
                            memory[address as usize + 13] = (self.registers_128[register] >> 16) as u8;
                            memory[address as usize + 14] = (self.registers_128[register] >> 8) as u8;
                            memory[address as usize + 15] = self.registers_128[register] as u8;
                            break;
                        },
                        Err(TryLockError::WouldBlock) => {
                            thread::yield_now();
                            continue;
                        },
                        Err(_) => {
                            return Err(Fault::CorruptedMemory);
                        },
                    }
                }

            },
            _ => return Err(Fault::InvalidSize),

        }
        Ok(())
    }

    fn derefreg_opcode(&mut self) -> Result<(), Fault> {
        let size = self.pipeline[self.pipeline_counter] as u8;
        self.advance_by_1_byte();
        let register = self.pipeline[self.pipeline_counter] as usize;
        self.advance_by_1_byte();
        let address_register = self.pipeline[self.pipeline_counter] as usize;
        if address_register >= REGISTER_64_COUNT {
            return Err(Fault::InvalidRegister(register, RegisterType::Register64));
        }
        self.advance_by_1_byte();
        let offset = self.pipeline[self.pipeline_counter] as i64;
        self.advance_by_8_bytes();
        let address = self.registers_64[address_register] as i64 + offset;
        let address = address as u64;

        match size {
            8 => {
                if register >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register, RegisterType::Register64));
                }

                loop {
                    match self.memory.try_read() {
                        Ok(memory) => {
                            if address >= memory.len() as u64 {
                                return Err(Fault::InvalidAddress(address));
                            }
                            self.registers_64[register] = (memory[address as usize] as u8) as u64;
                            break;
                        },
                        Err(TryLockError::WouldBlock) => {
                            thread::yield_now();
                            continue;
                        },
                        Err(_) => {
                            return Err(Fault::CorruptedMemory);
                        },
                    }
                }
                
            },
            16 => {
                if register >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register, RegisterType::Register64));
                }

                loop {
                    match self.memory.try_read() {
                        Ok(memory) => {
                            if address >= memory.len() as u64 - 1 {
                                return Err(Fault::InvalidAddress(address));
                            }
                            self.registers_64[register] = ((memory[address as usize] as u8) as u64) << 8;
                            self.registers_64[register] |= (memory[address as usize + 1] as u8) as u64;
                            break;
                        },
                        Err(TryLockError::WouldBlock) => {
                            thread::yield_now();
                            continue;
                        },
                        Err(_) => {
                            return Err(Fault::CorruptedMemory);
                        },
                    }
                }
            },
            32 => {
                if register >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register, RegisterType::Register64));
                }

                loop {
                    match self.memory.try_read() {
                        Ok(memory) => {
                            if address >= memory.len() as u64 - 3 {
                                return Err(Fault::InvalidAddress(address));
                            }
                            self.registers_64[register] = ((memory[address as usize] as u8) as u64) << 24;
                            self.registers_64[register] |= ((memory[address as usize + 1] as u8) as u64) << 16;
                            self.registers_64[register] |= ((memory[address as usize + 2] as u8) as u64) << 8;
                            self.registers_64[register] |= (memory[address as usize + 3] as u8) as u64;
                            break;
                        },
                        Err(TryLockError::WouldBlock) => {
                            thread::yield_now();
                            continue;
                        },
                        Err(_) => {
                            return Err(Fault::CorruptedMemory);
                        },
                    }
                }
            },
            64 => {
                if register >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register, RegisterType::Register64));
                }

                loop {
                    match self.memory.try_read() {
                        Ok(memory) => {
                            if address >= memory.len() as u64 - 7 {
                                return Err(Fault::InvalidAddress(address));
                            }
                            self.registers_64[register] = ((memory[address as usize] as u8) as u64) << 56;
                            self.registers_64[register] |= ((memory[address as usize + 1] as u8) as u64) << 48;
                            self.registers_64[register] |= ((memory[address as usize + 2] as u8) as u64) << 40;
                            self.registers_64[register] |= ((memory[address as usize + 3] as u8) as u64) << 32;
                            self.registers_64[register] |= ((memory[address as usize + 4] as u8) as u64) << 24;
                            self.registers_64[register] |= ((memory[address as usize + 5] as u8) as u64) << 16;
                            self.registers_64[register] |= ((memory[address as usize + 6] as u8) as u64) << 8;
                            self.registers_64[register] |= (memory[address as usize + 7] as u8) as u64;
                            break;
                        },
                        Err(TryLockError::WouldBlock) => {
                            thread::yield_now();
                            continue;
                        },
                        Err(_) => {
                            return Err(Fault::CorruptedMemory);
                        },
                    }
                }
            },
            128 => {
                if register >= REGISTER_128_COUNT {
                    return Err(Fault::InvalidRegister(register, RegisterType::Register128));
                }

                loop {
                    match self.memory.try_read() {
                        Ok(memory) => {
                            if address >= memory.len() as u64 - 15 {
                                return Err(Fault::InvalidAddress(address));
                            }
                            self.registers_128[register] = ((memory[address as usize] as u8) as u128) << 120;
                            self.registers_128[register] |= ((memory[address as usize + 1] as u8) as u128) << 112;
                            self.registers_128[register] |= ((memory[address as usize + 2] as u8) as u128) << 104;
                            self.registers_128[register] |= ((memory[address as usize + 3] as u8) as u128) << 96;
                            self.registers_128[register] |= ((memory[address as usize + 4] as u8) as u128) << 88;
                            self.registers_128[register] |= ((memory[address as usize + 5] as u8) as u128) << 80;
                            self.registers_128[register] |= ((memory[address as usize + 6] as u8) as u128) << 72;
                            self.registers_128[register] |= ((memory[address as usize + 7] as u8) as u128) << 64;
                            self.registers_128[register] |= ((memory[address as usize + 8] as u8) as u128) << 56;
                            self.registers_128[register] |= ((memory[address as usize + 9] as u8) as u128) << 48;
                            self.registers_128[register] |= ((memory[address as usize + 10] as u8) as u128) << 40;
                            self.registers_128[register] |= ((memory[address as usize + 11] as u8) as u128) << 32;
                            self.registers_128[register] |= ((memory[address as usize + 12] as u8) as u128) << 24;
                            self.registers_128[register] |= ((memory[address as usize + 13] as u8) as u128) << 16;
                            self.registers_128[register] |= ((memory[address as usize + 14] as u8) as u128) << 8;
                            self.registers_128[register] |= (memory[address as usize + 15] as u8) as u128;
                            break;
                        },
                        Err(TryLockError::WouldBlock) => {
                            thread::yield_now();
                            continue;
                        },
                        Err(_) => {
                            return Err(Fault::CorruptedMemory);
                        },
                    }
                }
            }
            _ => {
                return Err(Fault::InvalidSize);
            }

        }

        Ok(())
        
    }


    fn addi_opcode(&mut self) -> Result<(), Fault> {
        let size = self.pipeline[self.pipeline_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.pipeline[self.pipeline_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.pipeline[self.pipeline_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                if register1 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register1, RegisterType::Register64));
                }
                if register2 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register2, RegisterType::Register64));
                }
                let reg1_value = self.registers_64[register1] as i8;
                let reg2_value = self.registers_64[register2] as i8;

                let new_value = reg1_value + reg2_value;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if new_value % 2 != 0 {
                    self.odd_flag = true;
                }
                if new_value < 0 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                self.registers_64[register1] = new_value as u64;
            },
            16 => {
                if register1 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register1, RegisterType::Register64));
                }
                if register2 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register2, RegisterType::Register64));
                }
                let reg1_value = self.registers_64[register1] as i16;
                let reg2_value = self.registers_64[register2] as i16;

                let new_value = reg1_value + reg2_value;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if new_value % 2 != 0 {
                    self.odd_flag = true;
                }
                if new_value < 0 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                self.registers_64[register1] = new_value as u64;
            },

            32 => {
                if register1 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register1, RegisterType::Register64));
                }
                if register2 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register2, RegisterType::Register64));
                }
                let reg1_value = self.registers_64[register1] as i32;
                let reg2_value = self.registers_64[register2] as i32;

                let new_value = reg1_value + reg2_value;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if new_value % 2 != 0 {
                    self.odd_flag = true;
                }
                if new_value < 0 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                self.registers_64[register1] = new_value as u64;
            },
            64 => {
                if register1 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register1, RegisterType::Register64));
                }
                if register2 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register2, RegisterType::Register64));
                }
                let reg1_value = self.registers_64[register1] as i64;
                let reg2_value = self.registers_64[register2] as i64;

                let new_value = reg1_value + reg2_value;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }

                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if new_value % 2 != 0 {
                    self.odd_flag = true;
                }
                if new_value > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }

                self.registers_64[register1] = new_value as u64;
            },
            128 => {
                if register1 >= REGISTER_128_COUNT {
                    return Err(Fault::InvalidRegister(register1, RegisterType::Register128));
                }
                if register2 >= REGISTER_128_COUNT {
                    return Err(Fault::InvalidRegister(register2, RegisterType::Register128));
                }
                let reg1_value = self.registers_128[register1] as i128;
                let reg2_value = self.registers_128[register2] as i128;

                let new_value = reg1_value + reg2_value;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }

                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if new_value % 2 != 0 {
                    self.odd_flag = true;
                }
                if new_value > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }

                self.registers_128[register1] = new_value as u128;
            },
            _ => return Err(Fault::InvalidSize),

        }
        Ok(())
    }


    fn subi_opcode(&mut self) -> Result<(), Fault> {
        let size = self.pipeline[self.pipeline_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.pipeline[self.pipeline_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.pipeline[self.pipeline_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                if register1 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register1, RegisterType::Register64));
                }
                if register2 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register2, RegisterType::Register64));
                }
                let reg1_value = self.registers_64[register1] as i8;
                let reg2_value = self.registers_64[register2] as i8;

                let new_value = reg1_value - reg2_value;

                if new_value > reg1_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if new_value % 2 != 0 {
                    self.odd_flag = true;
                }
                if new_value < 0 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                self.registers_64[register1] = new_value as u64;
            },
            16 => {
                if register1 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register1, RegisterType::Register64));
                }
                if register2 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register2, RegisterType::Register64));
                }
                let reg1_value = self.registers_64[register1] as i16;
                let reg2_value = self.registers_64[register2] as i16;

                let new_value = reg1_value - reg2_value;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if new_value % 2 != 0 {
                    self.odd_flag = true;
                }
                if new_value < 0 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                self.registers_64[register1] = new_value as u64;
            },

            32 => {
                if register1 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register1, RegisterType::Register64));
                }
                if register2 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register2, RegisterType::Register64));
                }
                let reg1_value = self.registers_64[register1] as i32;
                let reg2_value = self.registers_64[register2] as i32;

                let new_value = reg1_value - reg2_value;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if new_value % 2 != 0 {
                    self.odd_flag = true;
                }
                if new_value < 0 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                self.registers_64[register1] = new_value as u64;
            },
            64 => {
                if register1 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register1, RegisterType::Register64));
                }
                if register2 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register2, RegisterType::Register64));
                }
                let reg1_value = self.registers_64[register1] as i64;
                let reg2_value = self.registers_64[register2] as i64;

                let new_value = reg1_value - reg2_value;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }

                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if new_value % 2 != 0 {
                    self.odd_flag = true;
                }
                if new_value > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }

                self.registers_64[register1] = new_value as u64;
            },
            128 => {
                if register1 >= REGISTER_128_COUNT {
                    return Err(Fault::InvalidRegister(register1, RegisterType::Register128));
                }
                if register2 >= REGISTER_128_COUNT {
                    return Err(Fault::InvalidRegister(register2, RegisterType::Register128));
                }
                let reg1_value = self.registers_128[register1] as i128;
                let reg2_value = self.registers_128[register2] as i128;

                let new_value = reg1_value - reg2_value;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }

                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if new_value % 2 != 0 {
                    self.odd_flag = true;
                }
                if new_value > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }

                self.registers_128[register1] = new_value as u128;
            },
            _ => return Err(Fault::InvalidSize),

        }
        Ok(())
    }


    fn muli_opcode(&mut self) -> Result<(), Fault> {
        let size = self.pipeline[self.pipeline_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.pipeline[self.pipeline_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.pipeline[self.pipeline_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                if register1 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register1, RegisterType::Register64));
                }
                if register2 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register2, RegisterType::Register64));
                }
                let reg1_value = self.registers_64[register1] as i8;
                let reg2_value = self.registers_64[register2] as i8;

                let new_value = reg1_value * reg2_value;

                if new_value > reg1_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if new_value % 2 != 0 {
                    self.odd_flag = true;
                }
                if new_value < 0 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                self.registers_64[register1] = new_value as u64;
            },
            16 => {
                if register1 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register1, RegisterType::Register64));
                }
                if register2 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register2, RegisterType::Register64));
                }
                let reg1_value = self.registers_64[register1] as i16;
                let reg2_value = self.registers_64[register2] as i16;

                let new_value = reg1_value * reg2_value;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if new_value % 2 != 0 {
                    self.odd_flag = true;
                }
                if new_value < 0 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                self.registers_64[register1] = new_value as u64;
            },

            32 => {
                if register1 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register1, RegisterType::Register64));
                }
                if register2 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register2, RegisterType::Register64));
                }
                let reg1_value = self.registers_64[register1] as i32;
                let reg2_value = self.registers_64[register2] as i32;

                let new_value = reg1_value * reg2_value;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if new_value % 2 != 0 {
                    self.odd_flag = true;
                }
                if new_value < 0 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                self.registers_64[register1] = new_value as u64;
            },
            64 => {
                if register1 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register1, RegisterType::Register64));
                }
                if register2 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register2, RegisterType::Register64));
                }
                let reg1_value = self.registers_64[register1] as i64;
                let reg2_value = self.registers_64[register2] as i64;

                let new_value = reg1_value * reg2_value;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }

                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if new_value % 2 != 0 {
                    self.odd_flag = true;
                }
                if new_value > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }

                self.registers_64[register1] = new_value as u64;
            },
            128 => {
                if register1 >= REGISTER_128_COUNT {
                    return Err(Fault::InvalidRegister(register1, RegisterType::Register128));
                }
                if register2 >= REGISTER_128_COUNT {
                    return Err(Fault::InvalidRegister(register2, RegisterType::Register128));
                }
                let reg1_value = self.registers_128[register1] as i128;
                let reg2_value = self.registers_128[register2] as i128;

                let new_value = reg1_value * reg2_value;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }

                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if new_value % 2 != 0 {
                    self.odd_flag = true;
                }
                if new_value > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }

                self.registers_128[register1] = new_value as u128;
            },
            _ => return Err(Fault::InvalidSize),

        }
        Ok(())
    }


    fn divi_opcode(&mut self) -> Result<(), Fault> {
        let size = self.pipeline[self.pipeline_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.pipeline[self.pipeline_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.pipeline[self.pipeline_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                if register1 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register1, RegisterType::Register64));
                }
                if register2 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register2, RegisterType::Register64));
                }
                let reg1_value = self.registers_64[register1] as i8;
                let reg2_value = self.registers_64[register2] as i8;

                self.remainder_64 = (reg1_value % reg2_value) as usize;
                
                let new_value = reg1_value / reg2_value;

                if new_value > reg1_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if new_value % 2 != 0 {
                    self.odd_flag = true;
                }
                if new_value < 0 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                self.registers_64[register1] = new_value as u64;
            },
            16 => {
                if register1 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register1, RegisterType::Register64));
                }
                if register2 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register2, RegisterType::Register64));
                }
                let reg1_value = self.registers_64[register1] as i16;
                let reg2_value = self.registers_64[register2] as i16;

                self.remainder_64 = (reg1_value % reg2_value) as usize;
                
                let new_value = reg1_value / reg2_value;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if new_value % 2 != 0 {
                    self.odd_flag = true;
                }
                if new_value < 0 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                self.registers_64[register1] = new_value as u64;
            },

            32 => {
                if register1 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register1, RegisterType::Register64));
                }
                if register2 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register2, RegisterType::Register64));
                }
                let reg1_value = self.registers_64[register1] as i32;
                let reg2_value = self.registers_64[register2] as i32;

                self.remainder_64 = (reg1_value % reg2_value) as usize;
                
                let new_value = reg1_value / reg2_value;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if new_value % 2 != 0 {
                    self.odd_flag = true;
                }
                if new_value < 0 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                self.registers_64[register1] = new_value as u64;
            },
            64 => {
                if register1 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register1, RegisterType::Register64));
                }
                if register2 >= REGISTER_64_COUNT {
                    return Err(Fault::InvalidRegister(register2, RegisterType::Register64));
                }
                let reg1_value = self.registers_64[register1] as i64;
                let reg2_value = self.registers_64[register2] as i64;

                self.remainder_64 = (reg1_value % reg2_value) as usize;
                
                let new_value = reg1_value / reg2_value;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }

                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if new_value % 2 != 0 {
                    self.odd_flag = true;
                }
                if new_value > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }

                self.registers_64[register1] = new_value as u64;
            },
            128 => {
                if register1 >= REGISTER_128_COUNT {
                    return Err(Fault::InvalidRegister(register1, RegisterType::Register128));
                }
                if register2 >= REGISTER_128_COUNT {
                    return Err(Fault::InvalidRegister(register2, RegisterType::Register128));
                }
                let reg1_value = self.registers_128[register1] as i128;
                let reg2_value = self.registers_128[register2] as i128;

                self.remainder_128 = (reg1_value % reg2_value) as u128;
                
                let new_value = reg1_value / reg2_value;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }

                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if new_value % 2 != 0 {
                    self.odd_flag = true;
                }
                if new_value > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }

                self.registers_128[register1] = new_value as u128;
            },
            _ => return Err(Fault::InvalidSize),

        }
        Ok(())
    }

}





#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_addi() {
        let program = Arc::new(RwLock::new(vec![6,0,64,0,1]));
        let memory = Arc::new(RwLock::new(Vec::new()));
        let mut core = Core::new(memory, program.clone());

        core.registers_64[0] = 1;
        core.registers_64[1] = 2;

        core.run(0).unwrap();

        assert_eq!(core.registers_64[0] as i64, 3);
    }

    #[test]
    fn test_subi() {
        let program = vec![7,0,64,0,1];
        let memory = Arc::new(RwLock::new(Vec::new()));
        let mut core = Core::new(memory, Arc::new(RwLock::new(program)));

        core.registers_64[0] = 1;
        core.registers_64[1] = 2;

        core.run(0).unwrap();

        assert_eq!(core.registers_64[0] as i64, -1);
        assert_eq!(core.sign_flag, Sign::Negative);
    }

    #[test]
    fn test_muli() {
        let program = vec![8,0,64,0,1];
        let memory = Arc::new(RwLock::new(Vec::new()));
        let mut core = Core::new(memory, Arc::new(RwLock::new(program)));

        core.registers_64[0] = 2;
        core.registers_64[1] = 2;

        core.run(0).unwrap();

        assert_eq!(core.registers_64[0] as i64, 4);
        assert_eq!(core.sign_flag, Sign::Positive);
    }

    #[test]
    fn test_divi() {
        let program = vec![9,0,64,0,1];
        let memory = Arc::new(RwLock::new(Vec::new()));
        let mut core = Core::new(memory, Arc::new(RwLock::new(program)));

        core.registers_64[0] = 4;
        core.registers_64[1] = 3;

        core.run(0).unwrap();

        assert_eq!(core.registers_64[0] as i64, 1);
        assert_eq!(core.remainder_64 as i64, 1);
        assert_eq!(core.sign_flag, Sign::Positive);
    }


}
