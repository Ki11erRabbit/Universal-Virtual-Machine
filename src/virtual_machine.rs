
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
    CorruptedMemory,

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
            return Ok(true);
        }
        use Opcode::*;
        match self.decode_opcode() {
            Halt | NoOp => return Ok(true),
            Set => self.set_opcode()?,
            DeRef => self.deref_opcode()?,
            Move => self.move_opcode()?,
            DeRefReg => self.deref_reg_opcode()?,

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

        }

        Ok(())
        
    }
        
        
}
