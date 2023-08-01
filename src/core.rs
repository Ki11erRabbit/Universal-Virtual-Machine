
use std::sync::{Arc,RwLock,TryLockError,TryLockResult};
use std::num::Wrapping;
use std::array::from_fn;
use std::sync::atomic::AtomicU64;
use std::thread;
use std::io::Write;

use crate::instruction::Opcode;
use crate::virtual_machine::Fault;
use crate::virtual_machine::RegisterType;


macro_rules! check_register64 {
    ($e:expr) => {
        if $e >= REGISTER_64_COUNT {
            return Err(Fault::InvalidRegister($e,RegisterType::Register64));
        }
    };

    ($e:expr, $($es:expr),+) => {
        check_register64!($e);
        check_register64!($($es),+);
    };
}

macro_rules! check_register128 {
    ($e:expr) => {
        if $e >= REGISTER_128_COUNT {
            return Err(Fault::InvalidRegister($e,RegisterType::Register128));
        }
    };

    ($e:expr, $($es:expr),+) => {
        check_register128!($e);
        check_register128!($($es),+);
    };
}

macro_rules! check_registerF32 {
    ($e:expr) => {
        if $e >= REGISTER_F32_COUNT {
            return Err(Fault::InvalidRegister($e,RegisterType::RegisterF32));
        }
    };

    ($e:expr, $($es:expr),+) => {
        check_registerF32!($e);
        check_registerF32!($($es),+);
    };
}

macro_rules! check_registerF64 {
    ($e:expr) => {
        if $e >= REGISTER_F64_COUNT {
            return Err(Fault::InvalidRegister($e,RegisterType::RegisterF64));
        }
    };

    ($e:expr, $($es:expr),+) => {
        check_registerF64!($e);
        check_registerF64!($($es),+);
    };
}

macro_rules! check_registerAtomic64 {
    ($e:expr) => {
        if $e >= REGISTER_ATOMIC_64_COUNT {
            return Err(Fault::InvalidRegister($e,RegisterType::RegisterAtomic64));
        }
    };

    ($e:expr, $($es:expr),+) => {
        check_registerAtomic64!($e);
        check_registerAtomic64!($(es),+);
    };
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


#[derive(Debug,PartialEq)]
pub enum Comparison {
    Equal,
    NotEqual,
    LessThan,
    NotLessThan,
    GreaterThan,
    NotGreaterThan,
    LessThanOrEqual,
    GreaterThanOrEqual,
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
    comparison_flag: Comparison,
    odd_flag: bool,
    zero_flag: bool,
    sign_flag: Sign,
    overflow_flag: bool,
    infinity_flag: bool,
    nan_flag: bool,
    /* other */
    remainder_64: usize,
    remainder_128: u128,
    program_counter: usize,
    stack: Vec<u8>,
    program: Arc<Vec<u8>>,
    memory: Arc<RwLock<Vec<u8>>>,
}
    
impl Core {
    pub fn new(memory: Arc<RwLock<Vec<u8>>>, program: Arc<Vec<u8>>) -> Core {
        Core {
            registers_64: [0; 16],
            registers_128: [0; 8],
            registers_f32: [0.0; 8],
            registers_f64: [0.0; 8],
            registers_atomic_64: from_fn(|_| AtomicU64::new(0)),
            remainder_64: 0,
            remainder_128: 0,
            comparison_flag: Comparison::Equal,
            odd_flag: false,
            zero_flag: false,
            sign_flag: Sign::Positive,
            overflow_flag: false,
            infinity_flag: false,
            nan_flag: false,
            program_counter: 0,
            stack: Vec::new(),
            program,
            memory,
        }
    }

    #[inline]
    fn advance_by_size(&mut self, size: usize) {
        self.program_counter += size;
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

        if self.program_counter >= self.program.len() {
            return Ok(true);
        }
        return Ok(false);
    }


    fn decode_opcode(&mut self) -> Opcode {
        let opcode = Opcode::from(self.program[self.program_counter] as u16);
        self.advance_by_2_bytes();
        return opcode;

    }

    fn execute_instruction(&mut self) -> Result<bool, Fault> {
        if self.check_program_counter()? {
            return Ok(true);
        }
        use Opcode::*;

        let opcode = self.decode_opcode();

        
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
            EqI => self.eqi_opcode()?,
            NeqI => self.neqi_opcode()?,
            LtI => self.lti_opcode()?,
            GtI => self.gti_opcode()?,
            LeqI => self.leqi_opcode()?,
            GeqI => self.geqi_opcode()?,
            AddU => self.addu_opcode()?,
            SubU => self.subu_opcode()?,
            MulU => self.mulu_opcode()?,
            DivU => self.divu_opcode()?,
            EqU => self.equ_opcode()?,
            NeqU => self.nequ_opcode()?,
            LtU => self.ltu_opcode()?,
            GtU => self.gtu_opcode()?,
            LeqU => self.lequ_opcode()?,
            GeqU => self.gequ_opcode()?,
            WriteByte => self.writebyte_opcode()?,
            Write => self.write_opcode()?,
            Flush => self.flush_opcode()?,
            And => self.and_opcode()?,
            Or => self.or_opcode()?,
            Xor => self.xor_opcode()?,
            Not => self.not_opcode()?,
            ShiftLeft => self.shiftleft_opcode()?,
            ShiftRight => self.shiftright_opcode()?,
            Clear => self.clear_opcode()?,
            Remainder => self.remainder_opcode()?,
            AddFI => self.addfi_opcode()?,
            SubFI => self.subfi_opcode()?,
            MulFI => self.mulfi_opcode()?,
            DivFI => self.divfi_opcode()?,
            AddIF => self.addif_opcode()?,
            SubIF => self.subif_opcode()?,
            MulIF => self.mulif_opcode()?,
            DivIF => self.divif_opcode()?,
            AddUF => self.adduf_opcode()?,
            SubUF => self.subuf_opcode()?,
            MulUF => self.muluf_opcode()?,
            DivUF => self.divuf_opcode()?,
            DeRefRegF => self.derefregf_opcode()?,
            DeRefF => self.dereff_opcode()?,
            MoveF => self.movef_opcode()?,
            SetF => self.setf_opcode()?,
            
            

            _ => return Err(Fault::InvalidOperation),
            

        }

        Ok(false)
            
    }


    fn set_opcode(&mut self) -> Result<(), Fault> {
        let size = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        let register = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        match size {
            8 => {
                let value = self.program[self.program_counter] as u8;
                check_register64!(register);
                self.registers_64[register] = value as u64;
                self.advance_by_1_byte();
            },
            16 => {
                let value = self.program[self.program_counter] as u16;
                check_register64!(register);
                self.registers_64[register] = value as u64;
                self.advance_by_2_bytes();
            },
            32 => {
                let value = self.program[self.program_counter] as u32;
                check_register64!(register);
                self.registers_64[register] = value as u64;
                self.advance_by_4_bytes();
            },
            64 => {
                let value = self.program[self.program_counter] as u64;
                check_register64!(register);
                self.registers_64[register] = value as u64;
                self.advance_by_8_bytes();
            },
            128 => {
                let value = self.program[self.program_counter] as u128;
                check_register128!(register);
                self.registers_128[register] = value as u128;
                self.advance_by_16_bytes();
            },
            _ => return Err(Fault::InvalidSize),
        }
        Ok(())
    }

    fn deref_opcode(&mut self) -> Result<(), Fault> {
        let size = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        let register = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        let address = self.registers_64[register];
        match size {
            8 => {
                //TODO: Check to see if we can even read from memory so we don't panic
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
        let size = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        let register = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        match size {
            8 => {
                check_register64!(register);
                let address = self.program[self.program_counter] as u64;
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
                check_register64!(register);
                let address = self.program[self.program_counter] as u64;
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
                check_register64!(register);
                let address = self.program[self.program_counter] as u64;
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
                check_register64!(register);
                let address = self.program[self.program_counter] as u64;
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
                check_register128!(register);
                let address = self.program[self.program_counter] as u64;
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
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        let address_register = self.program[self.program_counter] as usize;
        check_register64!(address_register);
        self.advance_by_1_byte();
        let offset = self.program[self.program_counter] as i64;
        self.advance_by_8_bytes();
        let address = self.registers_64[address_register] as i64 + offset;
        let address = address as u64;

        match size {
            8 => {
                check_register64!(register);

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
                check_register64!(register);

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
                check_register64!(register);

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
                check_register64!(register);

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
                check_register128!(register);

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
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as i8;
                let reg2_value = self.registers_64[register2] as i8;

                let new_value = (Wrapping(reg1_value) + Wrapping(reg2_value)).0;

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
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as i16;
                let reg2_value = self.registers_64[register2] as i16;

                let new_value = (Wrapping(reg1_value) + Wrapping(reg2_value)).0;

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
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as i32;
                let reg2_value = self.registers_64[register2] as i32;

                let new_value = (Wrapping(reg1_value) + Wrapping(reg2_value)).0;

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
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as i64;
                let reg2_value = self.registers_64[register2] as i64;

                let new_value = (Wrapping(reg1_value) + Wrapping(reg2_value)).0;

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
                check_register128!(register1, register2);
                let reg1_value = self.registers_128[register1] as i128;
                let reg2_value = self.registers_128[register2] as i128;

                let new_value = (Wrapping(reg1_value) + Wrapping(reg2_value)).0;

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
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as i8;
                let reg2_value = self.registers_64[register2] as i8;

                let new_value = (Wrapping(reg1_value) - Wrapping(reg2_value)).0;

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
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as i16;
                let reg2_value = self.registers_64[register2] as i16;

                let new_value = (Wrapping(reg1_value) - Wrapping(reg2_value)).0;

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
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as i32;
                let reg2_value = self.registers_64[register2] as i32;

                let new_value = (Wrapping(reg1_value) - Wrapping(reg2_value)).0;

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
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as i64;
                let reg2_value = self.registers_64[register2] as i64;

                let new_value = (Wrapping(reg1_value) - Wrapping(reg2_value)).0;

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
                check_register128!(register1, register2);
                let reg1_value = self.registers_128[register1] as i128;
                let reg2_value = self.registers_128[register2] as i128;

                let new_value = (Wrapping(reg1_value) - Wrapping(reg2_value)).0;

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
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as i8;
                let reg2_value = self.registers_64[register2] as i8;

                let new_value = (Wrapping(reg1_value) * Wrapping(reg2_value)).0;

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
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as i16;
                let reg2_value = self.registers_64[register2] as i16;

                let new_value = (Wrapping(reg1_value) * Wrapping(reg2_value)).0;

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
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as i32;
                let reg2_value = self.registers_64[register2] as i32;

                let new_value = (Wrapping(reg1_value) * Wrapping(reg2_value)).0;

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
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as i64;
                let reg2_value = self.registers_64[register2] as i64;

                let new_value = (Wrapping(reg1_value) * Wrapping(reg2_value)).0;

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
                check_register128!(register1, register2);
                let reg1_value = self.registers_128[register1] as i128;
                let reg2_value = self.registers_128[register2] as i128;

                let new_value = (Wrapping(reg1_value) * Wrapping(reg2_value)).0;

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
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as i8;
                let reg2_value = self.registers_64[register2] as i8;

                if reg2_value == 0 {
                    return Err(Fault::DivideByZero);
                }

                self.remainder_64 = (Wrapping(reg1_value) % Wrapping(reg2_value)).0 as usize;
                
                let new_value = (Wrapping(reg1_value) / Wrapping(reg2_value)).0;

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
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as i16;
                let reg2_value = self.registers_64[register2] as i16;

                if reg2_value == 0 {
                    return Err(Fault::DivideByZero);
                }

                self.remainder_64 = (Wrapping(reg1_value) % Wrapping(reg2_value)).0 as usize;
                
                let new_value = (Wrapping(reg1_value) / Wrapping(reg2_value)).0;

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
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as i32;
                let reg2_value = self.registers_64[register2] as i32;

                if reg2_value == 0 {
                    return Err(Fault::DivideByZero);
                }

                self.remainder_64 = (Wrapping(reg1_value) % Wrapping(reg2_value)).0 as usize;
                
                let new_value = (Wrapping(reg1_value) / Wrapping(reg2_value)).0;

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
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as i64;
                let reg2_value = self.registers_64[register2] as i64;

                if reg2_value == 0 {
                    return Err(Fault::DivideByZero);
                }

                self.remainder_64 = (Wrapping(reg1_value) % Wrapping(reg2_value)).0 as usize;
                
                let new_value = (Wrapping(reg1_value) / Wrapping(reg2_value)).0;

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
                check_register128!(register1, register2);
                let reg1_value = self.registers_128[register1] as i128;
                let reg2_value = self.registers_128[register2] as i128;

                if reg2_value == 0 {
                    return Err(Fault::DivideByZero);
                }

                self.remainder_128 = (Wrapping(reg1_value) % Wrapping(reg2_value)).0 as u128;
                
                let new_value = (Wrapping(reg1_value) / Wrapping(reg2_value)).0;

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

    fn neqi_opcode(&mut self) -> Result<(),Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as i8;
                let reg2_value = self.registers_64[register2] as i8;

                if reg1_value != reg2_value {
                    self.comparison_flag = Comparison::NotEqual;
                }
                else {
                    self.comparison_flag = Comparison::Equal;
                }

                if reg1_value == 0 && reg2_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
            },
            16 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as i16;
                let reg2_value = self.registers_64[register2] as i16;

                if reg1_value != reg2_value {
                    self.comparison_flag = Comparison::NotEqual;
                }
                else {
                    self.comparison_flag = Comparison::Equal;
                }

                if reg1_value == 0 && reg2_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
            },
            32 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as i32;
                let reg2_value = self.registers_64[register2] as i32;

                if reg1_value != reg2_value {
                    self.comparison_flag = Comparison::NotEqual;
                }
                else {
                    self.comparison_flag = Comparison::Equal;
                }

                if reg1_value == 0 && reg2_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
            },
            64 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as i64;
                let reg2_value = self.registers_64[register2] as i64;

                if reg1_value != reg2_value {
                    self.comparison_flag = Comparison::NotEqual;
                }
                else {
                    self.comparison_flag = Comparison::Equal;
                }

                if reg1_value == 0 && reg2_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
            },
            128 => {
                check_register128!(register1, register2);

                let reg1_value = self.registers_128[register1] as i128;
                let reg2_value = self.registers_128[register2] as i128;

                if reg1_value != reg2_value {
                    self.comparison_flag = Comparison::NotEqual;
                }
                else {
                    self.comparison_flag = Comparison::Equal;
                }

                if reg1_value == 0 && reg2_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
            },

            _ => return Err(Fault::InvalidSize),
            
        }
        
        Ok(())
    }
    
    fn eqi_opcode(&mut self) -> Result<(),Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as i8;
                let reg2_value = self.registers_64[register2] as i8;

                if reg1_value == reg2_value {
                    self.comparison_flag = Comparison::Equal;
                }
                else {
                    self.comparison_flag = Comparison::NotEqual;
                }

                if reg1_value == 0 && reg2_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
            },
            16 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as i16;
                let reg2_value = self.registers_64[register2] as i16;

                if reg1_value == reg2_value {
                    self.comparison_flag = Comparison::Equal;
                }
                else {
                    self.comparison_flag = Comparison::NotEqual;
                }

                if reg1_value == 0 && reg2_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
            },
            32 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as i32;
                let reg2_value = self.registers_64[register2] as i32;

                if reg1_value == reg2_value {
                    self.comparison_flag = Comparison::Equal;
                }
                else {
                    self.comparison_flag = Comparison::NotEqual;
                }

                if reg1_value == 0 && reg2_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
            },
            64 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as i64;
                let reg2_value = self.registers_64[register2] as i64;

                if reg1_value == reg2_value {
                    self.comparison_flag = Comparison::Equal;
                }
                else {
                    self.comparison_flag = Comparison::NotEqual;
                }

                if reg1_value == 0 && reg2_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
            },
            128 => {
                check_register128!(register1, register2);

                let reg1_value = self.registers_128[register1] as i128;
                let reg2_value = self.registers_128[register2] as i128;

                if reg1_value == reg2_value {
                    self.comparison_flag = Comparison::Equal;
                }
                else {
                    self.comparison_flag = Comparison::NotEqual;
                }

                if reg1_value == 0 && reg2_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
            },

            _ => return Err(Fault::InvalidSize),
            
        }
        
        Ok(())
    }

    fn lti_opcode(&mut self) -> Result<(),Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as i8;
                let reg2_value = self.registers_64[register2] as i8;

                if reg1_value < reg2_value {
                    self.comparison_flag = Comparison::LessThan;
                }
                else {
                    self.comparison_flag = Comparison::NotLessThan;
                }
            },
            16 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as i16;
                let reg2_value = self.registers_64[register2] as i16;

                if reg1_value < reg2_value {
                    self.comparison_flag = Comparison::LessThan;
                }
                else {
                    self.comparison_flag = Comparison::NotLessThan;
                }
            },
            32 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as i32;
                let reg2_value = self.registers_64[register2] as i32;

                if reg1_value < reg2_value {
                    self.comparison_flag = Comparison::LessThan;
                }
                else {
                    self.comparison_flag = Comparison::NotLessThan;
                }
            },
            64 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as i64;
                let reg2_value = self.registers_64[register2] as i64;

                if reg1_value < reg2_value {
                    self.comparison_flag = Comparison::LessThan;
                }
                else {
                    self.comparison_flag = Comparison::NotLessThan;
                }
            },
            128 => {
                check_register128!(register1, register2);

                let reg1_value = self.registers_128[register1] as i128;
                let reg2_value = self.registers_128[register2] as i128;

                if reg1_value < reg2_value {
                    self.comparison_flag = Comparison::LessThan;
                }
                else {
                    self.comparison_flag = Comparison::NotLessThan;
                }
            },
            _ => return Err(Fault::InvalidSize),

        }

        Ok(())
    }

    fn gti_opcode(&mut self) -> Result<(),Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as i8;
                let reg2_value = self.registers_64[register2] as i8;

                if reg1_value > reg2_value {
                    self.comparison_flag = Comparison::GreaterThan;
                }
                else {
                    self.comparison_flag = Comparison::NotGreaterThan;
                }
            },
            16 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as i16;
                let reg2_value = self.registers_64[register2] as i16;

                if reg1_value > reg2_value {
                    self.comparison_flag = Comparison::GreaterThan;
                }
                else {
                    self.comparison_flag = Comparison::NotGreaterThan;
                }
            },
            32 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as i32;
                let reg2_value = self.registers_64[register2] as i32;

                if reg1_value > reg2_value {
                    self.comparison_flag = Comparison::GreaterThan;
                }
                else {
                    self.comparison_flag = Comparison::NotGreaterThan;
                }
            },
            64 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as i64;
                let reg2_value = self.registers_64[register2] as i64;

                if reg1_value > reg2_value {
                    self.comparison_flag = Comparison::GreaterThan;
                }
                else {
                    self.comparison_flag = Comparison::NotGreaterThan;
                }
            },
            128 => {
                check_register128!(register1, register2);

                let reg1_value = self.registers_128[register1] as i128;
                let reg2_value = self.registers_128[register2] as i128;

                if reg1_value > reg2_value {
                    self.comparison_flag = Comparison::GreaterThan;
                }
                else {
                    self.comparison_flag = Comparison::NotGreaterThan;
                }
            },
            _ => return Err(Fault::InvalidSize),

        }

        Ok(())
    }
        
    fn leqi_opcode(&mut self) -> Result<(),Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as i8;
                let reg2_value = self.registers_64[register2] as i8;

                if reg1_value <= reg2_value {
                    self.comparison_flag = Comparison::LessThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::GreaterThan;
                }
            },
            16 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as i16;
                let reg2_value = self.registers_64[register2] as i16;

                if reg1_value <= reg2_value {
                    self.comparison_flag = Comparison::LessThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::GreaterThan;
                }
            },
            32 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as i32;
                let reg2_value = self.registers_64[register2] as i32;

                if reg1_value <= reg2_value {
                    self.comparison_flag = Comparison::LessThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::GreaterThan;
                }
            },
            64 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as i64;
                let reg2_value = self.registers_64[register2] as i64;

                if reg1_value <= reg2_value {
                    self.comparison_flag = Comparison::LessThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::GreaterThan;
                }
            },
            128 => {
                check_register128!(register1, register2);

                let reg1_value = self.registers_128[register1] as i128;
                let reg2_value = self.registers_128[register2] as i128;

                if reg1_value <= reg2_value {
                    self.comparison_flag = Comparison::LessThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::GreaterThan;
                }
            },
            _ => return Err(Fault::InvalidSize),

        }

        Ok(())
    }
        
    fn geqi_opcode(&mut self) -> Result<(),Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as i8;
                let reg2_value = self.registers_64[register2] as i8;

                if reg1_value >= reg2_value {
                    self.comparison_flag = Comparison::GreaterThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::LessThan;
                }
            },
            16 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as i16;
                let reg2_value = self.registers_64[register2] as i16;

                if reg1_value >= reg2_value {
                    self.comparison_flag = Comparison::GreaterThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::LessThan;
                }
            },
            32 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as i32;
                let reg2_value = self.registers_64[register2] as i32;

                if reg1_value >= reg2_value {
                    self.comparison_flag = Comparison::GreaterThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::LessThan;
                }
            },
            64 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as i64;
                let reg2_value = self.registers_64[register2] as i64;

                if reg1_value >= reg2_value {
                    self.comparison_flag = Comparison::GreaterThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::LessThan;
                }
            },
            128 => {
                check_register128!(register1, register2);

                let reg1_value = self.registers_128[register1] as i128;
                let reg2_value = self.registers_128[register2] as i128;

                if reg1_value >= reg2_value {
                    self.comparison_flag = Comparison::GreaterThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::LessThan;
                }
            },
            _ => return Err(Fault::InvalidSize),

        }

        Ok(())
    }


    fn addu_opcode(&mut self) -> Result<(), Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as u8;
                let reg2_value = self.registers_64[register2] as u8;

                let new_value = (Wrapping(reg1_value) + Wrapping(reg2_value)).0;

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
                self.sign_flag = Sign::Positive;

                self.registers_64[register1] = new_value as u64;
            },
            16 => {
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as u16;
                let reg2_value = self.registers_64[register2] as u16;

                let new_value = (Wrapping(reg1_value) + Wrapping(reg2_value)).0;

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
                self.sign_flag = Sign::Positive;

                self.registers_64[register1] = new_value as u64;
            },

            32 => {
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as u32;
                let reg2_value = self.registers_64[register2] as u32;

                let new_value = (Wrapping(reg1_value) + Wrapping(reg2_value)).0;

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
                self.sign_flag = Sign::Positive;

                self.registers_64[register1] = new_value as u64;
            },
            64 => {
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as u64;
                let reg2_value = self.registers_64[register2] as u64;

                let new_value = (Wrapping(reg1_value) + Wrapping(reg2_value)).0;

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
                self.sign_flag = Sign::Positive;

                self.registers_64[register1] = new_value as u64;
            },
            128 => {
                check_register128!(register1, register2);
                let reg1_value = self.registers_128[register1] as u128;
                let reg2_value = self.registers_128[register2] as u128;

                let new_value = (Wrapping(reg1_value) + Wrapping(reg2_value)).0;

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
                
                self.sign_flag = Sign::Positive;


                self.registers_128[register1] = new_value as u128;
            },
            _ => return Err(Fault::InvalidSize),

        }
        Ok(())
    }


    fn subu_opcode(&mut self) -> Result<(), Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as u8;
                let reg2_value = self.registers_64[register2] as u8;

                let new_value = (Wrapping(reg1_value) - Wrapping(reg2_value)).0;

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
                self.sign_flag = Sign::Positive;

                self.registers_64[register1] = new_value as u64;
            },
            16 => {
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as u16;
                let reg2_value = self.registers_64[register2] as u16;

                let new_value = (Wrapping(reg1_value) - Wrapping(reg2_value)).0;

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
                self.sign_flag = Sign::Positive;

                self.registers_64[register1] = new_value as u64;
            },
            32 => {
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as u32;
                let reg2_value = self.registers_64[register2] as u32;

                let new_value = (Wrapping(reg1_value) - Wrapping(reg2_value)).0;

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
                self.sign_flag = Sign::Positive;

                self.registers_64[register1] = new_value as u64;
            },
            64 => {
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as u64;
                let reg2_value = self.registers_64[register2] as u64;

                let new_value = (Wrapping(reg1_value) - Wrapping(reg2_value)).0;

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
                self.sign_flag = Sign::Positive;

                self.registers_64[register1] = new_value as u64;
            },
            128 => {
                check_register128!(register1, register2);
                let reg1_value = self.registers_128[register1] as u128;
                let reg2_value = self.registers_128[register2] as u128;

                let new_value = (Wrapping(reg1_value) - Wrapping(reg2_value)).0;

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
                self.sign_flag = Sign::Positive;

                self.registers_128[register1] = new_value as u128;
            },
            _ => return Err(Fault::InvalidSize),

        }
        Ok(())
    }


    fn mulu_opcode(&mut self) -> Result<(), Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as u8;
                let reg2_value = self.registers_64[register2] as u8;

                let new_value = (Wrapping(reg1_value) * Wrapping(reg2_value)).0;

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
                self.sign_flag = Sign::Positive;

                self.registers_64[register1] = new_value as u64;
            },
            16 => {
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as u16;
                let reg2_value = self.registers_64[register2] as u16;

                let new_value = (Wrapping(reg1_value) * Wrapping(reg2_value)).0;

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
                
                self.sign_flag = Sign::Positive;


                self.registers_64[register1] = new_value as u64;
            },

            32 => {
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as u32;
                let reg2_value = self.registers_64[register2] as u32;

                let new_value = (Wrapping(reg1_value) * Wrapping(reg2_value)).0;

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
                self.sign_flag = Sign::Positive;

                self.registers_64[register1] = new_value as u64;
            },
            64 => {
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as u64;
                let reg2_value = self.registers_64[register2] as u64;

                let new_value = (Wrapping(reg1_value) * Wrapping(reg2_value)).0;

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
                self.sign_flag = Sign::Positive;

                self.registers_64[register1] = new_value as u64;
            },
            128 => {
                check_register128!(register1, register2);
                let reg1_value = self.registers_128[register1] as u128;
                let reg2_value = self.registers_128[register2] as u128;

                let new_value = (Wrapping(reg1_value) * Wrapping(reg2_value)).0;

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
                self.sign_flag = Sign::Positive;

                self.registers_128[register1] = new_value as u128;
            },
            _ => return Err(Fault::InvalidSize),

        }
        Ok(())
    }


    fn divu_opcode(&mut self) -> Result<(), Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as u8;
                let reg2_value = self.registers_64[register2] as u8;

                if reg2_value == 0 {
                    return Err(Fault::DivideByZero);
                }

                self.remainder_64 = (Wrapping(reg1_value) % Wrapping(reg2_value)).0 as usize;
                
                let new_value = (Wrapping(reg1_value) / Wrapping(reg2_value)).0;

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
                self.sign_flag = Sign::Positive;

                self.registers_64[register1] = new_value as u64;
            },
            16 => {
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as u16;
                let reg2_value = self.registers_64[register2] as u16;

                if reg2_value == 0 {
                    return Err(Fault::DivideByZero);
                }

                self.remainder_64 = (Wrapping(reg1_value) % Wrapping(reg2_value)).0 as usize;
                
                let new_value = (Wrapping(reg1_value) / Wrapping(reg2_value)).0;

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
                self.sign_flag = Sign::Positive;

                self.registers_64[register1] = new_value as u64;
            },
            32 => {
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as u32;
                let reg2_value = self.registers_64[register2] as u32;

                if reg2_value == 0 {
                    return Err(Fault::DivideByZero);
                }

                self.remainder_64 = (Wrapping(reg1_value) % Wrapping(reg2_value)).0 as usize;
                
                let new_value = (Wrapping(reg1_value) / Wrapping(reg2_value)).0;

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
                self.sign_flag = Sign::Positive;

                self.registers_64[register1] = new_value as u64;
            },
            64 => {
                check_register64!(register1, register2);
                let reg1_value = self.registers_64[register1] as u64;
                let reg2_value = self.registers_64[register2] as u64;

                if reg2_value == 0 {
                    return Err(Fault::DivideByZero);
                }

                self.remainder_64 = (Wrapping(reg1_value) % Wrapping(reg2_value)).0 as usize;
                
                let new_value = (Wrapping(reg1_value) / Wrapping(reg2_value)).0;

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
                self.sign_flag = Sign::Positive;

                self.registers_64[register1] = new_value as u64;
            },
            128 => {
                check_register128!(register1, register2);
                let reg1_value = self.registers_128[register1] as u128;
                let reg2_value = self.registers_128[register2] as u128;

                if reg2_value == 0 {
                    return Err(Fault::DivideByZero);
                }

                self.remainder_128 = (Wrapping(reg1_value) % Wrapping(reg2_value)).0 as u128;
                
                let new_value = (Wrapping(reg1_value) / Wrapping(reg2_value)).0;

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
                self.sign_flag = Sign::Positive;

                self.registers_128[register1] = new_value as u128;
            },
            _ => return Err(Fault::InvalidSize),

        }
        Ok(())
    }

    fn nequ_opcode(&mut self) -> Result<(),Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as u8;
                let reg2_value = self.registers_64[register2] as u8;

                if reg1_value != reg2_value {
                    self.comparison_flag = Comparison::NotEqual;
                }
                else {
                    self.comparison_flag = Comparison::Equal;
                }

                if reg1_value == 0 && reg2_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
            },
            16 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as u16;
                let reg2_value = self.registers_64[register2] as u16;

                if reg1_value != reg2_value {
                    self.comparison_flag = Comparison::NotEqual;
                }
                else {
                    self.comparison_flag = Comparison::Equal;
                }

                if reg1_value == 0 && reg2_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
            },
            32 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as u32;
                let reg2_value = self.registers_64[register2] as u32;

                if reg1_value != reg2_value {
                    self.comparison_flag = Comparison::NotEqual;
                }
                else {
                    self.comparison_flag = Comparison::Equal;
                }

                if reg1_value == 0 && reg2_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
            },
            64 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as u64;
                let reg2_value = self.registers_64[register2] as u64;

                if reg1_value != reg2_value {
                    self.comparison_flag = Comparison::NotEqual;
                }
                else {
                    self.comparison_flag = Comparison::Equal;
                }

                if reg1_value == 0 && reg2_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
            },
            128 => {
                check_register128!(register1, register2);

                let reg1_value = self.registers_128[register1] as u128;
                let reg2_value = self.registers_128[register2] as u128;

                if reg1_value != reg2_value {
                    self.comparison_flag = Comparison::NotEqual;
                }
                else {
                    self.comparison_flag = Comparison::Equal;
                }

                if reg1_value == 0 && reg2_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
            },

            _ => return Err(Fault::InvalidSize),
            
        }
        
        Ok(())
    }
    
    fn equ_opcode(&mut self) -> Result<(),Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as u8;
                let reg2_value = self.registers_64[register2] as u8;

                if reg1_value == reg2_value {
                    self.comparison_flag = Comparison::Equal;
                }
                else {
                    self.comparison_flag = Comparison::NotEqual;
                }

                if reg1_value == 0 && reg2_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
            },
            16 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as u16;
                let reg2_value = self.registers_64[register2] as u16;

                if reg1_value == reg2_value {
                    self.comparison_flag = Comparison::Equal;
                }
                else {
                    self.comparison_flag = Comparison::NotEqual;
                }

                if reg1_value == 0 && reg2_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
            },
            32 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as u32;
                let reg2_value = self.registers_64[register2] as u32;

                if reg1_value == reg2_value {
                    self.comparison_flag = Comparison::Equal;
                }
                else {
                    self.comparison_flag = Comparison::NotEqual;
                }

                if reg1_value == 0 && reg2_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
            },
            64 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as u64;
                let reg2_value = self.registers_64[register2] as u64;

                if reg1_value == reg2_value {
                    self.comparison_flag = Comparison::Equal;
                }
                else {
                    self.comparison_flag = Comparison::NotEqual;
                }

                if reg1_value == 0 && reg2_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
            },
            128 => {
                check_register128!(register1, register2);

                let reg1_value = self.registers_128[register1] as u128;
                let reg2_value = self.registers_128[register2] as u128;

                if reg1_value == reg2_value {
                    self.comparison_flag = Comparison::Equal;
                }
                else {
                    self.comparison_flag = Comparison::NotEqual;
                }

                if reg1_value == 0 && reg2_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
            },

            _ => return Err(Fault::InvalidSize),
            
        }
        
        Ok(())
    }

    fn ltu_opcode(&mut self) -> Result<(),Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as u8;
                let reg2_value = self.registers_64[register2] as u8;

                if reg1_value < reg2_value {
                    self.comparison_flag = Comparison::LessThan;
                }
                else {
                    self.comparison_flag = Comparison::NotLessThan;
                }
            },
            16 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as u16;
                let reg2_value = self.registers_64[register2] as u16;

                if reg1_value < reg2_value {
                    self.comparison_flag = Comparison::LessThan;
                }
                else {
                    self.comparison_flag = Comparison::NotLessThan;
                }
            },
            32 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as u32;
                let reg2_value = self.registers_64[register2] as u32;

                if reg1_value < reg2_value {
                    self.comparison_flag = Comparison::LessThan;
                }
                else {
                    self.comparison_flag = Comparison::NotLessThan;
                }
            },
            64 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as u64;
                let reg2_value = self.registers_64[register2] as u64;

                if reg1_value < reg2_value {
                    self.comparison_flag = Comparison::LessThan;
                }
                else {
                    self.comparison_flag = Comparison::NotLessThan;
                }
            },
            128 => {
                check_register128!(register1, register2);

                let reg1_value = self.registers_128[register1] as u128;
                let reg2_value = self.registers_128[register2] as u128;

                if reg1_value < reg2_value {
                    self.comparison_flag = Comparison::LessThan;
                }
                else {
                    self.comparison_flag = Comparison::NotLessThan;
                }
            },
            _ => return Err(Fault::InvalidSize),

        }

        Ok(())
    }

    fn gtu_opcode(&mut self) -> Result<(),Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as u8;
                let reg2_value = self.registers_64[register2] as u8;

                if reg1_value > reg2_value {
                    self.comparison_flag = Comparison::GreaterThan;
                }
                else {
                    self.comparison_flag = Comparison::NotGreaterThan;
                }
            },
            16 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as u16;
                let reg2_value = self.registers_64[register2] as u16;

                if reg1_value > reg2_value {
                    self.comparison_flag = Comparison::GreaterThan;
                }
                else {
                    self.comparison_flag = Comparison::NotGreaterThan;
                }
            },
            32 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as u32;
                let reg2_value = self.registers_64[register2] as u32;

                if reg1_value > reg2_value {
                    self.comparison_flag = Comparison::GreaterThan;
                }
                else {
                    self.comparison_flag = Comparison::NotGreaterThan;
                }
            },
            64 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as u64;
                let reg2_value = self.registers_64[register2] as u64;

                if reg1_value > reg2_value {
                    self.comparison_flag = Comparison::GreaterThan;
                }
                else {
                    self.comparison_flag = Comparison::NotGreaterThan;
                }
            },
            128 => {
                check_register128!(register1, register2);

                let reg1_value = self.registers_128[register1] as u128;
                let reg2_value = self.registers_128[register2] as u128;

                if reg1_value > reg2_value {
                    self.comparison_flag = Comparison::GreaterThan;
                }
                else {
                    self.comparison_flag = Comparison::NotGreaterThan;
                }
            },
            _ => return Err(Fault::InvalidSize),

        }

        Ok(())
    }
        
    fn lequ_opcode(&mut self) -> Result<(),Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as u8;
                let reg2_value = self.registers_64[register2] as u8;

                if reg1_value <= reg2_value {
                    self.comparison_flag = Comparison::LessThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::GreaterThan;
                }
            },
            16 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as u16;
                let reg2_value = self.registers_64[register2] as u16;

                if reg1_value <= reg2_value {
                    self.comparison_flag = Comparison::LessThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::GreaterThan;
                }
            },
            32 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as u32;
                let reg2_value = self.registers_64[register2] as u32;

                if reg1_value <= reg2_value {
                    self.comparison_flag = Comparison::LessThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::GreaterThan;
                }
            },
            64 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as u64;
                let reg2_value = self.registers_64[register2] as u64;

                if reg1_value <= reg2_value {
                    self.comparison_flag = Comparison::LessThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::GreaterThan;
                }
            },
            128 => {
                check_register128!(register1, register2);

                let reg1_value = self.registers_128[register1] as u128;
                let reg2_value = self.registers_128[register2] as u128;

                if reg1_value <= reg2_value {
                    self.comparison_flag = Comparison::LessThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::GreaterThan;
                }
            },
            _ => return Err(Fault::InvalidSize),

        }

        Ok(())
    }
        
    fn gequ_opcode(&mut self) -> Result<(),Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as u8;
                let reg2_value = self.registers_64[register2] as u8;

                if reg1_value >= reg2_value {
                    self.comparison_flag = Comparison::GreaterThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::LessThan;
                }
            },
            16 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as u16;
                let reg2_value = self.registers_64[register2] as u16;

                if reg1_value >= reg2_value {
                    self.comparison_flag = Comparison::GreaterThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::LessThan;
                }
            },
            32 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as u32;
                let reg2_value = self.registers_64[register2] as u32;

                if reg1_value >= reg2_value {
                    self.comparison_flag = Comparison::GreaterThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::LessThan;
                }
            },
            64 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as u64;
                let reg2_value = self.registers_64[register2] as u64;

                if reg1_value >= reg2_value {
                    self.comparison_flag = Comparison::GreaterThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::LessThan;
                }
            },
            128 => {
                check_register128!(register1, register2);

                let reg1_value = self.registers_128[register1] as u128;
                let reg2_value = self.registers_128[register2] as u128;

                if reg1_value >= reg2_value {
                    self.comparison_flag = Comparison::GreaterThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::LessThan;
                }
            },
            _ => return Err(Fault::InvalidSize),

        }

        Ok(())
    }

    fn and_opcode(&mut self) -> Result<(), Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as u8;
                let reg2_value = self.registers_64[register2] as u8;

                self.registers_64[register1] = (reg1_value & reg2_value) as u64;

                if self.registers_64[register1] as u8 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }

                if self.registers_64[register1] as u8 & 0x80 == 0x80 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                if self.registers_64[register1] as u8 % 2 == 0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
                
            },
            16 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as u16;
                let reg2_value = self.registers_64[register2] as u16;

                self.registers_64[register1] = (reg1_value & reg2_value) as u64;

                if self.registers_64[register1] as u16 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }

                if self.registers_64[register1] as u16 & 0x80 == 0x80 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                if self.registers_64[register1] as u16 % 2 == 0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
                
            },
            32 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as u32;
                let reg2_value = self.registers_64[register2] as u32;

                self.registers_64[register1] = (reg1_value & reg2_value) as u64;

                if self.registers_64[register1] as u32 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }

                if self.registers_64[register1] as u32 & 0x80 == 0x80 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                if self.registers_64[register1] as u32 % 2 == 0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
                
            },
            64 => {
                check_register64!(register1, register2);

                let reg1_value = self.registers_64[register1] as u64;
                let reg2_value = self.registers_64[register2] as u64;

                self.registers_64[register1] = (reg1_value & reg2_value) as u64;

                if self.registers_64[register1] as u64 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }

                if self.registers_64[register1] as u64 & 0x80 == 0x80 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                if self.registers_64[register1] as u64 % 2 == 0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
                
            },
            128 => {
                check_register128!(register1, register2);

                let reg1_value = self.registers_128[register1] as u128;
                let reg2_value = self.registers_128[register2] as u128;

                self.registers_128[register1] = (reg1_value & reg2_value) as u128;

                if self.registers_64[register1] as u128 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }

                if self.registers_64[register1] as u128 & 0x80 == 0x80 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                if self.registers_64[register1] as u128 % 2 == 0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
            },
            _ => return Err(Fault::InvalidSize),
        }
        
        Ok(())
    }

    fn or_opcode(&mut self) -> Result<(),Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register2 = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1 as usize, register2 as usize);

                let reg1_value = self.registers_64[register1 as usize] as u8;
                let reg2_value = self.registers_64[register2 as usize] as u8;

                self.registers_64[register1 as usize] = (reg1_value | reg2_value) as u64;

                if self.registers_64[register1 as usize] as u8 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }

                if self.registers_64[register1 as usize] as u8 & 0x80 == 0x80 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                if self.registers_64[register1 as usize] as u8 % 2 == 0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
                
            },
            16 => {
                check_register64!(register1 as usize, register2 as usize);

                let reg1_value = self.registers_64[register1 as usize] as u16;
                let reg2_value = self.registers_64[register2 as usize] as u16;

                self.registers_64[register1 as usize] = (reg1_value | reg2_value) as u64;

                if self.registers_64[register1 as usize] as u16 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }

                if self.registers_64[register1 as usize] as u16 & 0x80 == 0x80 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                if self.registers_64[register1 as usize] as u16 % 2 == 0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
                
            },
            32 => {
                check_register64!(register1 as usize, register2 as usize);

                let reg1_value = self.registers_64[register1 as usize] as u32;
                let reg2_value = self.registers_64[register2 as usize] as u32;

                self.registers_64[register1 as usize] = (reg1_value | reg2_value) as u64;

                if self.registers_64[register1 as usize] as u32 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }

                if self.registers_64[register1 as usize] as u32 & 0x80 == 0x80 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                if self.registers_64[register1 as usize] as u32 % 2 == 0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
                
            },
            64 => {
                check_register64!(register1 as usize, register2 as usize);

                let reg1_value = self.registers_64[register1 as usize] as u64;
                let reg2_value = self.registers_64[register2 as usize] as u64;

                self.registers_64[register1 as usize] = (reg1_value | reg2_value) as u64;

                if self.registers_64[register1 as usize] as u64 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }

                if self.registers_64[register1 as usize] as u64 & 0x80 == 0x80 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                if self.registers_64[register1 as usize] as u64 % 2 == 0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
                
            },
            128 => {
                check_register128!(register1 as usize, register2 as usize);

                let reg1_value = self.registers_128[register1 as usize];
                let reg2_value = self.registers_128[register2 as usize];

                self.registers_128[register1 as usize] = reg1_value | reg2_value;

                if self.registers_128[register1 as usize] == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }

                if self.registers_128[register1 as usize] & 0x80 == 0x80 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                if self.registers_128[register1 as usize] % 2 == 0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
                
            },
            _ => return Err(Fault::InvalidSize),
        }
        Ok(())
    }

    fn xor_opcode(&mut self) -> Result<(), Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register2 = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1 as usize, register2 as usize);

                let reg1_value = self.registers_64[register1 as usize] as u8;
                let reg2_value = self.registers_64[register2 as usize] as u8;

                self.registers_64[register1 as usize] = (reg1_value ^ reg2_value) as u64;

                if self.registers_64[register1 as usize] as u8 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }

                if self.registers_64[register1 as usize] as u8 & 0x80 == 0x80 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                if self.registers_64[register1 as usize] as u8 % 2 == 0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
                
            },
            16 => {
                check_register64!(register1 as usize, register2 as usize);

                let reg1_value = self.registers_64[register1 as usize] as u16;
                let reg2_value = self.registers_64[register2 as usize] as u16;

                self.registers_64[register1 as usize] = (reg1_value ^ reg2_value) as u64;

                if self.registers_64[register1 as usize] as u16 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }

                if self.registers_64[register1 as usize] as u16 & 0x80 == 0x80 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                if self.registers_64[register1 as usize] as u16 % 2 == 0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
                
            },
            32 => {
                check_register64!(register1 as usize, register2 as usize);

                let reg1_value = self.registers_64[register1 as usize] as u32;
                let reg2_value = self.registers_64[register2 as usize] as u32;

                self.registers_64[register1 as usize] = (reg1_value ^ reg2_value) as u64;

                if self.registers_64[register1 as usize] as u32 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }

                if self.registers_64[register1 as usize] as u32 & 0x80 == 0x80 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                if self.registers_64[register1 as usize] as u32 % 2 == 0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
                
            },
            64 => {
                check_register64!(register1 as usize, register2 as usize);

                let reg1_value = self.registers_64[register1 as usize];
                let reg2_value = self.registers_64[register2 as usize];

                self.registers_64[register1 as usize] = reg1_value ^ reg2_value;

                if self.registers_64[register1 as usize] == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }

                if self.registers_64[register1 as usize] & 0x80 == 0x80 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                if self.registers_64[register1 as usize] % 2 == 0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
                
            },
            128 => {
                check_register128!(register1 as usize, register2 as usize);

                let reg1_value = self.registers_128[register1 as usize];
                let reg2_value = self.registers_128[register2 as usize];

                self.registers_128[register1 as usize] = reg1_value ^ reg2_value;

                if self.registers_128[register1 as usize] == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }

                if self.registers_128[register1 as usize] & 0x80 == 0x80 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                if self.registers_128[register1 as usize] % 2 == 0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
                
            },
            _ => return Err(Fault::InvalidSize),
        }
        Ok(())
    }

    fn not_opcode(&mut self) -> Result<(), Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register as usize);

                let reg_value = self.registers_64[register as usize] as u8;

                self.registers_64[register as usize] = (!reg_value) as u64;

                if self.registers_64[register as usize] as u8 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }

                if self.registers_64[register as usize] as u8 & 0x80 == 0x80 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                if self.registers_64[register as usize] as u8 % 2 == 0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
            },
            16 => {
                check_register64!(register as usize);

                let reg_value = self.registers_64[register as usize] as u16;

                self.registers_64[register as usize] = (!reg_value) as u64;

                if self.registers_64[register as usize] as u16 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }

                if self.registers_64[register as usize] as u16 & 0x80 == 0x80 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                if self.registers_64[register as usize] as u16 % 2 == 0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
            },
            32 => {
                check_register64!(register as usize);

                let reg_value = self.registers_64[register as usize] as u32;

                self.registers_64[register as usize] = (!reg_value) as u64;

                if self.registers_64[register as usize] as u32 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }

                if self.registers_64[register as usize] as u32 & 0x80 == 0x80 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                if self.registers_64[register as usize] as u32 % 2 == 0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
            },
            64 => {
                check_register64!(register as usize);

                let reg_value = self.registers_64[register as usize];

                self.registers_64[register as usize] = !reg_value;

                if self.registers_64[register as usize] == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }

                if self.registers_64[register as usize] & 0x80 == 0x80 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                if self.registers_64[register as usize] % 2 == 0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
            },
            128 => {
                check_register128!(register as usize);

                let reg_value = self.registers_128[register as usize];

                self.registers_128[register as usize] = !reg_value;

                if self.registers_128[register as usize] == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }

                if self.registers_128[register as usize] & 0x80 == 0x80 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                if self.registers_128[register as usize] % 2 == 0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
            },
            _ => return Err(Fault::InvalidSize),
        }
        Ok(())
    }

    fn shiftleft_opcode(&mut self) -> Result<(), Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let shift_amount = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register as usize);

                let reg_value = self.registers_64[register as usize] as u8;

                self.registers_64[register as usize] = (reg_value << shift_amount) as u64;

                if self.registers_64[register as usize] as u8 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }

                if self.registers_64[register as usize] as u8 & 0x80 == 0x80 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                if self.registers_64[register as usize] as u8 % 2 == 0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
            },
            16 => {
                check_register64!(register as usize);

                let reg_value = self.registers_64[register as usize] as u16;

                self.registers_64[register as usize] = (reg_value << shift_amount) as u64;

                if self.registers_64[register as usize] as u16 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }

                if self.registers_64[register as usize] as u16 & 0x80 == 0x80 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                if self.registers_64[register as usize] as u16 % 2 == 0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
            },
            32 => {
                check_register64!(register as usize);

                let reg_value = self.registers_64[register as usize] as u32;

                self.registers_64[register as usize] = (reg_value << shift_amount) as u64;

                if self.registers_64[register as usize] as u32 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }

                if self.registers_64[register as usize] as u32 & 0x80 == 0x80 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                if self.registers_64[register as usize] as u32 % 2 == 0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
            },
            64 => {
                check_register64!(register as usize);

                let reg_value = self.registers_64[register as usize];

                self.registers_64[register as usize] = reg_value << shift_amount;

                if self.registers_64[register as usize] == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }

                if self.registers_64[register as usize] & 0x80 == 0x80 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                if self.registers_64[register as usize] % 2 == 0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
            },
            128 => {
                check_register128!(register as usize);

                let reg_value = self.registers_128[register as usize];

                self.registers_128[register as usize] = reg_value << shift_amount;

                if self.registers_128[register as usize] == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }

                if self.registers_128[register as usize] & 0x80 == 0x80 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                if self.registers_128[register as usize] % 2 == 0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
            },
            _ => return Err(Fault::InvalidSize),
        }
        Ok(())
    }

    fn shiftright_opcode(&mut self) -> Result<(), Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let shift_amount = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register as usize);

                let reg_value = self.registers_64[register as usize] as u8;

                self.registers_64[register as usize] = (reg_value >> shift_amount) as u64;

                if self.registers_64[register as usize] as u8 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }

                if self.registers_64[register as usize] as u8 & 0x80 == 0x80 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                if self.registers_64[register as usize] as u8 % 2 == 0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
            },
            16 => {
                check_register64!(register as usize);

                let reg_value = self.registers_64[register as usize] as u16;

                self.registers_64[register as usize] = (reg_value >> shift_amount) as u64;

                if self.registers_64[register as usize] as u16 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }

                if self.registers_64[register as usize] as u16 & 0x80 == 0x80 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                if self.registers_64[register as usize] as u16 % 2 == 0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
            },
            32 => {
                check_register64!(register as usize);

                let reg_value = self.registers_64[register as usize] as u32;

                self.registers_64[register as usize] = (reg_value >> shift_amount) as u64;

                if self.registers_64[register as usize] as u32 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }

                if self.registers_64[register as usize] as u32 & 0x80 == 0x80 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                if self.registers_64[register as usize] as u32 % 2 == 0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
            },
            64 => {
                check_register64!(register as usize);

                let reg_value = self.registers_64[register as usize];

                self.registers_64[register as usize] = reg_value >> shift_amount;

                if self.registers_64[register as usize] == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }

                if self.registers_64[register as usize] & 0x80 == 0x80 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                if self.registers_64[register as usize] % 2 == 0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
            },
            128 => {
                check_register128!(register as usize);

                let reg_value = self.registers_128[register as usize];

                self.registers_128[register as usize] = reg_value >> shift_amount;

                if self.registers_128[register as usize] == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }

                if self.registers_128[register as usize] & 0x80 == 0x80 {
                    self.sign_flag = Sign::Negative;
                }
                else {
                    self.sign_flag = Sign::Positive;
                }

                if self.registers_128[register as usize] % 2 == 0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
            },
            _ => return Err(Fault::InvalidSize),
        }
        Ok(())
    }

    fn clear_opcode(&mut self) -> Result<(), Fault> {
        self.zero_flag = false;
        self.remainder_64 = 0;
        self.remainder_128 = 0;
        self.comparison_flag = Comparison::NotEqual;
        Ok(())
    }

    fn writebyte_opcode(&mut self) -> Result<(),Fault> {
        let fd_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let value_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        check_register64!(fd_register as usize, value_register as usize);

        let fd = self.registers_64[fd_register as usize] as i64;
        let value = self.registers_64[value_register as usize] as u8;

        //TODO: check if fd is valid. Make it used a shared memory object for non-std fds
        match fd {
            1 => {
                std::io::stdout().write(&[value]).unwrap();
            },
            2 => {
                std::io::stderr().write(&[value]).unwrap();
            },
            _ => return Err(Fault::InvalidFileDescriptor),
        }
        Ok(())
    }

    fn write_opcode(&mut self) -> Result<(), Fault> {
        let fd_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let pointer_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let length_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        check_register64!(fd_register as usize, pointer_register as usize, length_register as usize);

        let fd = self.registers_64[fd_register as usize] as i64;
        let pointer = self.registers_64[pointer_register as usize] as u64;
        let length = self.registers_64[length_register as usize] as usize;

        let memory = self.memory.read().unwrap();


        //TODO: check if fd is valid. Make it used a shared memory object for non-std fds
        match fd {
            1 => {
                std::io::stdout().write(&memory[pointer as usize..pointer as usize + length as usize]).unwrap();
            },
            2 => {
                std::io::stderr().write(&memory[pointer as usize..pointer as usize + length as usize]).unwrap();
            },
            _ => return Err(Fault::InvalidFileDescriptor),
        }
        Ok(())
    }

    fn flush_opcode(&mut self) -> Result<(), Fault> {
        let fd_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        check_register64!(fd_register as usize);

        let fd = self.registers_64[fd_register as usize] as i64;

        match fd {
            1 => {
                std::io::stdout().flush().unwrap();
            },
            2 => {
                std::io::stderr().flush().unwrap();
            },
            _ => return Err(Fault::InvalidFileDescriptor),
        }

        Ok(())
    }

    fn remainder_opcode(&mut self) -> Result<(), Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        let register = self.program[self.program_counter] as u8;

        match size {
            8 | 16 | 32 | 64 => {
                check_register64!(register as usize);

                self.registers_64[register as usize] = self.remainder_64 as u64;
                self.remainder_64 = 0;
            },
            128 => {
                check_register128!(register as usize);

                self.registers_128[register as usize] = self.remainder_128;
                self.remainder_128 = 0;
            },
            _ => return Err(Fault::InvalidSize),

        }

        Ok(())
    }

    fn addfi_opcode(&mut self) -> Result<(), Fault> {
        let float_size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        let float_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let int_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        check_register64!(int_register as usize);

        match float_size {
            32 => {
                check_registerF32!(float_register as usize);

                self.registers_f32[float_register as usize] += self.registers_64[int_register as usize] as f32;

                if self.registers_f32[float_register as usize].is_nan() {
                    self.nan_flag = true;
                }
                else {
                    self.nan_flag = false;
                }
                if self.registers_f32[float_register as usize].is_infinite() {
                    self.infinity_flag = true;
                }
                else {
                    self.infinity_flag = false;
                }
                if self.registers_f32[float_register as usize] == 0.0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                
            },
            64 => {
                check_registerF64!(float_register as usize);

                self.registers_f64[float_register as usize] += self.registers_64[int_register as usize] as f64;

                if self.registers_f64[float_register as usize].is_nan() {
                    self.nan_flag = true;
                }
                else {
                    self.nan_flag = false;
                }
                if self.registers_f64[float_register as usize].is_infinite() {
                    self.infinity_flag = true;
                }
                else {
                    self.infinity_flag = false;
                }
                if self.registers_f64[float_register as usize] == 0.0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
            },
            _ => return Err(Fault::InvalidSize),
        }
        Ok(())

    }

    fn subfi_opcode(&mut self) -> Result<(), Fault> {
        let float_size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        let float_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let int_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        check_register64!(int_register as usize);

        match float_size {
            32 => {
                check_registerF32!(float_register as usize);

                self.registers_f32[float_register as usize] -= self.registers_64[int_register as usize] as f32;

                if self.registers_f32[float_register as usize].is_nan() {
                    self.nan_flag = true;
                }
                else {
                    self.nan_flag = false;
                }
                if self.registers_f32[float_register as usize].is_infinite() {
                    self.infinity_flag = true;
                }
                else {
                    self.infinity_flag = false;
                }
                if self.registers_f32[float_register as usize] == 0.0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                
            },
            64 => {
                check_registerF64!(float_register as usize);

                self.registers_f64[float_register as usize] -= self.registers_64[int_register as usize] as f64;

                if self.registers_f64[float_register as usize].is_nan() {
                    self.nan_flag = true;
                }
                else {
                    self.nan_flag = false;
                }
                if self.registers_f64[float_register as usize].is_infinite() {
                    self.infinity_flag = true;
                }
                else {
                    self.infinity_flag = false;
                }
                if self.registers_f64[float_register as usize] == 0.0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
            },
            _ => return Err(Fault::InvalidSize),
        }
        Ok(())

    }

    fn mulfi_opcode(&mut self) -> Result<(), Fault> {
        let float_size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        let float_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let int_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        check_register64!(int_register as usize);

        match float_size {
            32 => {
                check_registerF32!(float_register as usize);

                self.registers_f32[float_register as usize] *= self.registers_64[int_register as usize] as f32;

                if self.registers_f32[float_register as usize].is_nan() {
                    self.nan_flag = true;
                }
                else {
                    self.nan_flag = false;
                }
                if self.registers_f32[float_register as usize].is_infinite() {
                    self.infinity_flag = true;
                }
                else {
                    self.infinity_flag = false;
                }
                if self.registers_f32[float_register as usize] == 0.0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                
            },
            64 => {
                check_registerF64!(float_register as usize);

                self.registers_f64[float_register as usize] *= self.registers_64[int_register as usize] as f64;

                if self.registers_f64[float_register as usize].is_nan() {
                    self.nan_flag = true;
                }
                else {
                    self.nan_flag = false;
                }
                if self.registers_f64[float_register as usize].is_infinite() {
                    self.infinity_flag = true;
                }
                else {
                    self.infinity_flag = false;
                }
                if self.registers_f64[float_register as usize] == 0.0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
            },
            _ => return Err(Fault::InvalidSize),
        }
        Ok(())

    }

    fn divfi_opcode(&mut self) -> Result<(), Fault> {
        let float_size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        let float_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let int_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        check_register64!(int_register as usize);

        match float_size {
            32 => {
                check_registerF32!(float_register as usize);

                self.registers_f32[float_register as usize] /= self.registers_64[int_register as usize] as f32;

                if self.registers_f32[float_register as usize].is_nan() {
                    self.nan_flag = true;
                }
                else {
                    self.nan_flag = false;
                }
                if self.registers_f32[float_register as usize].is_infinite() {
                    self.infinity_flag = true;
                }
                else {
                    self.infinity_flag = false;
                }
                if self.registers_f32[float_register as usize] == 0.0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                
            },
            64 => {
                check_registerF64!(float_register as usize);

                self.registers_f64[float_register as usize] /= self.registers_64[int_register as usize] as f64;

                if self.registers_f64[float_register as usize].is_nan() {
                    self.nan_flag = true;
                }
                else {
                    self.nan_flag = false;
                }
                if self.registers_f64[float_register as usize].is_infinite() {
                    self.infinity_flag = true;
                }
                else {
                    self.infinity_flag = false;
                }
                if self.registers_f64[float_register as usize] == 0.0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
            },
            _ => return Err(Fault::InvalidSize),
        }
        Ok(())
    }

    fn addif_opcode(&mut self) -> Result<(), Fault> {
        let int_size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let float_size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let int_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let float_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        check_register64!(int_register as usize);

        match int_size {
            8 => {
                let int_value = self.program[self.program_counter] as i8;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as i8
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as i8
                    },
                    _ => return Err(Fault::InvalidSize),

                };


                let new_value = (Wrapping(int_value) + Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] as u8 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] as i8 > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            16 => {
                let int_value = self.program[self.program_counter] as i16;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as i16
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as i16
                    },
                    _ => return Err(Fault::InvalidSize),

                };


                let new_value = (Wrapping(int_value) + Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] as u16 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] as i16 > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            32 => {
                let int_value = self.program[self.program_counter] as i32;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as i32
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as i32
                    },
                    _ => return Err(Fault::InvalidSize),

                };


                let new_value = (Wrapping(int_value) + Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] as u32 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] as i32 > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            64 => {
                let int_value = self.program[self.program_counter] as i64;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as i64
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as i64
                    },
                    _ => return Err(Fault::InvalidSize),

                };


                let new_value = (Wrapping(int_value) + Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            128 => {
                let int_value = self.program[self.program_counter] as i128;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as i128
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as i128
                    },
                    _ => return Err(Fault::InvalidSize),

                };

                let new_value = (Wrapping(int_value) + Wrapping(float_value)).0;

                self.registers_128[int_register as usize] = new_value as u128;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_128[int_register as usize] == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_128[int_register as usize] > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            _ => return Err(Fault::InvalidSize),
            

        }
        Ok(())
    }

    fn subif_opcode(&mut self) -> Result<(), Fault> {
        let int_size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let float_size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let int_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let float_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        check_register64!(int_register as usize);

        match int_size {
            8 => {
                let int_value = self.program[self.program_counter] as i8;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as i8
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as i8
                    },
                    _ => return Err(Fault::InvalidSize),

                };


                let new_value = (Wrapping(int_value) - Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] as u8 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] as i8 > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            16 => {
                let int_value = self.program[self.program_counter] as i16;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as i16
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as i16
                    },
                    _ => return Err(Fault::InvalidSize),

                };


                let new_value = (Wrapping(int_value) - Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] as u16 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] as i16 > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            32 => {
                let int_value = self.program[self.program_counter] as i32;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as i32
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as i32
                    },
                    _ => return Err(Fault::InvalidSize),

                };


                let new_value = (Wrapping(int_value) - Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] as u32 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] as i32 > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            64 => {
                let int_value = self.program[self.program_counter] as i64;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as i64
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as i64
                    },
                    _ => return Err(Fault::InvalidSize),

                };


                let new_value = (Wrapping(int_value) - Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            128 => {
                let int_value = self.program[self.program_counter] as i128;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as i128
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as i128
                    },
                    _ => return Err(Fault::InvalidSize),

                };

                let new_value = (Wrapping(int_value) - Wrapping(float_value)).0;

                self.registers_128[int_register as usize] = new_value as u128;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_128[int_register as usize] == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_128[int_register as usize] > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            _ => return Err(Fault::InvalidSize),
            

        }
        Ok(())
    }

    fn mulif_opcode(&mut self) -> Result<(), Fault> {
        let int_size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let float_size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let int_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let float_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        check_register64!(int_register as usize);

        match int_size {
            8 => {
                let int_value = self.program[self.program_counter] as i8;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as i8
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as i8
                    },
                    _ => return Err(Fault::InvalidSize),

                };


                let new_value = (Wrapping(int_value) * Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] as u8 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] as i8 > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            16 => {
                let int_value = self.program[self.program_counter] as i16;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as i16
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as i16
                    },
                    _ => return Err(Fault::InvalidSize),

                };


                let new_value = (Wrapping(int_value) * Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] as u16 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] as i16 > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            32 => {
                let int_value = self.program[self.program_counter] as i32;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as i32
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as i32
                    },
                    _ => return Err(Fault::InvalidSize),

                };


                let new_value = (Wrapping(int_value) * Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] as u32 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] as i32 > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            64 => {
                let int_value = self.program[self.program_counter] as i64;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as i64
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as i64
                    },
                    _ => return Err(Fault::InvalidSize),

                };


                let new_value = (Wrapping(int_value) * Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            128 => {
                let int_value = self.program[self.program_counter] as i128;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as i128
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as i128
                    },
                    _ => return Err(Fault::InvalidSize),

                };

                let new_value = (Wrapping(int_value) * Wrapping(float_value)).0;

                self.registers_128[int_register as usize] = new_value as u128;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_128[int_register as usize] == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_128[int_register as usize] > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            _ => return Err(Fault::InvalidSize),
            

        }
        Ok(())
    }

    fn divif_opcode(&mut self) -> Result<(), Fault> {
        let int_size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let float_size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let int_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let float_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        check_register64!(int_register as usize);

        match int_size {
            8 => {
                let int_value = self.program[self.program_counter] as i8;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as i8
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as i8
                    },
                    _ => return Err(Fault::InvalidSize),

                };

                self.remainder_64 = (Wrapping(int_value) % Wrapping(float_value)).0 as usize;

                let new_value = (Wrapping(int_value) / Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] as u8 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] as i8 > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            16 => {
                let int_value = self.program[self.program_counter] as i16;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as i16
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as i16
                    },
                    _ => return Err(Fault::InvalidSize),

                };

                self.remainder_64 = (Wrapping(int_value) % Wrapping(float_value)).0 as usize;

                let new_value = (Wrapping(int_value) / Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] as u16 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] as i16 > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            32 => {
                let int_value = self.program[self.program_counter] as i32;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as i32
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as i32
                    },
                    _ => return Err(Fault::InvalidSize),

                };

                self.remainder_64 = (Wrapping(int_value) % Wrapping(float_value)).0 as usize;

                let new_value = (Wrapping(int_value) / Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] as u32 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] as i32 > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            64 => {
                let int_value = self.program[self.program_counter] as i64;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as i64
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as i64
                    },
                    _ => return Err(Fault::InvalidSize),

                };

                self.remainder_64 = (Wrapping(int_value) % Wrapping(float_value)).0 as usize;

                let new_value = (Wrapping(int_value) / Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            128 => {
                let int_value = self.program[self.program_counter] as i128;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as i128
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as i128
                    },
                    _ => return Err(Fault::InvalidSize),

                };
                self.remainder_128 = (Wrapping(int_value) % Wrapping(float_value)).0 as u128;

                let new_value = (Wrapping(int_value) / Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_128[int_register as usize] == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_128[int_register as usize] > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            _ => return Err(Fault::InvalidSize),
            

        }
        Ok(())
    }
        
    fn adduf_opcode(&mut self) -> Result<(), Fault> {
        let int_size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let float_size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let int_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let float_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        check_register64!(int_register as usize);

        match int_size {
            8 => {
                let int_value = self.program[self.program_counter] as u8;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as u8
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as u8
                    },
                    _ => return Err(Fault::InvalidSize),

                };


                let new_value = (Wrapping(int_value) + Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] as u8 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] as u8 > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            16 => {
                let int_value = self.program[self.program_counter] as u16;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as u16
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as u16
                    },
                    _ => return Err(Fault::InvalidSize),

                };


                let new_value = (Wrapping(int_value) + Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] as u16 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] as u16 > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            32 => {
                let int_value = self.program[self.program_counter] as u32;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as u32
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as u32
                    },
                    _ => return Err(Fault::InvalidSize),

                };


                let new_value = (Wrapping(int_value) + Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] as u32 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] as u32 > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            64 => {
                let int_value = self.program[self.program_counter] as u64;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as u64
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as u64
                    },
                    _ => return Err(Fault::InvalidSize),

                };


                let new_value = (Wrapping(int_value) + Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            128 => {
                let int_value = self.program[self.program_counter] as u128;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as u128
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as u128
                    },
                    _ => return Err(Fault::InvalidSize),

                };

                let new_value = (Wrapping(int_value) + Wrapping(float_value)).0;

                self.registers_128[int_register as usize] = new_value as u128;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_128[int_register as usize] == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_128[int_register as usize] > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            _ => return Err(Fault::InvalidSize),
            

        }
        Ok(())
    }

    fn subuf_opcode(&mut self) -> Result<(), Fault> {
        let int_size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let float_size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let int_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let float_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        check_register64!(int_register as usize);

        match int_size {
            8 => {
                let int_value = self.program[self.program_counter] as u8;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as u8
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as u8
                    },
                    _ => return Err(Fault::InvalidSize),

                };


                let new_value = (Wrapping(int_value) - Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] as u8 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] as u8 > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            16 => {
                let int_value = self.program[self.program_counter] as u16;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as u16
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as u16
                    },
                    _ => return Err(Fault::InvalidSize),

                };


                let new_value = (Wrapping(int_value) - Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] as u16 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] as u16 > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            32 => {
                let int_value = self.program[self.program_counter] as u32;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as u32
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as u32
                    },
                    _ => return Err(Fault::InvalidSize),

                };


                let new_value = (Wrapping(int_value) - Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] as u32 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] as u32 > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            64 => {
                let int_value = self.program[self.program_counter] as u64;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as u64
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as u64
                    },
                    _ => return Err(Fault::InvalidSize),

                };


                let new_value = (Wrapping(int_value) - Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            128 => {
                let int_value = self.program[self.program_counter] as u128;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as u128
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as u128
                    },
                    _ => return Err(Fault::InvalidSize),

                };

                let new_value = (Wrapping(int_value) - Wrapping(float_value)).0;

                self.registers_128[int_register as usize] = new_value as u128;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_128[int_register as usize] == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_128[int_register as usize] > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            _ => return Err(Fault::InvalidSize),
            

        }
        Ok(())
    }

    fn muluf_opcode(&mut self) -> Result<(), Fault> {
        let int_size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let float_size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let int_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let float_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        check_register64!(int_register as usize);

        match int_size {
            8 => {
                let int_value = self.program[self.program_counter] as u8;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as u8
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as u8
                    },
                    _ => return Err(Fault::InvalidSize),

                };


                let new_value = (Wrapping(int_value) * Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] as u8 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] as u8 > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            16 => {
                let int_value = self.program[self.program_counter] as u16;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as u16
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as u16
                    },
                    _ => return Err(Fault::InvalidSize),

                };


                let new_value = (Wrapping(int_value) * Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] as u16 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] as u16 > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            32 => {
                let int_value = self.program[self.program_counter] as u32;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as u32
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as u32
                    },
                    _ => return Err(Fault::InvalidSize),

                };


                let new_value = (Wrapping(int_value) * Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] as u32 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] as u32 > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            64 => {
                let int_value = self.program[self.program_counter] as u64;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as u64
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as u64
                    },
                    _ => return Err(Fault::InvalidSize),

                };


                let new_value = (Wrapping(int_value) * Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            128 => {
                let int_value = self.program[self.program_counter] as u128;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as u128
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as u128
                    },
                    _ => return Err(Fault::InvalidSize),

                };

                let new_value = (Wrapping(int_value) * Wrapping(float_value)).0;

                self.registers_128[int_register as usize] = new_value as u128;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_128[int_register as usize] == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_128[int_register as usize] > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            _ => return Err(Fault::InvalidSize),
            

        }
        Ok(())
    }

    fn divuf_opcode(&mut self) -> Result<(), Fault> {
        let int_size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let float_size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let int_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let float_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        check_register64!(int_register as usize);

        match int_size {
            8 => {
                let int_value = self.program[self.program_counter] as u8;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as u8
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as u8
                    },
                    _ => return Err(Fault::InvalidSize),

                };

                self.remainder_64 = (Wrapping(int_value) % Wrapping(float_value)).0 as usize;

                let new_value = (Wrapping(int_value) / Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] as u8 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] as u8 > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            16 => {
                let int_value = self.program[self.program_counter] as u16;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as u16
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as u16
                    },
                    _ => return Err(Fault::InvalidSize),

                };

                self.remainder_64 = (Wrapping(int_value) % Wrapping(float_value)).0 as usize;

                let new_value = (Wrapping(int_value) / Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] as u16 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] as u16 > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            32 => {
                let int_value = self.program[self.program_counter] as u32;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as u32
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as u32
                    },
                    _ => return Err(Fault::InvalidSize),

                };

                self.remainder_64 = (Wrapping(int_value) % Wrapping(float_value)).0 as usize;

                let new_value = (Wrapping(int_value) / Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] as u32 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] as u32 > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            64 => {
                let int_value = self.program[self.program_counter] as u64;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as u64
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as u64
                    },
                    _ => return Err(Fault::InvalidSize),

                };

                self.remainder_64 = (Wrapping(int_value) % Wrapping(float_value)).0 as usize;

                let new_value = (Wrapping(int_value) / Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_64[int_register as usize] > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            128 => {
                let int_value = self.program[self.program_counter] as u128;
                
                let float_value = match float_size {
                    32 => {
                        check_registerF32!(float_register as usize);

                        self.registers_f32[float_register as usize] as u128
                    },
                    64 => {
                        check_registerF64!(float_register as usize);

                        self.registers_f64[float_register as usize] as u128
                    },
                    _ => return Err(Fault::InvalidSize),

                };
                self.remainder_128 = (Wrapping(int_value) % Wrapping(float_value)).0 as u128;

                let new_value = (Wrapping(int_value) / Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = new_value as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_128[int_register as usize] == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                }
                if self.registers_128[int_register as usize] > 0 {
                    self.sign_flag = Sign::Positive;
                }
                else {
                    self.sign_flag = Sign::Negative;
                }
            },
            _ => return Err(Fault::InvalidSize),
            

        }
        Ok(())
    }


    fn setf_opcode(&mut self) -> Result<(), Fault> {
        let size = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        let register = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        match size {
            32 => {
                check_registerF32!(register as usize);
                let mut value = 0.0f32.to_ne_bytes();
                value[0] = self.program[self.program_counter];
                value[1] = self.program[self.program_counter + 1];
                value[2] = self.program[self.program_counter + 2];
                value[3] = self.program[self.program_counter + 3];

                self.advance_by_4_bytes();
                
                self.registers_f32[register as usize] = f32::from_ne_bytes(value);
            },
            64 => {
                check_registerF64!(register as usize);
                let mut value = 0.0f64.to_ne_bytes();
                value[0] = self.program[self.program_counter];
                value[1] = self.program[self.program_counter + 1];
                value[2] = self.program[self.program_counter + 2];
                value[3] = self.program[self.program_counter + 3];
                value[4] = self.program[self.program_counter + 4];
                value[5] = self.program[self.program_counter + 5];
                value[6] = self.program[self.program_counter + 6];
                value[7] = self.program[self.program_counter + 7];
                
                self.advance_by_8_bytes();

                self.registers_f64[register as usize] = f64::from_ne_bytes(value);
            },
            _ => return Err(Fault::InvalidSize),
        }
        Ok(())
    }

    fn dereff_opcode(&mut self) -> Result<(), Fault> {
        let size = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        let register = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        let address = self.registers_64[register];
        match size {
            32 => {
                loop {
                    match self.memory.try_read() {
                        Ok(memory) => {
                            if address >= memory.len() as u64 {
                                return Err(Fault::InvalidAddress(address));
                            }
                            let mut bytes = 0.0f32.to_ne_bytes();
                            bytes[0] = memory[address as usize];
                            bytes[1] = memory[address as usize + 1];
                            bytes[2] = memory[address as usize + 2];
                            bytes[3] = memory[address as usize + 3];

                            self.registers_f32[register] = f32::from_ne_bytes(bytes);
                            break;
                        },
                        Err(TryLockError::WouldBlock) => {
                            continue;
                        },
                        Err(_) => return Err(Fault::CorruptedMemory),
                    }

                }
                self.advance_by_4_bytes();
            }
            64 => {
                loop {
                    match self.memory.try_read() {
                        Ok(memory) => {
                            if address >= memory.len() as u64 {
                                return Err(Fault::InvalidAddress(address));
                            }
                            let mut bytes = 0.0f64.to_ne_bytes();
                            bytes[0] = memory[address as usize];
                            bytes[1] = memory[address as usize + 1];
                            bytes[2] = memory[address as usize + 2];
                            bytes[3] = memory[address as usize + 3];
                            bytes[4] = memory[address as usize + 4];
                            bytes[5] = memory[address as usize + 5];
                            bytes[6] = memory[address as usize + 6];
                            bytes[7] = memory[address as usize + 7];

                            self.registers_f64[register] = f64::from_ne_bytes(bytes);
                            break;
                        },
                        Err(TryLockError::WouldBlock) => {
                            continue;
                        },
                        Err(_) => return Err(Fault::CorruptedMemory),
                    }

                }
                self.advance_by_8_bytes();
            }
            _ => return Err(Fault::InvalidSize),
        }
        Ok(())
    }

    fn movef_opcode(&mut self) -> Result<(), Fault> {
        let size = self.program[self.program_counter] as u8 as usize;
        self.advance_by_1_byte();
        let register = self.program[self.program_counter] as u8 as usize;
        self.advance_by_1_byte();
        match size {
            32 => {
                check_registerF32!(register as usize);
                let address = self.program[self.program_counter] as u64;
                self.advance_by_8_bytes();
                loop {
                    match self.memory.try_write() {
                        Ok(mut memory) => {
                            if address >= memory.len() as u64 {
                                return Err(Fault::InvalidAddress(address));
                            }
                            let float_bytes = self.registers_f32[register as usize].to_ne_bytes();
                            memory[address as usize] = float_bytes[0];
                            memory[address as usize + 1] = float_bytes[1];
                            memory[address as usize + 2] = float_bytes[2];
                            memory[address as usize + 3] = float_bytes[3];
                            
                            break;
                        },
                        Err(TryLockError::WouldBlock) => {
                            thread::yield_now();
                            continue;
                        },
                        Err(_) => return Err(Fault::CorruptedMemory),
                    }
                }

            },
            64 => {
                check_registerF64!(register as usize);
                let address = self.program[self.program_counter] as u64;
                self.advance_by_8_bytes();
                loop {
                    match self.memory.try_write() {
                        Ok(mut memory) => {
                            if address >= memory.len() as u64 {
                                return Err(Fault::InvalidAddress(address));
                            }
                            let float_bytes = self.registers_f64[register as usize].to_ne_bytes();
                            memory[address as usize] = float_bytes[0];
                            memory[address as usize + 1] = float_bytes[1];
                            memory[address as usize + 2] = float_bytes[2];
                            memory[address as usize + 3] = float_bytes[3];
                            memory[address as usize + 4] = float_bytes[4];
                            memory[address as usize + 5] = float_bytes[5];
                            memory[address as usize + 6] = float_bytes[6];
                            memory[address as usize + 7] = float_bytes[7];
                            
                            break;
                        },
                        Err(TryLockError::WouldBlock) => {
                            thread::yield_now();
                            continue;
                        },
                        Err(_) => return Err(Fault::CorruptedMemory),
                    }
                }
            },
            _ => return Err(Fault::InvalidSize),

        }
        Ok(())
    }

    fn derefregf_opcode(&mut self) -> Result<(), Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        let address_register = self.program[self.program_counter] as usize;
        check_register64!(address_register);
        self.advance_by_1_byte();
        let offset = self.program[self.program_counter] as i64;
        self.advance_by_8_bytes();
        let address = self.registers_64[address_register] as i64 + offset;
        let address = address as u64;

        match size {
            32 => {
                check_registerF32!(register);
                loop {
                    match self.memory.try_read() {
                        Ok(memory) => {
                            if address >= memory.len() as u64 {
                                return Err(Fault::InvalidAddress(address));
                            }
                            let mut bytes = 0.0f32.to_ne_bytes();
                            bytes[0] = memory[address as usize];
                            bytes[1] = memory[address as usize + 1];
                            bytes[2] = memory[address as usize + 2];
                            bytes[3] = memory[address as usize + 3];

                            self.registers_f32[register] = f32::from_ne_bytes(bytes);
                            break;
                        },
                        Err(TryLockError::WouldBlock) => {
                            continue;
                        },
                        Err(_) => return Err(Fault::CorruptedMemory),
                    }

                }
            },
            64 => {
                check_registerF64!(register);
                loop {
                    match self.memory.try_read() {
                        Ok(memory) => {
                            if address >= memory.len() as u64 {
                                return Err(Fault::InvalidAddress(address));
                            }
                            let mut bytes = 0.0f64.to_ne_bytes();
                            bytes[0] = memory[address as usize];
                            bytes[1] = memory[address as usize + 1];
                            bytes[2] = memory[address as usize + 2];
                            bytes[3] = memory[address as usize + 3];
                            bytes[4] = memory[address as usize + 4];
                            bytes[5] = memory[address as usize + 5];
                            bytes[6] = memory[address as usize + 6];
                            bytes[7] = memory[address as usize + 7];

                            self.registers_f64[register] = f64::from_ne_bytes(bytes);
                            break;
                        },
                        Err(TryLockError::WouldBlock) => {
                            continue;
                        },
                        Err(_) => return Err(Fault::CorruptedMemory),
                    }

                }
            },
            _ => {
                return Err(Fault::InvalidSize);
            }

        }

        Ok(())
        
    }

        
    
}





#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_addi() {
        let program = Arc::new(vec![6,0,64,0,1]);
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
        let mut core = Core::new(memory, Arc::new(program));

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
        let mut core = Core::new(memory, Arc::new(program));

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
        let mut core = Core::new(memory, Arc::new(program));

        core.registers_64[0] = 4;
        core.registers_64[1] = 3;

        core.run(0).unwrap();

        assert_eq!(core.registers_64[0] as i64, 1);
        assert_eq!(core.remainder_64 as i64, 1);
        assert_eq!(core.sign_flag, Sign::Positive);
    }

    #[test]
    fn test_divi_by_zero() {
        let program = vec![9,0,64,0,1];
        let memory = Arc::new(RwLock::new(Vec::new()));
        let mut core = Core::new(memory, Arc::new(program));

        core.registers_64[0] = 4;
        core.registers_64[1] = 0;

        let result = core.run(0);

        if result.is_ok() {
            panic!("Divide by zero did not return an error");
        }
        else {
            assert_eq!(result.unwrap_err(), Fault::DivideByZero, "Divide by zero was successfull but should have failed");
        }
    }

    #[test]
    fn test_addi_overflow() {
        let program = vec![6,0,8,0,1];
        let memory = Arc::new(RwLock::new(Vec::new()));
        let mut core = Core::new(memory, Arc::new(program));

        core.registers_64[0] = 127;
        core.registers_64[1] = 2;

        core.run(0).unwrap();

        assert_eq!(core.registers_64[0] as i8, -127);
        assert_eq!(core.overflow_flag, true);
    }

    #[test]
    fn test_eqi() {
        let program = vec![10,0,64,0,1];
        let memory = Arc::new(RwLock::new(Vec::new()));
        let mut core = Core::new(memory, Arc::new(program));

        core.registers_64[0] = 1;
        core.registers_64[1] = 1;

        core.run(0).unwrap();

        assert_eq!(core.comparison_flag, Comparison::Equal);
    }

    #[test]
    fn test_lti() {
        let program = vec![12,0,64,0,1];
        let memory = Arc::new(RwLock::new(Vec::new()));
        let mut core = Core::new(memory, Arc::new(program));

        core.registers_64[0] = 1;
        core.registers_64[1] = 2;

        core.run(0).unwrap();

        assert_eq!(core.comparison_flag, Comparison::LessThan);
    }

    #[test]
    fn test_geqi() {
        let program = vec![15,0,64,0,1];
        let memory = Arc::new(RwLock::new(Vec::new()));
        let mut core = Core::new(memory, Arc::new(program));

        core.registers_64[0] = 2;
        core.registers_64[1] = 1;

        core.run(0).unwrap();

        assert_eq!(core.comparison_flag, Comparison::GreaterThanOrEqual);
    }

    #[test]
    fn test_addu() {
        let program = vec![16,0,128,0,1];
        let memory = Arc::new(RwLock::new(Vec::new()));
        let mut core = Core::new(memory, Arc::new(program));

        let value1 = 1;
        let value2 = 2;

        core.registers_128[0] = value1;
        core.registers_128[1] = value2;

        core.run(0).unwrap();

        assert_eq!(core.registers_128[0], value1 + value2);
    }

    #[test]
    fn test_unsigned_overflow() {
        let program = vec![16,0,8,0,1];
        let memory = Arc::new(RwLock::new(Vec::new()));
        let mut core = Core::new(memory, Arc::new(program));
        
        let value1:u8 = 1;
        let value2:u8 = 255;

        core.registers_64[0] = value1 as u64;
        core.registers_64[1] = value2 as u64;

        core.run(0).unwrap();

        assert_eq!(core.registers_64[0] as u8, value1.wrapping_add(value2));
        assert_eq!(core.overflow_flag, true);
    }

    #[test]
    fn test_write_byte() {
        let program = vec![145,0,0,1,145,0,0,2,145,0,0,3,145,0,0,4];
        let memory = Arc::new(RwLock::new(vec![]));
        let mut core = Core::new(memory.clone(), Arc::new(program));

        core.registers_64[0] = 1;
        core.registers_64[1] = 72;
        core.registers_64[2] = 105;
        core.registers_64[3] = 33;
        core.registers_64[4] = 10;

        
        core.run(0).unwrap();
        
        std::io::stdout().flush().unwrap();
        
    }

    #[test]
    fn test_hello_world() {
        let program = vec![146,0,0,1,2,149,0,0];
        let memory = vec![0, 104,101,108,108,111,32,119,111,114,108,100,10];
        let memory = Arc::new(RwLock::new(memory));
        let mut core = Core::new(memory.clone(), Arc::new(program));

        core.registers_64[0] = 1;
        core.registers_64[1] = 1;
        core.registers_64[2] = 12;

        core.run(0).unwrap();
        
        
        //std::io::stdout().flush().unwrap();
        
    }

    #[test]
    fn test_dereff_opcode() {
        let program = vec![27,0,32,0,1,0,0,0,0,0,0,0];
        let memory = vec![0, 0x00,0x00,0xb8,0x41];
        let memory = Arc::new(RwLock::new(memory));
        let mut core = Core::new(memory.clone(), Arc::new(program));

        core.run(0).unwrap();

        println!("{}", core.registers_f32[0]);
        println!("{:?}", 23.0f32.to_ne_bytes());
        println!("{}", f32::from_ne_bytes([0x00,0x00,0xb8,0x41]));
        println!("{}", f32::from_ne_bytes([0x41,0xb8,0x00,0x00]));

        assert_eq!(core.registers_f32[0], f32::from_ne_bytes([0x00,0x00,0xb8,0x41]));
    }


}
