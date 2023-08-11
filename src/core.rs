
use std::arch::x86_64::_CMP_TRUE_UQ;
use std::collections::HashSet;
use std::sync::{Arc,RwLock,TryLockError};
use std::num::Wrapping;
use std::array::from_fn;
use std::sync::atomic::AtomicU64;
use std::thread;
use std::fmt;
use std::sync::mpsc::{Sender,Receiver, TryRecvError};
use std::time::Duration;

use crate::instruction::Opcode;
use crate::{RegisterType,Message,Fault,CoreId,Byte,Pointer,FileDescriptor, Core, SimpleResult, CoreResult, RegCore};


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
/// An Enum that represents the sign flag
pub enum Sign {
    Positive,
    Negative,
}


#[derive(Debug,PartialEq)]
/// An Enum that represents the comparison flag
pub enum Comparison {
    None,
    Equal,
    NotEqual,
    LessThan,
    NotLessThan,
    GreaterThan,
    NotGreaterThan,
    LessThanOrEqual,
    GreaterThanOrEqual,
}

/// The values of this enum are used so that we can implement garbage collection.
/// The use of the enum is so that we don't have to worry about the added runtime cost of
/// protecting the stack when we don't need to.
#[derive(Debug)]
pub struct Stack {
    buffer: *mut Byte,
    size: usize,
    offset: usize,
    stack_pointer: usize,
}

impl Stack {
    pub fn new() -> Stack {
        Stack {
            buffer: std::ptr::null_mut(),
            size: 0,
            offset: 0,
            stack_pointer: 0,
        }
    }

    pub fn set_buffer(&mut self, buffer: *mut Byte, size: usize, offset: usize) {
        self.buffer = buffer;
        self.size = size;
        self.offset = offset;
    }

    pub fn get_offset(&self) -> usize {
        self.offset
    }

    pub fn get_size(&self) -> usize {
        self.size
    }

    pub fn get_stack_pointer(&self) -> usize {
        self.stack_pointer
    }

    pub fn get_bytes(&self, start: usize, size: usize) -> &[Byte] {
        unsafe {
            std::slice::from_raw_parts(self.buffer.offset(start as isize), size)
        }
    }

    pub fn write_bytes(&mut self, start: usize, bytes: &[Byte]) {
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.buffer.offset(start as isize), bytes.len());
        }
    }

    pub fn push(&mut self, bytes: &[Byte]) -> Result<(),Fault> {
        if self.stack_pointer + bytes.len() > self.size {
            return Err(Fault::StackOverflow);
        }

        self.write_bytes(self.stack_pointer, bytes);
        self.stack_pointer += bytes.len();
        Ok(())
    }

    pub fn pop(&mut self, size: usize) -> Result<Vec<Byte>,Fault> {
        if self.stack_pointer < size {
            return Err(Fault::StackUnderflow);
        }

        self.stack_pointer -= size;
        Ok(self.get_bytes(self.stack_pointer, size).to_vec())
    }
    
}

impl fmt::Debug for MachineCore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "Core:\n")?;
        write!(f, "    registers_64: {:?}\n", self.registers_64)?;
        write!(f, "    registers_128: {:?}\n", self.registers_128)?;
        write!(f, "    registers_f32: {:?}\n", self.registers_f32)?;
        write!(f, "    registers_f64: {:?}\n", self.registers_f64)?;
        write!(f, "    comparison_flag: {:?}\n", self.comparison_flag)?;
        write!(f, "    odd_flag: {:?}\n", self.odd_flag)?;
        write!(f, "    zero_flag: {:?}\n", self.zero_flag)?;
        write!(f, "    sign_flag: {:?}\n", self.sign_flag)?;
        write!(f, "    overflow_flag: {:?}\n", self.overflow_flag)?;
        write!(f, "    infinity_flag: {:?}\n", self.infinity_flag)?;
        write!(f, "    nan_flag: {:?}\n", self.nan_flag)?;
        write!(f, "    remainder_64: {:?}\n", self.remainder_64)?;
        write!(f, "    remainder_128: {:?}\n", self.remainder_128)?;
        write!(f, "    program_counter: {:?}\n", self.program_counter)?;
        write!(f, "    stack: {:?}\n", self.stack)?;
        write!(f, "    program: {:?}\n", *self.program)

    }

}

/// A struct that represents a processor core.
pub struct MachineCore {
    /// 64-bit registers
    pub registers_64: [u64; REGISTER_64_COUNT],
    /// 128-bit registers
    pub registers_128: [u128; REGISTER_128_COUNT],
    /// 32-bit floating point registers
    pub registers_f32: [f32; REGISTER_F32_COUNT],
    /// 64-bit floating point registers
    pub registers_f64: [f64; REGISTER_F64_COUNT],
    /* flags */
    /// The comparison flag
    /// This flag is set when a comparison instruction is executed
    pub comparison_flag: Comparison,
    /// The odd flag
    /// This flag is set when a result of an operation is odd
    pub odd_flag: bool,
    /// The zero flag
    /// This flag is set when a result of an operation is zero
    pub zero_flag: bool,
    /// The sign flag
    /// This flag is set when a result of an operation is negative or positive
    /// When working with unsigned numbers, this flag is always set to positive
    pub sign_flag: Sign,
    /// The overflow flag
    /// This flag is set when we wrap around the maximum or minimum value of a number
    pub overflow_flag: bool,
    /// The infinity flag
    /// This flag is set when a result of an operation is infinity
    pub infinity_flag: bool,
    /// The nan flag
    /// This flag is set when a result of an operation is Not a Number
    pub nan_flag: bool,
    /* other */
    /// The remainder of a division operation
    pub remainder_64: usize,
    /// The remainder of a division operation
    pub remainder_128: u128,
    /// The program counter
    pub program_counter: usize,
    /// The program
    pub program: Arc<Vec<Byte>>,
    /// The data segment
    /// This part of memory ir read-only
    pub data_segment: Arc<Vec<Byte>>,
    /// The stack
    /// The stack is always local to the core in order to prevent slowdowns from locking memory
    pub stack: Stack,
    /// The heap
    /// This is a reference to the heap that is shared between all cores
    pub heap: Arc<RwLock<Vec<Byte>>>,
    /// The send channel
    /// This is a channel that is used to send messages to the machine's event loop.
    pub send_channel: Option<Sender<Message>>,
    /// The receive channel
    /// This is a channel that is used to receive messages from the machine's event loop.
    pub recv_channel: Option<Receiver<Message>>,
    /// The threads
    /// This is a set of all the threads that this core has spawned
    pub threads: HashSet<CoreId>,
}

impl Core for MachineCore {
    /// The main loop for the machine
    /// This function will run the program until it is done
    fn run(&mut self, program_counter: usize) -> SimpleResult {
        self.program_counter = program_counter;

        let mut is_done = false;
        while !is_done {
            self.check_messages()?;
            
            is_done = self.execute_instruction()?;
        }
        Ok(())
    }

    /// Function that runs the machine for one instruction
    /// This function is useful for debugging
    fn run_once(&mut self) -> SimpleResult {
        self.execute_instruction()?;
        Ok(())
    }

    /// Used for adding a program to the core
    /// This should not be called after the core has been started
    fn add_program(&mut self, program: Arc<Vec<Byte>>) {
        self.program = program;
    }

    fn add_channels(&mut self, send: Sender<Message>, recv: Receiver<Message>) {
        self.send_channel = Some(send);
        self.recv_channel = Some(recv);
    }

    /// Wrapper for sending a message to the machine's event loop
    fn send_message(&self, message: Message) -> SimpleResult {
        match self.send_channel.as_ref().expect("Send channel was not initialized").send(message) {
            Ok(_) => Ok(()),
            Err(_) => Err(Fault::MachineCrash("Could not send message"))
        }
    }

    /// Wrapper for making a blocking call to receive a message from the machine's event loop
    fn recv_message(&mut self) -> CoreResult<Message> {
        let msg = self.recv_channel.as_ref().expect("Recieve channel was not initialized").recv();
        match msg {
            Ok(message) => {
                match message {
                    Message::CollectGarbage => {
                        self.prepare_for_collection();
                        return self.recv_message();
                    },
                    _ => Ok(message),
                }
                
            },
            Err(_) => Err(Fault::MachineCrash("Could not receive message"))
        }
    }

    /// Function for checking for messages from the machine thread
    fn check_messages(&mut self) -> SimpleResult {
        match self.recv_channel.as_ref().expect("Recieve channel was not initialized").try_recv() {
            Ok(message) => {
                match message {
                    Message::ThreadDone(core_id) => {
                    },
                    Message::CollectGarbage => {
                        self.prepare_for_collection();
                    },

                    _ => unimplemented!(),

                }

            },
            Err(TryRecvError::Disconnected) => {
                return Err(Fault::MachineCrash("Could not check on messages"))
            },
            Err(TryRecvError::Empty) => {

            }
        }
        Ok(())
    }

    #[inline]
    /// Function that checks if the program counter is out of bounds
    fn check_program_counter(&self) -> CoreResult<bool> {

        if self.program_counter >= self.program.len() {
            return Ok(true);
        }
        return Ok(false);
    }


    /// Function that decodes the opcode from the program
    fn decode_opcode(&mut self) -> Opcode {
        let opcode = Opcode::from(self.get_2_bytes());
        self.advance_by_2_bytes();
        return opcode;

    }

    fn get_register_64<'input>(&'input mut self, register: usize) -> CoreResult<&'input mut u64> {
        if register >= REGISTER_64_COUNT {
            return Err(Fault::InvalidRegister(register, RegisterType::Register64));
        }
        Ok(&mut self.registers_64[register])
    }

    fn get_register_128<'input>(&'input mut self, register: usize) -> CoreResult<&'input mut u128> {
        if register >= REGISTER_128_COUNT {
            return Err(Fault::InvalidRegister(register, RegisterType::Register128));
        }
        Ok(&mut self.registers_128[register])
    }

    fn get_register_f32<'input>(&'input mut self, register: usize) -> CoreResult<&'input mut f32> {
        if register >= REGISTER_F32_COUNT {
            return Err(Fault::InvalidRegister(register, RegisterType::RegisterF32));
        }
        Ok(&mut self.registers_f32[register])
    }

    fn get_register_f64<'input>(&'input mut self, register: usize) -> CoreResult<&'input mut f64> {
        if register >= REGISTER_F64_COUNT {
            return Err(Fault::InvalidRegister(register, RegisterType::RegisterF64));
        }
        Ok(&mut self.registers_f64[register])
    }
}

impl RegCore for MachineCore {
    fn add_heap(&mut self, memory: Arc<RwLock<Vec<Byte>>>) {
        self.heap = memory;
    }

    fn add_data_segment(&mut self, data: Arc<Vec<Byte>>) {
        self.data_segment = data;
    }

    fn add_stack(&mut self, stack: *mut Byte, offset: usize, size: usize) {
        self.stack.set_buffer(stack, size, offset);
    }
}
    
impl MachineCore {
    pub fn new() -> MachineCore {
        MachineCore {
            registers_64: [0; 16],
            registers_128: [0; 8],
            registers_f32: [0.0; 8],
            registers_f64: [0.0; 8],
            remainder_64: 0,
            remainder_128: 0,
            comparison_flag: Comparison::None,
            odd_flag: false,
            zero_flag: false,
            sign_flag: Sign::Positive,
            overflow_flag: false,
            infinity_flag: false,
            nan_flag: false,
            program_counter: 0,
            program: Arc::new(Vec::new()),
            data_segment: Arc::new(Vec::new()),
            stack: Stack::new(),
            heap: Arc::new(RwLock::new(Vec::new())),
            send_channel: None,
            recv_channel: None,
            threads: HashSet::new(),
        }
    }

    fn prepare_for_collection(&mut self) -> SimpleResult {
        let mut reg_memory = Vec::new();
        for reg in self.registers_64.iter() {
            reg_memory.extend_from_slice(&reg.to_le_bytes());
        }
        self.push_stack(&reg_memory);
        let message = self.recv_message()?;

        match message {
            Message::Success => {},
            _ => return Err(Fault::MachineCrash("Could not prepare for collection")),
        }

        self.pop_stack(reg_memory.len() * 8);
        
        Ok(())
    }

    fn get_heap(&mut self, address: Pointer, size: u64) -> CoreResult<Vec<u8>> {
        loop {
            match self.heap.try_read() {
                Ok(heap) => {
                    if address + size > heap.len() as u64 {
                        return Err(Fault::SegmentationFault);
                    }
                    return Ok(heap[address as usize..(address + size) as usize].to_vec());
                },
                Err(TryLockError::WouldBlock) => {
                    thread::yield_now();
                },
                Err(TryLockError::Poisoned(_)) => {
                    return Err(Fault::CorruptedMemory);
                }
            }
        }
    }

    /// Function allows us to access the 3 parts of memory with a single address space
    /// size is the number of bytes we want to access
    pub fn get_from_memory(&mut self, address: Pointer, size: u64) -> CoreResult<Vec<u8>> {
        let data_segment_size = self.data_segment.len() as u64;
        let stack_size = self.stack.get_size() as u64;
        let heap_size;

        loop {
            match self.heap.try_read() {
                Ok(heap) => {
                    heap_size = heap.len() as u64;
                    break;
                },
                Err(TryLockError::WouldBlock) => {
                    thread::yield_now();
                },
                Err(TryLockError::Poisoned(_)) => {
                    return Err(Fault::CorruptedMemory);
                }
            }
        }
        
        let mem_size = data_segment_size + stack_size + heap_size;

        if address + size > mem_size {
            return Err(Fault::SegmentationFault);
        }

        if address < data_segment_size {
            return Ok(self.data_segment[address as usize..address as usize + size as usize].to_vec());
        } else if address < data_segment_size + stack_size {
            let real_address = address - data_segment_size;
            return Ok(self.stack.get_bytes(real_address as usize, size as usize).to_vec());
        } else {
            let real_address = address - data_segment_size - stack_size;
            return self.get_heap(real_address, size);
        }
    }

    fn write_to_heap(&mut self, address: Pointer, bytes: &[Byte]) -> SimpleResult {
        loop {
            match self.heap.try_write() {
                Ok(mut heap) => {
                    if address + bytes.len() as u64 > heap.len() as u64 {
                        return Err(Fault::SegmentationFault);
                    }
                    //Verify that this works
                    heap[address as usize..(address + bytes.len() as u64) as usize].copy_from_slice(bytes);
                    return Ok(());
                },
                Err(TryLockError::WouldBlock) => {
                    thread::yield_now();
                },
                Err(TryLockError::Poisoned(_)) => {
                    return Err(Fault::CorruptedMemory);
                }
            }
        }

    }

    pub fn write_to_memory(&mut self, address: Pointer, bytes: &[Byte]) -> SimpleResult {
        let data_segment_size = self.data_segment.len() as u64;
        let stack_size = self.stack.get_size() as u64;
        let heap_size;

        loop {
            match self.heap.try_read() {
                Ok(heap) => {
                    heap_size = heap.len() as u64;
                    break;
                },
                Err(TryLockError::WouldBlock) => {
                    thread::yield_now();
                },
                Err(TryLockError::Poisoned(_)) => {
                    return Err(Fault::CorruptedMemory);
                }
            }
        }
        
        let mem_size = data_segment_size + stack_size + heap_size;

        if address + bytes.len() as u64 > mem_size {
            return Err(Fault::SegmentationFault);
        }

        if address < data_segment_size {
            return Err(Fault::SegmentationFault);
        } else if address < data_segment_size + stack_size {
            let real_address = address - data_segment_size;
            self.stack.write_bytes(real_address as usize, bytes);
        } else {
            let real_address = address - data_segment_size - stack_size;
            let mut heap = self.heap.write().unwrap();
            heap[real_address as usize..real_address as usize + bytes.len()].copy_from_slice(bytes);
        }
        Ok(())
    }

    /// Convenience function for getting the bytes of a string from memory
    fn get_string(&mut self, address: Pointer, size: u64) -> CoreResult<Vec<u8>> {
        self.get_from_memory(address, size)
    }

    /// Convenience function for getting the size of the stack.
    fn stack_len(&self) -> usize {
        self.stack.get_size()
    }



    /// Convenience function for pushing a value to the stack
    fn push_stack(&mut self,value: &[Byte]) -> SimpleResult {
        if value.len() + self.stack.get_size() > self.stack.get_size() {
            return Err(Fault::StackOverflow);
        }

        self.stack.push(value)
    }

    /// Convenience function for popping a value from the stack
    /// size is in bits
    fn pop_stack(&mut self, size: usize) -> CoreResult<Vec<Byte>> {
        let size = size / 8;

        if size > self.stack_len() {
            return Err(Fault::StackUnderflow);
        }

        self.stack.pop(size)
    }

    #[inline]
    /// Function that advances the program counter by a given size
    pub fn advance_by_size(&mut self, size: usize) {
        self.program_counter += size;
    }
    /// Function that advances the program counter by 1 byte or 8 bits
    pub fn advance_by_1_byte(&mut self) {
        self.advance_by_size(1);
    }
    /// Function that advances the program counter by 2 bytes or 16 bits
    pub fn advance_by_2_bytes(&mut self) {
        self.advance_by_size(2);
    }
    /// Function that advances the program counter by 4 bytes or 32 bits
    pub fn advance_by_4_bytes(&mut self) {
        self.advance_by_size(4);
    }
    /// Function that advances the program counter by 8 bytes or 64 bits
    pub fn advance_by_8_bytes(&mut self) {
        self.advance_by_size(8);
    }
    /// Function that advances the program counter by 16 bytes or 128 bits
    pub fn advance_by_16_bytes(&mut self) {
        self.advance_by_size(16);
    }

    /// Function that grabs a byte from the program without advancing the program counter
    pub fn get_1_byte(&mut self) -> Byte {
        let value = self.program[self.program_counter];
        value
    }
    /// Function that grabs 2 bytes from the program without advancing the program counter
    pub fn get_2_bytes(&mut self) -> u16 {
        let value = u16::from_le_bytes([self.program[self.program_counter], self.program[self.program_counter + 1]]);
        value
    }
    /// Function that grabs 4 bytes from the program without advancing the program counter
    pub fn get_4_bytes(&mut self) -> u32 {
        let value = u32::from_le_bytes([self.program[self.program_counter], self.program[self.program_counter + 1], self.program[self.program_counter + 2], self.program[self.program_counter + 3]]);
        value
    }
    /// Function that grabs 8 bytes from the program without advancing the program counter
    pub fn get_8_bytes(&mut self) -> u64 {
        let value = u64::from_le_bytes([self.program[self.program_counter], self.program[self.program_counter + 1], self.program[self.program_counter + 2], self.program[self.program_counter + 3], self.program[self.program_counter + 4], self.program[self.program_counter + 5], self.program[self.program_counter + 6], self.program[self.program_counter + 7]]);
        value
    }
    /// Function that grabs 16 bytes from the program without advancing the program counter
    pub fn get_16_bytes(&mut self) -> u128 {
        let value = u128::from_le_bytes([self.program[self.program_counter], self.program[self.program_counter + 1], self.program[self.program_counter + 2], self.program[self.program_counter + 3], self.program[self.program_counter + 4], self.program[self.program_counter + 5], self.program[self.program_counter + 6], self.program[self.program_counter + 7], self.program[self.program_counter + 8], self.program[self.program_counter + 9], self.program[self.program_counter + 10], self.program[self.program_counter + 11], self.program[self.program_counter + 12], self.program[self.program_counter + 13], self.program[self.program_counter + 14], self.program[self.program_counter + 15]]);
        value
    }

    /// Convenience function for removing a known thread from the machine
    fn remove_thread(&mut self, core_id: CoreId) {
        self.threads.remove(&core_id);
    }

    /// The main logic function for the core.
    /// This function will check to see if there are any messages from the machine thread
    /// If not then we see if the program counter is out of bounds
    /// If not then we decode the opcode and execute the instruction
    pub fn execute_instruction(&mut self) -> CoreResult<bool> {
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
            Add => self.add_opcode()?,
            Sub => self.sub_opcode()?,
            Mul => self.mul_opcode()?,
            Div => self.div_opcode()?,
            Eq => self.eq_opcode()?,
            Neq => self.neq_opcode()?,
            Lt => self.lt_opcode()?,
            Gt => self.gt_opcode()?,
            Leq => self.leq_opcode()?,
            Geq => self.geq_opcode()?,
            AddC => self.addc_opcode()?,
            SubC => self.subc_opcode()?,
            MulC => self.mulc_opcode()?,
            DivC => self.divc_opcode()?,
            EqC => self.eqc_opcode()?,
            NeqC => self.neqc_opcode()?,
            LtC => self.ltc_opcode()?,
            GtC => self.gtc_opcode()?,
            LeqC => self.leqc_opcode()?,
            GeqC => self.geqc_opcode()?,
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
            DeRefRegF => self.derefregf_opcode()?,
            DeRefF => self.dereff_opcode()?,
            MoveF => self.movef_opcode()?,
            SetF => self.setf_opcode()?,
            AddF => self.addf_opcode()?,
            SubF => self.subf_opcode()?,
            MulF => self.mulf_opcode()?,
            DivF => self.divf_opcode()?,
            EqF => self.eqf_opcode()?,
            NeqF => self.neqf_opcode()?,
            LtF => self.ltf_opcode()?,
            GtF => self.gtf_opcode()?,
            LeqF => self.leqf_opcode()?,
            GeqF => self.geqf_opcode()?,
            AddFC => self.addfc_opcode()?,
            SubFC => self.subfc_opcode()?,
            MulFC => self.mulfc_opcode()?,
            DivFC => self.divfc_opcode()?,
            EqFC => self.eqfc_opcode()?,
            NeqFC => self.neqfc_opcode()?,
            LtFC => self.ltfc_opcode()?,
            GtFC => self.gtfc_opcode()?,
            LeqFC => self.leqfc_opcode()?,
            GeqFC => self.geqfc_opcode()?,
            Jump => self.jump_opcode()?,
            JumpEq => self.jumpeq_opcode()?,
            JumpNeq => self.jumpneq_opcode()?,
            JumpLt => self.jumplt_opcode()?,
            JumpGt => self.jumpgt_opcode()?,
            JumpLeq => self.jumpleq_opcode()?,
            JumpGeq => self.jumpgeq_opcode()?,
            JumpZero => self.jumpzero_opcode()?,
            JumpNotZero => self.jumpnotzero_opcode()?,
            JumpNeg => self.jumpneg_opcode()?,
            JumpPos => self.jumppos_opcode()?,
            JumpEven => self.jumpeven_opcode()?,
            JumpOdd => self.jumpodd_opcode()?,
            JumpBack => self.jumpback_opcode()?,
            JumpForward => self.jumpforward_opcode()?,
            JumpInfinity => self.jumpinfinity_opcode()?,
            JumpNotInfinity => self.jumpnotinfinity_opcode()?,
            JumpOverflow => self.jumpoverflow_opcode()?,
            JumpNotOverflow => self.jumpnotoverflow_opcode()?,
            JumpUnderflow => self.jumpunderflow_opcode()?,
            JumpNotUnderflow => self.jumpnotunderflow_opcode()?,
            JumpNaN => self.jumpnan_opcode()?,
            JumpNotNaN => self.jumpnotnan_opcode()?,
            JumpRemainder => self.jumpremainder_opcode()?,
            JumpNotRemainder => self.jumpnotremainder_opcode()?,
            Call => self.call_opcode()?,
            Return => self.return_opcode()?,
            Pop => self.pop_opcode()?,
            Push => self.push_opcode()?,
            PopF => self.popf_opcode()?,
            PushF => self.pushf_opcode()?,
            RegMove => self.regmove_opcode()?,
            RegMoveF => self.regmovef_opcode()?,
            Open => self.open_opcode()?,
            Close => self.close_opcode()?,
            ThreadSpawn => self.threadspawn_opcode()?,
            ThreadReturn => self.threadreturn_opcode()?,
            ThreadJoin => self.threadjoin_opcode()?,
            ThreadDetach => self.threaddetach_opcode()?,
            ForeignCall => self.foreigncall_opcode()?,
            Malloc => self.malloc_opcode()?,
            Free => self.free_opcode()?,
            Realloc => self.realloc_opcode()?,
            Sleep => self.sleep_opcode()?,
            SleepReg => self.sleepreg_opcode()?,
            Random => self.random_opcode()?,
            RandomF => self.randomf_opcode()?,
            Read => self.read_opcode()?,
            ReadByte => self.readbyte_opcode()?,
            

            x => {
                println!("Invalid opcode: {:?} {} at {}", x, u16::from_le_bytes(self.program[self.program_counter - 2..self.program_counter].try_into().unwrap()), self.program_counter - 2);
                return Err(Fault::InvalidOperation)},
            

        }

        Ok(false)
            
    }

    fn set_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as u8 as usize;
        self.advance_by_1_byte();
        let register = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        match size {
            8 => {
                let value = self.get_1_byte();
                check_register64!(register);
                self.registers_64[register] = value as u64;
                self.advance_by_1_byte();
            },
            16 => {
                let value = self.get_2_bytes();
                check_register64!(register);
                self.registers_64[register] = value as u64;
                self.advance_by_2_bytes();
            },
            32 => {
                let value = self.get_4_bytes();
                check_register64!(register);
                self.registers_64[register] = value as u64;
                self.advance_by_4_bytes();
            },
            64 => {
                let value = self.get_8_bytes();
                check_register64!(register);
                self.registers_64[register] = value as u64;
                self.advance_by_8_bytes();
            },
            128 => {
                let value = self.get_16_bytes();
                check_register128!(register);
                self.registers_128[register] = value as u128;
                self.advance_by_16_bytes();
            },
            _ => return Err(Fault::InvalidSize),
        }
        Ok(())
    }

    fn deref_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        let register = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        let address = self.get_8_bytes();
        self.advance_by_8_bytes();
        match size {
            8 => {
                self.registers_64[register] = u8::from_le_bytes(self.get_from_memory(address, 1)?.try_into().unwrap()) as u64;
            },
            16 => {
                self.registers_64[register] = u16::from_le_bytes(self.get_from_memory(address, 2)?.try_into().unwrap()) as u64;
            },
            32 => {
                self.registers_64[register] = u32::from_le_bytes(self.get_from_memory(address, 4)?.try_into().unwrap()) as u64;
            },
            64 => {
                self.registers_64[register] = u64::from_le_bytes(self.get_from_memory(address, 8)?.try_into().unwrap()) as u64;
            },
            128 => {
                self.registers_128[register] = u128::from_le_bytes(self.get_from_memory(address, 16)?.try_into().unwrap()) as u128;
            },
            _ => return Err(Fault::InvalidSize),
        }
        Ok(())
    }

    fn move_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let address_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        check_register64!(address_register as usize);

        let address = self.registers_64[address_register as usize];
        
        match size {
            8 => {
                let register = self.program[self.program_counter] as u8;
                check_register64!(register as usize);
                self.advance_by_1_byte();

                let bytes = (self.registers_64[register as usize] as u8).to_le_bytes();

                self.write_to_memory(address, &bytes)?;
            },
            16 => {
                let register = self.program[self.program_counter] as u8;
                check_register64!(register as usize);
                self.advance_by_1_byte();

                let bytes = (self.registers_64[register as usize] as u16).to_le_bytes();

                self.write_to_memory(address, &bytes)?;
            },
            32 => {
                let register = self.program[self.program_counter] as u8;;
                check_register64!(register as usize);
                self.advance_by_1_byte();

                let bytes = (self.registers_64[register as usize] as u32).to_le_bytes();

                self.write_to_memory(address, &bytes)?;
            },
            64 => {
                let register = self.program[self.program_counter] as u8;
                check_register64!(register as usize);
                self.advance_by_1_byte();

                let bytes = (self.registers_64[register as usize] as u64).to_le_bytes();

                self.write_to_memory(address, &bytes)?;
            },
            128 => {
                let register = self.program[self.program_counter] as u8;
                check_register128!(register as usize);
                self.advance_by_1_byte();

                let bytes = self.registers_128[register as usize].to_le_bytes();

                self.write_to_memory(address, &bytes)?;

            },
            _ => return Err(Fault::InvalidSize),

        }
        Ok(())
    }

    fn derefreg_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register = self.program[self.program_counter] as u8 as usize;
        self.advance_by_1_byte();
        let address_register = self.program[self.program_counter] as u8 as usize;
        check_register64!(address_register);
        self.advance_by_1_byte();
        let offset = i64::from_le_bytes(self.program[self.program_counter..self.program_counter + 8].try_into().unwrap());
        self.advance_by_8_bytes();

        let sign = if offset < 0 { -1 } else { 1 };
        let offset = offset.abs() as u64;
        let address = match sign {
            -1 => self.registers_64[address_register] - offset,
            1 => self.registers_64[address_register] + offset,
            _ => unreachable!(),
        };

        match size {
            8 => {
                check_register64!(register);

                let bytes = self.get_from_memory(address, 1)?;

                self.registers_64[register] = u8::from_le_bytes(bytes.try_into().unwrap()) as u64;
            },
            16 => {
                check_register64!(register);

                let bytes = self.get_from_memory(address, 2)?;

                self.registers_64[register] = u16::from_le_bytes(bytes.try_into().unwrap()) as u64;
            },
            32 => {
                check_register64!(register);

                let bytes = self.get_from_memory(address, 4)?;

                self.registers_64[register] = u32::from_le_bytes(bytes.try_into().unwrap()) as u64;
            },
            64 => {
                check_register64!(register);

                let bytes = self.get_from_memory(address, 8)?;

                self.registers_64[register] = u64::from_le_bytes(bytes.try_into().unwrap()) as u64;
            },
            128 => {
                check_register128!(register);

                let bytes = self.get_from_memory(address, 16)?;

                self.registers_128[register] = u128::from_le_bytes(bytes.try_into().unwrap()) as u128;
            }
            _ => {
                return Err(Fault::InvalidSize);
            }

        }

        Ok(())
        
    }


    fn add_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1, register2);
                let reg1_value = u8::from_le_bytes(self.registers_64[register1].to_le_bytes()[0..1].try_into().unwrap());
                let reg2_value = u8::from_le_bytes(self.registers_64[register2].to_le_bytes()[0..1].try_into().unwrap());

                let new_value = (Wrapping(reg1_value) + Wrapping(reg2_value)).0;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x80 == (Wrapping(new_value) + Wrapping(0x80)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register1] = new_value as u64;
            },
            16 => {
                check_register64!(register1, register2);
                let reg1_value = u16::from_le_bytes(self.registers_64[register1].to_le_bytes()[0..2].try_into().unwrap());
                let reg2_value = u16::from_le_bytes(self.registers_64[register2].to_le_bytes()[0..2].try_into().unwrap());

                let new_value = (Wrapping(reg1_value) + Wrapping(reg2_value)).0;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x8000 == (Wrapping(new_value) + Wrapping(0x8000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register1] = new_value as u64;
            },

            32 => {
                check_register64!(register1, register2);
                let reg1_value = u32::from_le_bytes(self.registers_64[register1].to_le_bytes()[0..4].try_into().unwrap());
                let reg2_value = u32::from_le_bytes(self.registers_64[register2].to_le_bytes()[0..4].try_into().unwrap());

                let new_value = (Wrapping(reg1_value) + Wrapping(reg2_value)).0;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x80000000 == (Wrapping(new_value) + Wrapping(0x80000000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register1] = new_value as u64;
            },
            64 => {
                check_register64!(register1, register2);
                let reg1_value = u64::from_le_bytes(self.registers_64[register1].to_le_bytes().try_into().unwrap());
                let reg2_value = u64::from_le_bytes(self.registers_64[register2].to_le_bytes().try_into().unwrap());

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
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x8000000000000000 == (Wrapping(new_value) + Wrapping(0x8000000000000000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register1] = new_value as u64;
            },
            128 => {
                check_register128!(register1, register2);
                let reg1_value = u128::from_le_bytes(self.registers_128[register1].to_le_bytes().try_into().unwrap());
                let reg2_value = u128::from_le_bytes(self.registers_128[register2].to_le_bytes().try_into().unwrap());

                let new_value = (Wrapping(reg1_value) + Wrapping(reg2_value)).0;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }

                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x80000000000000000000000000000000 == (Wrapping(new_value) + Wrapping(0x80000000000000000000000000000000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_128[register1] = new_value as u128;
            },
            _ => return Err(Fault::InvalidSize),

        }
        Ok(())
    }


    fn sub_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1, register2);
                let reg1_value = u8::from_le_bytes(self.registers_64[register1].to_le_bytes()[0..1].try_into().unwrap());
                let reg2_value = u8::from_le_bytes(self.registers_64[register2].to_le_bytes()[0..1].try_into().unwrap());

                let new_value = (Wrapping(reg1_value) - Wrapping(reg2_value)).0;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x80 == (Wrapping(new_value) + Wrapping(0x80)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register1] = new_value as u64;
            },
            16 => {
                check_register64!(register1, register2);
                let reg1_value = u16::from_le_bytes(self.registers_64[register1].to_le_bytes()[0..2].try_into().unwrap());
                let reg2_value = u16::from_le_bytes(self.registers_64[register2].to_le_bytes()[0..2].try_into().unwrap());

                let new_value = (Wrapping(reg1_value) - Wrapping(reg2_value)).0;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x8000 == (Wrapping(new_value) + Wrapping(0x8000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register1] = new_value as u64;
            },

            32 => {
                check_register64!(register1, register2);
                let reg1_value = u32::from_le_bytes(self.registers_64[register1].to_le_bytes()[0..4].try_into().unwrap());
                let reg2_value = u32::from_le_bytes(self.registers_64[register2].to_le_bytes()[0..4].try_into().unwrap());

                let new_value = (Wrapping(reg1_value) - Wrapping(reg2_value)).0;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x80000000 == (Wrapping(new_value) + Wrapping(0x80000000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register1] = new_value as u64;
            },
            64 => {
                check_register64!(register1, register2);
                let reg1_value = u64::from_le_bytes(self.registers_64[register1].to_le_bytes().try_into().unwrap());
                let reg2_value = u64::from_le_bytes(self.registers_64[register2].to_le_bytes().try_into().unwrap());

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
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x8000000000000000 == (Wrapping(new_value) + Wrapping(0x8000000000000000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register1] = new_value as u64;
            },
            128 => {
                check_register128!(register1, register2);
                let reg1_value = u128::from_le_bytes(self.registers_128[register1].to_le_bytes().try_into().unwrap());
                let reg2_value = u128::from_le_bytes(self.registers_128[register2].to_le_bytes().try_into().unwrap());

                let new_value = (Wrapping(reg1_value) - Wrapping(reg2_value)).0;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }

                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x80000000000000000000000000000000 == (Wrapping(new_value) + Wrapping(0x80000000000000000000000000000000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_128[register1] = new_value as u128;
            },
            _ => return Err(Fault::InvalidSize),

        }
        Ok(())
    }


    fn mul_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1, register2);
                let reg1_value = u8::from_le_bytes(self.registers_64[register1].to_le_bytes()[0..1].try_into().unwrap());
                let reg2_value = u8::from_le_bytes(self.registers_64[register2].to_le_bytes()[0..1].try_into().unwrap());

                let new_value = (Wrapping(reg1_value) * Wrapping(reg2_value)).0;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x80 == (Wrapping(new_value) + Wrapping(0x80)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register1] = new_value as u64;
            },
            16 => {
                check_register64!(register1, register2);
                let reg1_value = u16::from_le_bytes(self.registers_64[register1].to_le_bytes()[0..2].try_into().unwrap());
                let reg2_value = u16::from_le_bytes(self.registers_64[register2].to_le_bytes()[0..2].try_into().unwrap());

                let new_value = (Wrapping(reg1_value) * Wrapping(reg2_value)).0;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x8000 == (Wrapping(new_value) + Wrapping(0x8000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register1] = new_value as u64;
            },

            32 => {
                check_register64!(register1, register2);
                let reg1_value = u32::from_le_bytes(self.registers_64[register1].to_le_bytes()[0..4].try_into().unwrap());
                let reg2_value = u32::from_le_bytes(self.registers_64[register2].to_le_bytes()[0..4].try_into().unwrap());

                let new_value = (Wrapping(reg1_value) * Wrapping(reg2_value)).0;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x80000000 == (Wrapping(new_value) + Wrapping(0x80000000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register1] = new_value as u64;
            },
            64 => {
                check_register64!(register1, register2);
                let reg1_value = u64::from_le_bytes(self.registers_64[register1].to_le_bytes().try_into().unwrap());
                let reg2_value = u64::from_le_bytes(self.registers_64[register2].to_le_bytes().try_into().unwrap());

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
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x8000000000000000 == (Wrapping(new_value) + Wrapping(0x8000000000000000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register1] = new_value as u64;
            },
            128 => {
                check_register128!(register1, register2);
                let reg1_value = u128::from_le_bytes(self.registers_128[register1].to_le_bytes().try_into().unwrap());
                let reg2_value = u128::from_le_bytes(self.registers_128[register2].to_le_bytes().try_into().unwrap());

                let new_value = (Wrapping(reg1_value) * Wrapping(reg2_value)).0;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }

                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x80000000000000000000000000000000 == (Wrapping(new_value) + Wrapping(0x80000000000000000000000000000000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_128[register1] = new_value as u128;
            },
            _ => return Err(Fault::InvalidSize),

        }
        Ok(())
    }


    fn div_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1, register2);
                let reg1_value = u8::from_le_bytes(self.registers_64[register1].to_le_bytes()[0..1].try_into().unwrap());
                let reg2_value = u8::from_le_bytes(self.registers_64[register2].to_le_bytes()[0..1].try_into().unwrap());
                if reg2_value == 0 {
                    return Err(Fault::DivideByZero);
                }

                let remainder = reg1_value % reg2_value;

                let new_value = (Wrapping(reg1_value) / Wrapping(reg2_value)).0;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x80 == (Wrapping(new_value) + Wrapping(0x80)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register1] = new_value as u64;
            },
            16 => {
                check_register64!(register1, register2);
                let reg1_value = u16::from_le_bytes(self.registers_64[register1].to_le_bytes()[0..2].try_into().unwrap());
                let reg2_value = u16::from_le_bytes(self.registers_64[register2].to_le_bytes()[0..2].try_into().unwrap());
                if reg2_value == 0 {
                    return Err(Fault::DivideByZero);
                }

                let remainder = reg1_value % reg2_value;

                let new_value = (Wrapping(reg1_value) / Wrapping(reg2_value)).0;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x8000 == (Wrapping(new_value) + Wrapping(0x8000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register1] = new_value as u64;
            },

            32 => {
                check_register64!(register1, register2);
                let reg1_value = u32::from_le_bytes(self.registers_64[register1].to_le_bytes()[0..4].try_into().unwrap());
                let reg2_value = u32::from_le_bytes(self.registers_64[register2].to_le_bytes()[0..4].try_into().unwrap());

                if reg2_value == 0 {
                    return Err(Fault::DivideByZero);
                }

                let remainder = reg1_value % reg2_value;

                let new_value = (Wrapping(reg1_value) / Wrapping(reg2_value)).0;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x80000000 == (Wrapping(new_value) + Wrapping(0x80000000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register1] = new_value as u64;
            },
            64 => {
                check_register64!(register1, register2);
                let reg1_value = u64::from_le_bytes(self.registers_64[register1].to_le_bytes().try_into().unwrap());
                let reg2_value = u64::from_le_bytes(self.registers_64[register2].to_le_bytes().try_into().unwrap());

                if reg2_value == 0 {
                    return Err(Fault::DivideByZero);
                }

                let remainder = reg1_value % reg2_value;

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
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x8000000000000000 == (Wrapping(new_value) + Wrapping(0x8000000000000000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register1] = new_value as u64;
            },
            128 => {
                check_register128!(register1, register2);
                let reg1_value = u128::from_le_bytes(self.registers_128[register1].to_le_bytes().try_into().unwrap());
                let reg2_value = u128::from_le_bytes(self.registers_128[register2].to_le_bytes().try_into().unwrap());

                if reg2_value == 0 {
                    return Err(Fault::DivideByZero);
                }

                let remainder = reg1_value % reg2_value;

                let new_value = (Wrapping(reg1_value) / Wrapping(reg2_value)).0;

                if new_value < reg1_value {
                    self.overflow_flag = true;
                }

                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x80000000000000000000000000000000 == (Wrapping(new_value) + Wrapping(0x80000000000000000000000000000000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_128[register1] = new_value as u128;
            },
            _ => return Err(Fault::InvalidSize),

        }
        Ok(())
    }

    fn neq_opcode(&mut self) -> Result<(),Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1, register2);

                let reg1_value = u8::from_le_bytes(self.registers_64[register1].to_le_bytes()[0..1].try_into().unwrap());
                let reg2_value = u8::from_le_bytes(self.registers_64[register2].to_le_bytes()[0..1].try_into().unwrap());

                if reg1_value != reg2_value {
                    self.comparison_flag = Comparison::NotEqual;
                }
                else {
                    self.comparison_flag = Comparison::Equal;
                }
            },
            16 => {
                check_register64!(register1, register2);

                let reg1_value = u16::from_le_bytes(self.registers_64[register1].to_le_bytes()[0..2].try_into().unwrap());
                let reg2_value = u16::from_le_bytes(self.registers_64[register2].to_le_bytes()[0..2].try_into().unwrap());

                if reg1_value != reg2_value {
                    self.comparison_flag = Comparison::NotEqual;
                }
                else {
                    self.comparison_flag = Comparison::Equal;
                }
            },
            32 => {
                check_register64!(register1, register2);

                let reg1_value = u32::from_le_bytes(self.registers_64[register1].to_le_bytes()[0..4].try_into().unwrap());
                let reg2_value = u32::from_le_bytes(self.registers_64[register2].to_le_bytes()[0..4].try_into().unwrap());

                if reg1_value != reg2_value {
                    self.comparison_flag = Comparison::NotEqual;
                }
                else {
                    self.comparison_flag = Comparison::Equal;
                }
            },
            64 => {
                check_register64!(register1, register2);

                let reg1_value = u64::from_le_bytes(self.registers_64[register1].to_le_bytes().try_into().unwrap());
                let reg2_value = u64::from_le_bytes(self.registers_64[register2].to_le_bytes().try_into().unwrap());
                

                if reg1_value != reg2_value {
                    self.comparison_flag = Comparison::NotEqual;
                }
                else {
                    self.comparison_flag = Comparison::Equal;
                }
            },
            128 => {
                check_register128!(register1, register2);

                let reg1_value = u128::from_le_bytes(self.registers_128[register1].to_le_bytes());
                let reg2_value = u128::from_le_bytes(self.registers_128[register2].to_le_bytes());

                if reg1_value != reg2_value {
                    self.comparison_flag = Comparison::NotEqual;
                }
                else {
                    self.comparison_flag = Comparison::Equal;
                }
            },

            _ => return Err(Fault::InvalidSize),
            
        }
        
        Ok(())
    }
    
    fn eq_opcode(&mut self) -> Result<(),Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1, register2);

                let reg1_value = u8::from_le_bytes(self.registers_64[register1].to_le_bytes()[0..1].try_into().unwrap());
                let reg2_value = u8::from_le_bytes(self.registers_64[register2].to_le_bytes()[0..1].try_into().unwrap());

                if reg1_value == reg2_value {
                    self.comparison_flag = Comparison::Equal;
                }
                else {
                    self.comparison_flag = Comparison::NotEqual;
                }
            },
            16 => {
                check_register64!(register1, register2);

                let reg1_value = u16::from_le_bytes(self.registers_64[register1].to_le_bytes()[0..2].try_into().unwrap());
                let reg2_value = u16::from_le_bytes(self.registers_64[register2].to_le_bytes()[0..2].try_into().unwrap());

                if reg1_value == reg2_value {
                    self.comparison_flag = Comparison::Equal;
                }
                else {
                    self.comparison_flag = Comparison::NotEqual;
                }
            },
            32 => {
                check_register64!(register1, register2);

                let reg1_value = u32::from_le_bytes(self.registers_64[register1].to_le_bytes()[0..4].try_into().unwrap());
                let reg2_value = u32::from_le_bytes(self.registers_64[register2].to_le_bytes()[0..4].try_into().unwrap());

                if reg1_value == reg2_value {
                    self.comparison_flag = Comparison::Equal;
                }
                else {
                    self.comparison_flag = Comparison::NotEqual;
                }
            },
            64 => {
                check_register64!(register1, register2);

                let reg1_value = u64::from_le_bytes(self.registers_64[register1].to_le_bytes().try_into().unwrap());
                let reg2_value = u64::from_le_bytes(self.registers_64[register2].to_le_bytes().try_into().unwrap());

                if reg1_value == reg2_value {
                    self.comparison_flag = Comparison::Equal;
                }
                else {
                    self.comparison_flag = Comparison::NotEqual;
                }
            },
            128 => {
                check_register128!(register1, register2);

                let reg1_value = u128::from_le_bytes(self.registers_128[register1].to_le_bytes());
                let reg2_value = u128::from_le_bytes(self.registers_128[register2].to_le_bytes());

                if reg1_value == reg2_value {
                    self.comparison_flag = Comparison::Equal;
                }
                else {
                    self.comparison_flag = Comparison::NotEqual;
                }
            },

            _ => return Err(Fault::InvalidSize),
            
        }
        
        Ok(())
    }

    fn lt_opcode(&mut self) -> Result<(),Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1, register2);

                let reg1_value = u8::from_le_bytes(self.registers_64[register1].to_le_bytes()[0..1].try_into().unwrap());
                let reg2_value = u8::from_le_bytes(self.registers_64[register2].to_le_bytes()[0..1].try_into().unwrap());

                if reg1_value < reg2_value {
                    self.comparison_flag = Comparison::LessThan;
                }
                else {
                    self.comparison_flag = Comparison::GreaterThanOrEqual;
                }
            },
            16 => {
                check_register64!(register1, register2);

                let reg1_value = u16::from_le_bytes(self.registers_64[register1].to_le_bytes()[0..2].try_into().unwrap());
                let reg2_value = u16::from_le_bytes(self.registers_64[register2].to_le_bytes()[0..2].try_into().unwrap());

                if reg1_value < reg2_value {
                    self.comparison_flag = Comparison::LessThan;
                }
                else {
                    self.comparison_flag = Comparison::GreaterThanOrEqual;
                }
            },
            32 => {
                check_register64!(register1, register2);

                let reg1_value = u32::from_le_bytes(self.registers_64[register1].to_le_bytes()[0..4].try_into().unwrap());
                let reg2_value = u32::from_le_bytes(self.registers_64[register2].to_le_bytes()[0..4].try_into().unwrap());

                if reg1_value < reg2_value {
                    self.comparison_flag = Comparison::LessThan;
                }
                else {
                    self.comparison_flag = Comparison::GreaterThanOrEqual;
                }
            },
            64 => {
                check_register64!(register1, register2);

                let reg1_value = u64::from_le_bytes(self.registers_64[register1].to_le_bytes().try_into().unwrap());
                let reg2_value = u64::from_le_bytes(self.registers_64[register2].to_le_bytes().try_into().unwrap());

                if reg1_value < reg2_value {
                    self.comparison_flag = Comparison::LessThan;
                }
                else {
                    self.comparison_flag = Comparison::GreaterThanOrEqual;
                }
            },
            128 => {
                check_register128!(register1, register2);

                let reg1_value = u128::from_le_bytes(self.registers_128[register1].to_le_bytes());
                let reg2_value = u128::from_le_bytes(self.registers_128[register2].to_le_bytes());

                if reg1_value < reg2_value {
                    self.comparison_flag = Comparison::LessThan;
                }
                else {
                    self.comparison_flag = Comparison::GreaterThanOrEqual;
                }
            },
            _ => return Err(Fault::InvalidSize),

        }

        Ok(())
    }

    fn gt_opcode(&mut self) -> Result<(),Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1, register2);

                let reg1_value = u8::from_le_bytes(self.registers_64[register1].to_le_bytes()[0..1].try_into().unwrap());
                let reg2_value = u8::from_le_bytes(self.registers_64[register2].to_le_bytes()[0..1].try_into().unwrap());

                if reg1_value > reg2_value {
                    self.comparison_flag = Comparison::GreaterThan;
                }
                else {
                    self.comparison_flag = Comparison::LessThanOrEqual;
                }
            },
            16 => {
                check_register64!(register1, register2);

                let reg1_value = u16::from_le_bytes(self.registers_64[register1].to_le_bytes()[0..2].try_into().unwrap());
                let reg2_value = u16::from_le_bytes(self.registers_64[register2].to_le_bytes()[0..2].try_into().unwrap());

                if reg1_value > reg2_value {
                    self.comparison_flag = Comparison::GreaterThan;
                }
                else {
                    self.comparison_flag = Comparison::LessThanOrEqual;
                }
            },
            32 => {
                check_register64!(register1, register2);

                let reg1_value = u32::from_le_bytes(self.registers_64[register1].to_le_bytes()[0..4].try_into().unwrap());
                let reg2_value = u32::from_le_bytes(self.registers_64[register2].to_le_bytes()[0..4].try_into().unwrap());

                if reg1_value > reg2_value {
                    self.comparison_flag = Comparison::GreaterThan;
                }
                else {
                    self.comparison_flag = Comparison::LessThanOrEqual;
                }
            },
            64 => {
                check_register64!(register1, register2);

                let reg1_value = u64::from_le_bytes(self.registers_64[register1].to_le_bytes().try_into().unwrap());
                let reg2_value = u64::from_le_bytes(self.registers_64[register2].to_le_bytes().try_into().unwrap());

                if reg1_value > reg2_value {
                    self.comparison_flag = Comparison::GreaterThan;
                }
                else {
                    self.comparison_flag = Comparison::LessThanOrEqual;
                }
            },
            128 => {
                check_register128!(register1, register2);

                let reg1_value = u128::from_le_bytes(self.registers_128[register1].to_le_bytes());
                let reg2_value = u128::from_le_bytes(self.registers_128[register2].to_le_bytes());

                if reg1_value > reg2_value {
                    self.comparison_flag = Comparison::GreaterThan;
                }
                else {
                    self.comparison_flag = Comparison::LessThanOrEqual;
                }
            },
            _ => return Err(Fault::InvalidSize),

        }

        Ok(())
    }
        
    fn leq_opcode(&mut self) -> Result<(),Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1, register2);

                let reg1_value = u8::from_le_bytes(self.registers_64[register1].to_le_bytes()[0..1].try_into().unwrap());
                let reg2_value = u8::from_le_bytes(self.registers_64[register2].to_le_bytes()[0..1].try_into().unwrap());
                
                if reg1_value <= reg2_value {
                    self.comparison_flag = Comparison::LessThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::GreaterThan;
                }
            },
            16 => {
                check_register64!(register1, register2);

                let reg1_value = u16::from_le_bytes(self.registers_64[register1].to_le_bytes()[0..2].try_into().unwrap());
                let reg2_value = u16::from_le_bytes(self.registers_64[register2].to_le_bytes()[0..2].try_into().unwrap());

                if reg1_value <= reg2_value {
                    self.comparison_flag = Comparison::LessThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::GreaterThan;
                }
            },
            32 => {
                check_register64!(register1, register2);

                let reg1_value = u32::from_le_bytes(self.registers_64[register1].to_le_bytes()[0..4].try_into().unwrap());
                let reg2_value = u32::from_le_bytes(self.registers_64[register2].to_le_bytes()[0..4].try_into().unwrap());

                if reg1_value <= reg2_value {
                    self.comparison_flag = Comparison::LessThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::GreaterThan;
                }
            },
            64 => {
                check_register64!(register1, register2);

                let reg1_value = u64::from_le_bytes(self.registers_64[register1].to_le_bytes().try_into().unwrap());
                let reg2_value = u64::from_le_bytes(self.registers_64[register2].to_le_bytes().try_into().unwrap());

                if reg1_value <= reg2_value {
                    self.comparison_flag = Comparison::LessThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::GreaterThan;
                }
            },
            128 => {
                check_register128!(register1, register2);

                let reg1_value = u128::from_le_bytes(self.registers_128[register1].to_le_bytes());
                let reg2_value = u128::from_le_bytes(self.registers_128[register2].to_le_bytes());

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
        
    fn geq_opcode(&mut self) -> Result<(),Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();
        let register2 = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register1, register2);

                let reg1_value = u8::from_le_bytes(self.registers_64[register1].to_le_bytes()[0..1].try_into().unwrap());
                let reg2_value = u8::from_le_bytes(self.registers_64[register2].to_le_bytes()[0..1].try_into().unwrap());

                if reg1_value >= reg2_value {
                    self.comparison_flag = Comparison::GreaterThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::LessThan;
                }
            },
            16 => {
                check_register64!(register1, register2);

                let reg1_value = u16::from_le_bytes(self.registers_64[register1].to_le_bytes()[0..2].try_into().unwrap());
                let reg2_value = u16::from_le_bytes(self.registers_64[register2].to_le_bytes()[0..2].try_into().unwrap());

                if reg1_value >= reg2_value {
                    self.comparison_flag = Comparison::GreaterThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::LessThan;
                }
            },
            32 => {
                check_register64!(register1, register2);

                let reg1_value = u32::from_le_bytes(self.registers_64[register1].to_le_bytes()[0..4].try_into().unwrap());
                let reg2_value = u32::from_le_bytes(self.registers_64[register2].to_le_bytes()[0..4].try_into().unwrap());

                if reg1_value >= reg2_value {
                    self.comparison_flag = Comparison::GreaterThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::LessThan;
                }
            },
            64 => {
                check_register64!(register1, register2);

                let reg1_value = u64::from_le_bytes(self.registers_64[register1].to_le_bytes().try_into().unwrap());
                let reg2_value = u64::from_le_bytes(self.registers_64[register2].to_le_bytes().try_into().unwrap());

                if reg1_value >= reg2_value {
                    self.comparison_flag = Comparison::GreaterThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::LessThan;
                }
            },
            128 => {
                check_register128!(register1, register2);

                let reg1_value = u128::from_le_bytes(self.registers_128[register1].to_le_bytes());
                let reg2_value = u128::from_le_bytes(self.registers_128[register2].to_le_bytes());

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


    fn addc_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register);
                let reg_value = self.registers_64[register] as u8;

                let constant = self.get_1_byte() as u8;
                self.advance_by_1_byte();

                let new_value = (Wrapping(reg_value) + Wrapping(constant)).0;

                if new_value < reg_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x80 == (Wrapping(new_value) + Wrapping(0x80)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register] = new_value as u64;
            },
            16 => {
                check_register64!(register);
                let reg_value = self.registers_64[register] as u16;

                let constant = self.get_2_bytes() as u16;
                self.advance_by_2_bytes();

                let new_value = (Wrapping(reg_value) + Wrapping(constant)).0;

                if new_value < reg_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x8000 == (Wrapping(new_value) + Wrapping(0x8000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register] = new_value as u64;
            },
            32 => {
                check_register64!(register);
                let reg_value = self.registers_64[register] as u32;

                let constant = self.get_4_bytes() as u32;
                self.advance_by_4_bytes();

                let new_value = (Wrapping(reg_value) + Wrapping(constant)).0;

                if new_value < reg_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x80000000 == (Wrapping(new_value) + Wrapping(0x80000000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register] = new_value as u64;
            },
            64 => {
                check_register64!(register);
                let reg_value = self.registers_64[register];

                let constant = self.get_8_bytes();
                self.advance_by_8_bytes();

                let new_value = (Wrapping(reg_value) + Wrapping(constant)).0;

                if new_value < reg_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x8000000000000000 == (Wrapping(new_value) + Wrapping(0x8000000000000000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register] = new_value;
            },
            128 => {
                check_register128!(register);
                let reg_value = self.registers_128[register];

                let constant = self.get_16_bytes();
                self.advance_by_16_bytes();

                let new_value = (Wrapping(reg_value) + Wrapping(constant)).0;

                if new_value < reg_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x80000000000000000000000000000000 == (Wrapping(new_value) + Wrapping(0x80000000000000000000000000000000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_128[register] = new_value;
            },
            _ => return Err(Fault::InvalidSize),

        }
        Ok(())
    }


    fn subc_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register);
                let reg_value = self.registers_64[register] as u8;

                let constant = self.get_1_byte() as u8;
                self.advance_by_1_byte();

                let new_value = (Wrapping(reg_value) - Wrapping(constant)).0;

                if new_value < reg_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x80 == (Wrapping(new_value) + Wrapping(0x80)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register] = new_value as u64;
            },
            16 => {
                check_register64!(register);
                let reg_value = self.registers_64[register] as u16;

                let constant = self.get_2_bytes() as u16;
                self.advance_by_2_bytes();

                let new_value = (Wrapping(reg_value) - Wrapping(constant)).0;

                if new_value < reg_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x8000 == (Wrapping(new_value) + Wrapping(0x8000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register] = new_value as u64;
            },
            32 => {
                check_register64!(register);
                let reg_value = self.registers_64[register] as u32;

                let constant = self.get_4_bytes() as u32;
                self.advance_by_4_bytes();

                let new_value = (Wrapping(reg_value) - Wrapping(constant)).0;

                if new_value < reg_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x80000000 == (Wrapping(new_value) + Wrapping(0x80000000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register] = new_value as u64;
            },
            64 => {
                check_register64!(register);
                let reg_value = self.registers_64[register];

                let constant = self.get_8_bytes();
                self.advance_by_8_bytes();

                let new_value = (Wrapping(reg_value) - Wrapping(constant)).0;

                if new_value < reg_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x8000000000000000 == (Wrapping(new_value) + Wrapping(0x8000000000000000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register] = new_value;
            },
            128 => {
                check_register128!(register);
                let reg_value = self.registers_128[register];

                let constant = self.get_16_bytes();
                self.advance_by_16_bytes();

                let new_value = (Wrapping(reg_value) - Wrapping(constant)).0;

                if new_value < reg_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x80000000000000000000000000000000 == (Wrapping(new_value) + Wrapping(0x80000000000000000000000000000000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_128[register] = new_value;
            },
            _ => return Err(Fault::InvalidSize),

        }
        Ok(())
    }


    fn mulc_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register);
                let reg_value = self.registers_64[register] as u8;

                let constant = self.get_1_byte() as u8;
                self.advance_by_1_byte();

                let new_value = (Wrapping(reg_value) * Wrapping(constant)).0;

                if new_value < reg_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x80 == (Wrapping(new_value) + Wrapping(0x80)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register] = new_value as u64;
            },
            16 => {
                check_register64!(register);
                let reg_value = self.registers_64[register] as u16;

                let constant = self.get_2_bytes() as u16;
                self.advance_by_2_bytes();

                let new_value = (Wrapping(reg_value) * Wrapping(constant)).0;

                if new_value < reg_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x8000 == (Wrapping(new_value) + Wrapping(0x8000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register] = new_value as u64;
            },
            32 => {
                check_register64!(register);
                let reg_value = self.registers_64[register] as u32;

                let constant = self.get_4_bytes() as u32;
                self.advance_by_4_bytes();

                let new_value = (Wrapping(reg_value) * Wrapping(constant)).0;

                if new_value < reg_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x80000000 == (Wrapping(new_value) + Wrapping(0x80000000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register] = new_value as u64;
            },
            64 => {
                check_register64!(register);
                let reg_value = self.registers_64[register];

                let constant = self.get_8_bytes();
                self.advance_by_8_bytes();

                let new_value = (Wrapping(reg_value) * Wrapping(constant)).0;

                if new_value < reg_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x8000000000000000 == (Wrapping(new_value) + Wrapping(0x8000000000000000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register] = new_value;
            },
            128 => {
                check_register128!(register);
                let reg_value = self.registers_128[register];

                let constant = self.get_16_bytes();
                self.advance_by_16_bytes();

                let new_value = (Wrapping(reg_value) * Wrapping(constant)).0;

                if new_value < reg_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x80000000000000000000000000000000 == (Wrapping(new_value) + Wrapping(0x80000000000000000000000000000000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_128[register] = new_value;
            },
            _ => return Err(Fault::InvalidSize),

        }
        Ok(())
    }


    fn divu_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register);
                let reg_value = self.registers_64[register] as u8;

                let constant = self.get_1_byte() as u8;
                self.advance_by_1_byte();

                if constant == 0 {
                    return Err(Fault::DivideByZero);
                }

                let remainder_64 = reg_value % constant;

                let new_value = (Wrapping(reg_value) / Wrapping(constant)).0;

                if new_value < reg_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x80 == (Wrapping(new_value) + Wrapping(0x80)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register] = new_value as u64;
            },
            16 => {
                check_register64!(register);
                let reg_value = self.registers_64[register] as u16;

                let constant = self.get_2_bytes() as u16;
                self.advance_by_2_bytes();

                if constant == 0 {
                    return Err(Fault::DivideByZero);
                }

                let remainder_64 = reg_value % constant;

                let new_value = (Wrapping(reg_value) / Wrapping(constant)).0;

                if new_value < reg_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x8000 == (Wrapping(new_value) + Wrapping(0x8000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register] = new_value as u64;
            },
            32 => {
                check_register64!(register);
                let reg_value = self.registers_64[register] as u32;

                let constant = self.get_4_bytes() as u32;
                self.advance_by_4_bytes();

                if constant == 0 {
                    return Err(Fault::DivideByZero);
                }

                let remainder_64 = reg_value % constant;

                let new_value = (Wrapping(reg_value) / Wrapping(constant)).0;

                if new_value < reg_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x80000000 == (Wrapping(new_value) + Wrapping(0x80000000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register] = new_value as u64;
            },
            64 => {
                check_register64!(register);
                let reg_value = self.registers_64[register];

                let constant = self.get_8_bytes();
                self.advance_by_8_bytes();

                if constant == 0 {
                    return Err(Fault::DivideByZero);
                }

                let remainder_64 = reg_value % constant;

                let new_value = (Wrapping(reg_value) / Wrapping(constant)).0;

                if new_value < reg_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x8000000000000000 == (Wrapping(new_value) + Wrapping(0x8000000000000000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_64[register] = new_value;
            },
            128 => {
                check_register128!(register);
                let reg_value = self.registers_128[register];

                let constant = self.get_16_bytes();
                self.advance_by_16_bytes();

                if constant == 0 {
                    return Err(Fault::DivideByZero);
                }

                let remainder_128 = reg_value % constant;

                let new_value = (Wrapping(reg_value) / Wrapping(constant)).0;

                if new_value < reg_value {
                    self.overflow_flag = true;
                }
                if new_value == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x80000000000000000000000000000000 == (Wrapping(new_value) + Wrapping(0x80000000000000000000000000000000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }

                self.registers_128[register] = new_value;
            },
            _ => return Err(Fault::InvalidSize),

        }
        Ok(())
    }

    fn neqc_opcode(&mut self) -> Result<(),Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register);

                let reg_value = self.registers_64[register] as u8;
                let constant = self.get_1_byte() as u8;
                self.advance_by_1_byte();

                if reg_value != constant {
                    self.comparison_flag = Comparison::NotEqual;
                }
                else {
                    self.comparison_flag = Comparison::Equal;
                }
            },
            16 => {
                check_register64!(register);

                let reg_value = self.registers_64[register] as u16;
                let constant = self.get_2_bytes() as u16;
                self.advance_by_2_bytes();

                if reg_value != constant {
                    self.comparison_flag = Comparison::NotEqual;
                }
                else {
                    self.comparison_flag = Comparison::Equal;
                }
            },
            32 => {
                check_register64!(register);

                let reg_value = self.registers_64[register] as u32;
                let constant = self.get_4_bytes() as u32;
                self.advance_by_4_bytes();

                if reg_value != constant {
                    self.comparison_flag = Comparison::NotEqual;
                }
                else {
                    self.comparison_flag = Comparison::Equal;
                }
            },
            64 => {
                check_register64!(register);

                let reg_value = self.registers_64[register] as u64;
                let constant = self.get_8_bytes() as u64;
                self.advance_by_8_bytes();

                if reg_value != constant {
                    self.comparison_flag = Comparison::NotEqual;
                }
                else {
                    self.comparison_flag = Comparison::Equal;
                }
            },
            128 => {
                check_register128!(register);

                let reg_value = self.registers_128[register] as u128;
                let constant = self.get_16_bytes() as u128;
                self.advance_by_16_bytes();

                if reg_value != constant {
                    self.comparison_flag = Comparison::NotEqual;
                }
                else {
                    self.comparison_flag = Comparison::Equal;
                }
            },

            _ => return Err(Fault::InvalidSize),
            
        }
        
        Ok(())
    }
    
    fn eqc_opcode(&mut self) -> Result<(),Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register);

                let reg_value = self.registers_64[register] as u8;
                let constant = self.get_1_byte() as u8;
                self.advance_by_1_byte();

                if reg_value == constant {
                    self.comparison_flag = Comparison::Equal;
                }
                else {
                    self.comparison_flag = Comparison::NotEqual;
                }
            },
            16 => {
                check_register64!(register);

                let reg_value = self.registers_64[register] as u16;
                let constant = self.get_2_bytes() as u16;
                self.advance_by_2_bytes();

                if reg_value == constant {
                    self.comparison_flag = Comparison::Equal;
                }
                else {
                    self.comparison_flag = Comparison::NotEqual;
                }
            },
            32 => {
                check_register64!(register);

                let reg_value = self.registers_64[register] as u32;
                let constant = self.get_4_bytes() as u32;
                self.advance_by_4_bytes();

                if reg_value == constant {
                    self.comparison_flag = Comparison::Equal;
                }
                else {
                    self.comparison_flag = Comparison::NotEqual;
                }
            },
            64 => {
                check_register64!(register);

                let reg_value = self.registers_64[register] as u64;
                let constant = self.get_8_bytes() as u64;
                self.advance_by_8_bytes();

                if reg_value == constant {
                    self.comparison_flag = Comparison::Equal;
                }
                else {
                    self.comparison_flag = Comparison::NotEqual;
                }
            },
            128 => {
                check_register128!(register);

                let reg_value = self.registers_128[register] as u128;
                let constant = self.get_16_bytes() as u128;
                self.advance_by_16_bytes();

                if reg_value == constant {
                    self.comparison_flag = Comparison::Equal;
                }
                else {
                    self.comparison_flag = Comparison::Equal;
                }
            },

            _ => return Err(Fault::InvalidSize),
            
        }
        
        Ok(())
    }

    fn ltc_opcode(&mut self) -> Result<(),Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register);

                let reg_value = self.registers_64[register] as u8;
                let constant = self.get_1_byte() as u8;
                self.advance_by_1_byte();

                if reg_value < constant {
                    self.comparison_flag = Comparison::LessThan;
                }
                else {
                    self.comparison_flag = Comparison::GreaterThanOrEqual;
                }
            },
            16 => {
                check_register64!(register);

                let reg_value = self.registers_64[register] as u16;
                let constant = self.get_2_bytes() as u16;
                self.advance_by_2_bytes();

                if reg_value < constant {
                    self.comparison_flag = Comparison::LessThan;
                }
                else {
                    self.comparison_flag = Comparison::GreaterThanOrEqual;
                }
            },
            32 => {
                check_register64!(register);

                let reg_value = self.registers_64[register] as u32;
                let constant = self.get_4_bytes() as u32;
                self.advance_by_4_bytes();

                if reg_value < constant {
                    self.comparison_flag = Comparison::LessThan;
                }
                else {
                    self.comparison_flag = Comparison::GreaterThanOrEqual;
                }
            },
            64 => {
                check_register64!(register);

                let reg_value = self.registers_64[register] as u64;
                let constant = self.get_8_bytes() as u64;
                self.advance_by_8_bytes();

                if reg_value < constant {
                    self.comparison_flag = Comparison::LessThan;
                }
                else {
                    self.comparison_flag = Comparison::GreaterThanOrEqual;
                }
            },
            128 => {
                check_register128!(register);

                let reg_value = self.registers_128[register] as u128;
                let constant = self.get_16_bytes() as u128;
                self.advance_by_16_bytes();

                if reg_value < constant {
                    self.comparison_flag = Comparison::LessThan;
                }
                else {
                    self.comparison_flag = Comparison::GreaterThanOrEqual;
                }
            },
            _ => return Err(Fault::InvalidSize),

        }

        Ok(())
    }

    fn gtc_opcode(&mut self) -> Result<(),Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register);

                let reg_value = self.registers_64[register] as u8;
                let constant = self.get_1_byte() as u8;
                self.advance_by_1_byte();

                if reg_value > constant {
                    self.comparison_flag = Comparison::GreaterThan;
                }
                else {
                    self.comparison_flag = Comparison::LessThanOrEqual;
                }
            },
            16 => {
                check_register64!(register);

                let reg_value = self.registers_64[register] as u16;
                let constant = self.get_2_bytes() as u16;
                self.advance_by_2_bytes();

                if reg_value > constant {
                    self.comparison_flag = Comparison::GreaterThan;
                }
                else {
                    self.comparison_flag = Comparison::LessThanOrEqual;
                }
            },
            32 => {
                check_register64!(register);

                let reg_value = self.registers_64[register] as u32;
                let constant = self.get_4_bytes() as u32;
                self.advance_by_4_bytes();

                if reg_value > constant {
                    self.comparison_flag = Comparison::GreaterThan;
                }
                else {
                    self.comparison_flag = Comparison::LessThanOrEqual;
                }
            },
            64 => {
                check_register64!(register);

                let reg_value = self.registers_64[register] as u64;
                let constant = self.get_8_bytes() as u64;
                self.advance_by_8_bytes();

                if reg_value > constant {
                    self.comparison_flag = Comparison::GreaterThan;
                }
                else {
                    self.comparison_flag = Comparison::LessThanOrEqual;
                }
            },
            128 => {
                check_register128!(register);

                let reg_value = self.registers_128[register] as u128;
                let constant = self.get_16_bytes() as u128;
                self.advance_by_16_bytes();

                if reg_value > constant {
                    self.comparison_flag = Comparison::GreaterThan;
                }
                else {
                    self.comparison_flag = Comparison::LessThanOrEqual;
                }
            },
            _ => return Err(Fault::InvalidSize),

        }

        Ok(())
    }
        
    fn leqc_opcode(&mut self) -> Result<(),Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register);

                let reg_value = self.registers_64[register] as u8;
                let constant = self.get_1_byte() as u8;
                self.advance_by_1_byte();

                if reg_value <= constant {
                    self.comparison_flag = Comparison::LessThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::GreaterThan;
                }
            },
            16 => {
                check_register64!(register);

                let reg_value = self.registers_64[register] as u16;
                let constant = self.get_2_bytes() as u16;
                self.advance_by_2_bytes();

                if reg_value <= constant {
                    self.comparison_flag = Comparison::LessThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::GreaterThan;
                }
            },
            32 => {
                check_register64!(register);

                let reg_value = self.registers_64[register] as u32;
                let constant = self.get_4_bytes() as u32;
                self.advance_by_4_bytes();

                if reg_value <= constant {
                    self.comparison_flag = Comparison::LessThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::GreaterThan;
                }
            },
            64 => {
                check_register64!(register);

                let reg_value = self.registers_64[register] as u64;
                let constant = self.get_8_bytes() as u64;
                self.advance_by_8_bytes();

                if reg_value <= constant {
                    self.comparison_flag = Comparison::LessThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::GreaterThan;
                }
            },
            128 => {
                check_register128!(register);

                let reg_value = self.registers_128[register] as u128;
                let constant = self.get_16_bytes() as u128;
                self.advance_by_16_bytes();

                if reg_value <= constant {
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
        
    fn geqc_opcode(&mut self) -> Result<(),Fault> {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register = (self.program[self.program_counter] as u8) as usize;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register);

                let reg_value = self.registers_64[register] as u8;
                let constant = self.get_1_byte() as u8;
                self.advance_by_1_byte();

                if reg_value >= constant {
                    self.comparison_flag = Comparison::GreaterThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::LessThan;
                }
            },
            16 => {
                check_register64!(register);

                let reg_value = self.registers_64[register] as u16;
                let constant = self.get_2_bytes() as u16;
                self.advance_by_2_bytes();

                if reg_value >= constant {
                    self.comparison_flag = Comparison::GreaterThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::LessThan;
                }
            },
            32 => {
                check_register64!(register);

                let reg_value = self.registers_64[register] as u32;
                let constant = self.get_4_bytes() as u32;
                self.advance_by_4_bytes();

                if reg_value >= constant {
                    self.comparison_flag = Comparison::GreaterThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::LessThan;
                }
            },
            64 => {
                check_register64!(register);

                let reg_value = self.registers_64[register] as u64;
                let constant = self.get_8_bytes() as u64;
                self.advance_by_8_bytes();

                if reg_value >= constant {
                    self.comparison_flag = Comparison::GreaterThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::LessThan;
                }
            },
            128 => {
                check_register128!(register);

                let reg_value = self.registers_128[register] as u128;
                let constant = self.get_16_bytes() as u128;
                self.advance_by_16_bytes();

                if reg_value >= constant {
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

    fn and_opcode(&mut self) -> SimpleResult {
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

    fn xor_opcode(&mut self) -> SimpleResult {
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

    fn not_opcode(&mut self) -> SimpleResult {
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

    fn shiftleft_opcode(&mut self) -> SimpleResult {
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

    fn shiftright_opcode(&mut self) -> SimpleResult {
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

    fn clear_opcode(&mut self) -> SimpleResult {
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

        let fd = self.registers_64[fd_register as usize];
        let value = self.registers_64[value_register as usize] as u8;

        let message = Message::WriteFile(fd, vec![value]);

        self.send_message(message)?;

        let response = self.recv_message()?;

        match response {
            Message::Success => {},
            _ => return Err(Fault::InvalidMessage),
        }
        
        Ok(())
    }

    fn write_opcode(&mut self) -> SimpleResult {
        let fd_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let pointer_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let length_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        check_register64!(fd_register as usize, pointer_register as usize, length_register as usize);

        let fd = self.registers_64[fd_register as usize];
        let pointer = self.registers_64[pointer_register as usize] as u64;
        let length = self.registers_64[length_register as usize] as u64;

        let string = self.get_string(pointer, length)?;

        let message = Message::WriteFile(fd, string);

        self.send_message(message)?;

        let response = self.recv_message()?;

        match response {
            Message::Success => {},
            _ => return Err(Fault::InvalidMessage),
        }


        Ok(())
    }

    fn flush_opcode(&mut self) -> SimpleResult {
        let fd_register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        check_register64!(fd_register as usize);

        let fd = self.registers_64[fd_register as usize];

        let message = Message::Flush(fd);

        self.send_message(message)?;

        let response = self.recv_message()?;

        match response {
            Message::Success => {},
            _ => return Err(Fault::InvalidMessage),
        }

        Ok(())
    }

    fn remainder_opcode(&mut self) -> SimpleResult {
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

    fn addfi_opcode(&mut self) -> SimpleResult {
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

    fn subfi_opcode(&mut self) -> SimpleResult {
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

    fn mulfi_opcode(&mut self) -> SimpleResult {
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

    fn divfi_opcode(&mut self) -> SimpleResult {
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

                if self.registers_f64[int_register as usize] == 0.0 {
                    return Err(Fault::DivideByZero);
                }
                

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

                if self.registers_f64[int_register as usize] == 0.0 {
                    return Err(Fault::DivideByZero);
                }

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

    fn addif_opcode(&mut self) -> SimpleResult {
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
                let int_value = i8::from_le_bytes((self.registers_64[int_register as usize] as u8).to_le_bytes().try_into().unwrap());
                
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

                self.registers_64[int_register as usize] = u8::from_le_bytes(new_value.to_le_bytes().try_into().unwrap()) as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] as u8 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x8000 == (Wrapping(new_value) + Wrapping(0x8000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
            },
            16 => {
                let int_value = i16::from_le_bytes((self.registers_64[int_register as usize] as u16).to_le_bytes().try_into().unwrap());
                
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

                self.registers_64[int_register as usize] = u16::from_le_bytes(new_value.to_le_bytes().try_into().unwrap()) as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] as u16 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x8000 == (Wrapping(new_value) + Wrapping(0x8000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
            },
            32 => {
                let int_value = i32::from_le_bytes((self.registers_64[int_register as usize] as u32).to_le_bytes().try_into().unwrap());
                
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

                self.registers_64[int_register as usize] = u32::from_le_bytes(new_value.to_le_bytes().try_into().unwrap()) as u64;

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] as u32 == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x80000000 == (Wrapping(new_value) + Wrapping(0x80000000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
                }
            },
            64 => {
                let int_value = i64::from_le_bytes((self.registers_64[int_register as usize] as u64).to_le_bytes().try_into().unwrap());
                
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

                self.registers_64[int_register as usize] = u64::from_le_bytes(new_value.to_le_bytes().try_into().unwrap());

                if new_value > int_value {
                    self.overflow_flag = true;
                }
                if self.registers_64[int_register as usize] == 0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    // Checking to see if the most significant bit is set
                    if new_value ^ 0x80000000 == (Wrapping(new_value) + Wrapping(0x80000000)).0 {
                        self.sign_flag = Sign::Positive;
                    }
                    else {
                        self.sign_flag = Sign::Negative;
                    }
                }
                // Fast way to check if the number is odd
                if new_value ^ 1 == (Wrapping(new_value) + Wrapping(1)).0 {
                    self.odd_flag = false;
                }
                else {
                    self.odd_flag = true;
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

    fn subif_opcode(&mut self) -> SimpleResult {
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
                let int_value = i8::from_le_bytes((self.registers_64[int_register as usize] as u8).to_le_bytes().try_into().unwrap());
                
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

                self.registers_64[int_register as usize] = u8::from_le_bytes(new_value.to_le_bytes().try_into().unwrap()) as u64;

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
                let int_value = i16::from_le_bytes((self.registers_64[int_register as usize] as u16).to_le_bytes().try_into().unwrap());
                
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

                self.registers_64[int_register as usize] = u16::from_le_bytes(new_value.to_le_bytes().try_into().unwrap()) as u64;

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
                let int_value = i32::from_le_bytes((self.registers_64[int_register as usize] as u32).to_le_bytes().try_into().unwrap());
                
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

                self.registers_64[int_register as usize] = u32::from_le_bytes(new_value.to_le_bytes().try_into().unwrap()) as u64;

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
                let int_value = i64::from_le_bytes(self.registers_64[int_register as usize].to_le_bytes().try_into().unwrap());
                
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
                let int_value = i128::from_le_bytes(self.registers_128[int_register as usize].to_le_bytes().try_into().unwrap());
                
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

    fn mulif_opcode(&mut self) -> SimpleResult {
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
                let int_value = i8::from_le_bytes((self.registers_64[int_register as usize] as u8).to_le_bytes().try_into().unwrap());
                
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

                self.registers_64[int_register as usize] = u8::from_le_bytes(new_value.to_le_bytes().try_into().unwrap()) as u64;

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
                let int_value = i16::from_le_bytes((self.registers_64[int_register as usize] as u16).to_le_bytes().try_into().unwrap());
                
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

                self.registers_64[int_register as usize] = u16::from_le_bytes(new_value.to_le_bytes().try_into().unwrap()) as u64;

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
                let int_value = i32::from_le_bytes((self.registers_64[int_register as usize] as u32).to_le_bytes().try_into().unwrap());
                
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

                self.registers_64[int_register as usize] = u32::from_le_bytes(new_value.to_le_bytes().try_into().unwrap()) as u64;

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
                let int_value = i64::from_le_bytes(self.registers_64[int_register as usize].to_le_bytes().try_into().unwrap());
                
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
                let int_value = i128::from_le_bytes(self.registers_128[int_register as usize].to_le_bytes().try_into().unwrap());
                
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

    fn divif_opcode(&mut self) -> SimpleResult {
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
                let int_value = i8::from_le_bytes((self.registers_64[int_register as usize] as u8).to_le_bytes().try_into().unwrap());
                
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

                if float_value == 0 {
                    return Err(Fault::DivideByZero);
                }

                self.remainder_64 = (Wrapping(int_value) % Wrapping(float_value)).0 as usize;

                let new_value = (Wrapping(int_value) / Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = u8::from_le_bytes(new_value.to_le_bytes().try_into().unwrap()) as u64;

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
                let int_value = i16::from_le_bytes((self.registers_64[int_register as usize] as u16).to_le_bytes().try_into().unwrap());
                
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

                if float_value == 0 {
                    return Err(Fault::DivideByZero);
                }

                self.remainder_64 = (Wrapping(int_value) % Wrapping(float_value)).0 as usize;

                let new_value = (Wrapping(int_value) / Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = u16::from_le_bytes(new_value.to_le_bytes().try_into().unwrap()) as u64;

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
                let int_value = i32::from_le_bytes((self.registers_64[int_register as usize] as u32).to_le_bytes().try_into().unwrap());
                
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

                if float_value == 0 {
                    return Err(Fault::DivideByZero);
                }

                self.remainder_64 = (Wrapping(int_value) % Wrapping(float_value)).0 as usize;

                let new_value = (Wrapping(int_value) / Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = u32::from_le_bytes(new_value.to_le_bytes().try_into().unwrap()) as u64;

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
                let int_value = i64::from_le_bytes((self.registers_64[int_register as usize] as u64).to_le_bytes().try_into().unwrap());
                
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

                if float_value == 0 {
                    return Err(Fault::DivideByZero);
                }

                self.remainder_64 = (Wrapping(int_value) % Wrapping(float_value)).0 as usize;

                let new_value = (Wrapping(int_value) / Wrapping(float_value)).0;

                self.registers_64[int_register as usize] = u64::from_le_bytes(new_value.to_le_bytes().try_into().unwrap());

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
                let int_value = i128::from_le_bytes((self.registers_64[int_register as usize] as u128).to_le_bytes().try_into().unwrap());
                
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
                
                if float_value == 0 {
                    return Err(Fault::DivideByZero);
                }
                
                self.remainder_128 = (Wrapping(int_value) % Wrapping(float_value)).0 as u128;

                let new_value = (Wrapping(int_value) / Wrapping(float_value)).0;

                self.registers_128[int_register as usize] = u128::from_le_bytes(new_value.to_le_bytes().try_into().unwrap());

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

    fn setf_opcode(&mut self) -> SimpleResult {
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

    fn dereff_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        let register = self.program[self.program_counter] as u8 as usize;
        self.advance_by_1_byte();
        let address = self.program[self.program_counter] as u64;
        self.advance_by_8_bytes();
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
            }
            _ => return Err(Fault::InvalidSize),
        }
        Ok(())
    }

    fn movef_opcode(&mut self) -> SimpleResult {
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

    fn derefregf_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        let address_register = self.program[self.program_counter] as usize;
        check_register64!(address_register);
        self.advance_by_1_byte();
        let offset = i64::from_le_bytes(self.program[self.program_counter..self.program_counter + 8].try_into().unwrap());
        self.advance_by_8_bytes();

        let sign = if offset < 0 { -1 } else { 1 };
        let offset = offset.abs() as u64;
        let address = match sign {
            -1 => self.registers_64[address_register] - offset,
            1 => self.registers_64[address_register] + offset,
            _ => unreachable!(),
        };

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

    fn addf_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        let register2 = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();

        match size {
            32 => {
                check_registerF32!(register1, register2);
                self.registers_f32[register1] += self.registers_f32[register2];

                if self.registers_f32[register1].is_nan() {
                    self.nan_flag = true;
                }
                else {
                    self.nan_flag = false;
                }
                if self.registers_f32[register1].is_infinite() {
                    self.infinity_flag = true;
                }
                else {
                    self.infinity_flag = false;
                }
                if self.registers_f32[register1] == 0.0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    if self.registers_f32[register1] < 0.0 {
                        self.sign_flag = Sign::Negative;
                    }
                    else {
                        self.sign_flag = Sign::Positive;
                    }
                }
            },
            64 => {
                check_registerF64!(register1, register2);
                self.registers_f64[register1] += self.registers_f64[register2];

                if self.registers_f64[register1].is_nan() {
                    self.nan_flag = true;
                }
                else {
                    self.nan_flag = false;
                }
                if self.registers_f64[register1].is_infinite() {
                    self.infinity_flag = true;
                }
                else {
                    self.infinity_flag = false;
                }
                if self.registers_f64[register1] == 0.0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    if self.registers_f64[register1] < 0.0 {
                        self.sign_flag = Sign::Negative;
                    }
                    else {
                        self.sign_flag = Sign::Positive;
                    }
                }
                
                
                
            },
            _ => return Err(Fault::InvalidSize),
        }

        Ok(())
    }

    fn subf_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        let register2 = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();

        match size {
            32 => {
                check_registerF32!(register1, register2);
                self.registers_f32[register1] -= self.registers_f32[register2];

                if self.registers_f32[register1].is_nan() {
                    self.nan_flag = true;
                }
                else {
                    self.nan_flag = false;
                }
                if self.registers_f32[register1].is_infinite() {
                    self.infinity_flag = true;
                }
                else {
                    self.infinity_flag = false;
                }
                if self.registers_f32[register1] == 0.0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    if self.registers_f32[register1] < 0.0 {
                        self.sign_flag = Sign::Negative;
                    }
                    else {
                        self.sign_flag = Sign::Positive;
                    }
                }
            },
            64 => {
                check_registerF64!(register1, register2);
                self.registers_f64[register1] -= self.registers_f64[register2];

                if self.registers_f64[register1].is_nan() {
                    self.nan_flag = true;
                }
                else {
                    self.nan_flag = false;
                }
                if self.registers_f64[register1].is_infinite() {
                    self.infinity_flag = true;
                }
                else {
                    self.infinity_flag = false;
                }
                if self.registers_f64[register1] == 0.0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    if self.registers_f64[register1] < 0.0 {
                        self.sign_flag = Sign::Negative;
                    }
                    else {
                        self.sign_flag = Sign::Positive;
                    }
                }
                
                
                
            },
            _ => return Err(Fault::InvalidSize),
        }

        Ok(())
    }
        
    fn mulf_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        let register2 = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();

        match size {
            32 => {
                check_registerF32!(register1, register2);
                self.registers_f32[register1] *= self.registers_f32[register2];

                if self.registers_f32[register1].is_nan() {
                    self.nan_flag = true;
                }
                else {
                    self.nan_flag = false;
                }
                if self.registers_f32[register1].is_infinite() {
                    self.infinity_flag = true;
                }
                else {
                    self.infinity_flag = false;
                }
                if self.registers_f32[register1] == 0.0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    if self.registers_f32[register1] < 0.0 {
                        self.sign_flag = Sign::Negative;
                    }
                    else {
                        self.sign_flag = Sign::Positive;
                    }
                }
            },
            64 => {
                check_registerF64!(register1, register2);
                self.registers_f64[register1] *= self.registers_f64[register2];

                if self.registers_f64[register1].is_nan() {
                    self.nan_flag = true;
                }
                else {
                    self.nan_flag = false;
                }
                if self.registers_f64[register1].is_infinite() {
                    self.infinity_flag = true;
                }
                else {
                    self.infinity_flag = false;
                }
                if self.registers_f64[register1] == 0.0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    if self.registers_f64[register1] < 0.0 {
                        self.sign_flag = Sign::Negative;
                    }
                    else {
                        self.sign_flag = Sign::Positive;
                    }
                }
                
                
                
            },
            _ => return Err(Fault::InvalidSize),
        }

        Ok(())
    }


    fn divf_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        let register2 = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();

        match size {
            32 => {
                check_registerF32!(register1, register2);

                if self.registers_f32[register2] == 0.0 {
                    return Err(Fault::DivideByZero);
                }
                
                self.registers_f32[register1] /= self.registers_f32[register2];

                if self.registers_f32[register1].is_nan() {
                    self.nan_flag = true;
                }
                else {
                    self.nan_flag = false;
                }
                if self.registers_f32[register1].is_infinite() {
                    self.infinity_flag = true;
                }
                else {
                    self.infinity_flag = false;
                }
                if self.registers_f32[register1] == 0.0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    if self.registers_f32[register1] < 0.0 {
                        self.sign_flag = Sign::Negative;
                    }
                    else {
                        self.sign_flag = Sign::Positive;
                    }
                }
            },
            64 => {
                check_registerF64!(register1, register2);

                if self.registers_f64[register2] == 0.0 {
                    return Err(Fault::DivideByZero);
                }
                
                self.registers_f64[register1] /= self.registers_f64[register2];

                if self.registers_f64[register1].is_nan() {
                    self.nan_flag = true;
                }
                else {
                    self.nan_flag = false;
                }
                if self.registers_f64[register1].is_infinite() {
                    self.infinity_flag = true;
                }
                else {
                    self.infinity_flag = false;
                }
                if self.registers_f64[register1] == 0.0 {
                    self.zero_flag = true;
                }
                else {
                    self.zero_flag = false;
                    if self.registers_f64[register1] < 0.0 {
                        self.sign_flag = Sign::Negative;
                    }
                    else {
                        self.sign_flag = Sign::Positive;
                    }
                }
                
                
                
            },
            _ => return Err(Fault::InvalidSize),
        }

        Ok(())
    }


    fn eqf_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        let register2 = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();

        match size {
            32 => {
                check_registerF32!(register1, register2);
                if self.registers_f32[register1] == self.registers_f32[register2] {
                    self.comparison_flag = Comparison::Equal;
                }
                else {
                    self.comparison_flag = Comparison::NotEqual;
                }
            },
            64 => {
                check_registerF64!(register1, register2);
                if self.registers_f64[register1] == self.registers_f64[register2] {
                    self.comparison_flag = Comparison::Equal;
                }
                else {
                    self.comparison_flag = Comparison::NotEqual;
                }
            },
            _ => return Err(Fault::InvalidSize),
        }

        Ok(())
    }

    fn neqf_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        let register2 = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();

        match size {
            32 => {
                check_registerF32!(register1, register2);
                if self.registers_f32[register1] != self.registers_f32[register2] {
                    self.comparison_flag = Comparison::NotEqual
                }
                else {
                    self.comparison_flag = Comparison::Equal;
                }
            },
            64 => {
                check_registerF64!(register1, register2);
                if self.registers_f64[register1] != self.registers_f64[register2] {
                    self.comparison_flag = Comparison::NotEqual;
                }
                else {
                    self.comparison_flag = Comparison::Equal;
                }
            },
            _ => return Err(Fault::InvalidSize),
        }

        Ok(())
    }

    fn ltf_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        let register2 = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();

        match size {
            32 => {
                check_registerF32!(register1, register2);
                if self.registers_f32[register1] < self.registers_f32[register2] {
                    self.comparison_flag = Comparison::LessThan;
                }
                else {
                    self.comparison_flag = Comparison::GreaterThanOrEqual;
                }
            },
            64 => {
                check_registerF64!(register1, register2);
                if self.registers_f64[register1] < self.registers_f64[register2] {
                    self.comparison_flag = Comparison::LessThan;
                }
                else {
                    self.comparison_flag = Comparison::GreaterThanOrEqual;
                }
            },
            _ => return Err(Fault::InvalidSize),
        }

        Ok(())
    }

    fn gtf_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        let register2 = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();

        match size {
            32 => {
                check_registerF32!(register1, register2);
                if self.registers_f32[register1] > self.registers_f32[register2] {
                    self.comparison_flag = Comparison::GreaterThan;
                }
                else {
                    self.comparison_flag = Comparison::LessThanOrEqual;
                }
            },
            64 => {
                check_registerF64!(register1, register2);
                if self.registers_f64[register1] > self.registers_f64[register2] {
                    self.comparison_flag = Comparison::GreaterThan;
                }
                else {
                    self.comparison_flag = Comparison::LessThanOrEqual;
                }
            },
            _ => return Err(Fault::InvalidSize),
        }

        Ok(())
    }

    fn leqf_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        let register2 = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();

        match size {
            32 => {
                check_registerF32!(register1, register2);
                if self.registers_f32[register1] <= self.registers_f32[register2] {
                    self.comparison_flag = Comparison::LessThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::GreaterThan;
                }
            },
            64 => {
                check_registerF64!(register1, register2);
                if self.registers_f64[register1] <= self.registers_f64[register2] {
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

    fn geqf_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register1 = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        let register2 = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();

        match size {
            32 => {
                check_registerF32!(register1, register2);
                if self.registers_f32[register1] >= self.registers_f32[register2] {
                    self.comparison_flag = Comparison::GreaterThanOrEqual;
                }
                else {
                    self.comparison_flag = Comparison::LessThan;
                }
            },
            64 => {
                check_registerF64!(register1, register2);
                if self.registers_f64[register1] >= self.registers_f64[register2] {
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

    fn jump_opcode(&mut self) -> SimpleResult {
        let line = self.program[self.program_counter] as usize;
        self.advance_by_8_bytes();

        if line >= self.program.len() {
            return Err(Fault::InvalidJump);
        }
        
        self.program_counter = line;

        Ok(())
    }

    fn jumpeq_opcode(&mut self) -> SimpleResult {
        let line = self.program[self.program_counter] as usize;
        self.advance_by_8_bytes();

        if line >= self.program.len() {
            return Err(Fault::InvalidJump);
        }

        if self.comparison_flag == Comparison::Equal {
            self.program_counter = line;
        }
        Ok(())
    }

    fn jumpneq_opcode(&mut self) -> SimpleResult {
        let line = self.program[self.program_counter] as usize;
        self.advance_by_8_bytes();

        if line >= self.program.len() {
            return Err(Fault::InvalidJump);
        }

        if self.comparison_flag == Comparison::NotEqual {
            self.program_counter = line;
        }
        Ok(())
    }

    fn jumplt_opcode(&mut self) -> SimpleResult {
        let line = self.program[self.program_counter] as usize;
        self.advance_by_8_bytes();

        if line >= self.program.len() {
            return Err(Fault::InvalidJump);
        }

        if self.comparison_flag == Comparison::LessThan {
            self.program_counter = line;
        }
        Ok(())
    }

    fn jumpgt_opcode(&mut self) -> SimpleResult {
        let line = self.program[self.program_counter] as usize;
        self.advance_by_8_bytes();

        if line >= self.program.len() {
            return Err(Fault::InvalidJump);
        }

        if self.comparison_flag == Comparison::GreaterThan {
            self.program_counter = line;
        }
        Ok(())
    }

    fn jumpleq_opcode(&mut self) -> SimpleResult {
        let line = self.program[self.program_counter] as usize;
        self.advance_by_8_bytes();

        if line >= self.program.len() {
            return Err(Fault::InvalidJump);
        }

        if self.comparison_flag == Comparison::LessThanOrEqual {
            self.program_counter = line;
        }
        Ok(())
    }

    fn jumpgeq_opcode(&mut self) -> SimpleResult {
        let line = self.program[self.program_counter] as usize;
        self.advance_by_8_bytes();

        if line >= self.program.len() {
            return Err(Fault::InvalidJump);
        }

        if self.comparison_flag == Comparison::GreaterThanOrEqual {
            self.program_counter = line;
        }
        Ok(())
    }

    fn jumpzero_opcode(&mut self) -> SimpleResult {
        let line = self.program[self.program_counter] as usize;
        self.advance_by_8_bytes();

        if line >= self.program.len() {
            return Err(Fault::InvalidJump);
        }

        if self.zero_flag {
            self.program_counter = line;
        }
        Ok(())
    }

    fn jumpnotzero_opcode(&mut self) -> SimpleResult {
        let line = self.program[self.program_counter] as usize;
        self.advance_by_8_bytes();

        if line >= self.program.len() {
            return Err(Fault::InvalidJump);
        }

        if !self.zero_flag {
            self.program_counter = line;
        }
        Ok(())
    }

    fn jumpneg_opcode(&mut self) -> SimpleResult {
        let line = self.program[self.program_counter] as usize;
        self.advance_by_8_bytes();
        
        if line >= self.program.len() {
            return Err(Fault::InvalidJump);
        }

        match self.sign_flag {
            Sign::Negative => {
                self.program_counter = line;
            },
            _ => {},
        }
        
        Ok(())
    }

    fn jumppos_opcode(&mut self) -> SimpleResult {
        let line = self.program[self.program_counter] as usize;
        self.advance_by_8_bytes();
        
        if line >= self.program.len() {
            return Err(Fault::InvalidJump);
        }

        match self.sign_flag {
            Sign::Positive => {
                self.program_counter = line;
            },
            _ => {},
        }
        
        Ok(())
    }

    fn jumpeven_opcode(&mut self) -> SimpleResult {
        let line = self.program[self.program_counter] as usize;
        self.advance_by_8_bytes();
        
        if line >= self.program.len() {
            return Err(Fault::InvalidJump);
        }

        if !self.odd_flag {
            self.program_counter = line;
        }
        
        Ok(())
    }

    fn jumpodd_opcode(&mut self) -> SimpleResult {
        let line = self.program[self.program_counter] as usize;
        self.advance_by_8_bytes();
        
        if line >= self.program.len() {
            return Err(Fault::InvalidJump);
        }

        if self.odd_flag {
            self.program_counter = line;
        }
        
        Ok(())
    }

    fn jumpback_opcode(&mut self) -> SimpleResult {
        let line = self.program[self.program_counter] as usize;
        self.advance_by_8_bytes();
        
        if self.program_counter - line >= self.program.len() {
            return Err(Fault::InvalidJump);
        }

        self.program_counter -= line;
        
        Ok(())
    }

    fn jumpforward_opcode(&mut self) -> SimpleResult {
        let line = self.program[self.program_counter] as usize;
        self.advance_by_8_bytes();
        
        if self.program_counter + line >= self.program.len() {
            return Err(Fault::InvalidJump);
        }

        self.program_counter += line;
        
        Ok(())
    }

    fn jumpinfinity_opcode(&mut self) -> SimpleResult {
        let line = self.program[self.program_counter] as usize;
        self.advance_by_8_bytes();
        
        if line >= self.program.len() {
            return Err(Fault::InvalidJump);
        }

        if self.infinity_flag {
            self.program_counter = line;
        }
        
        Ok(())
    }

    fn jumpnotinfinity_opcode(&mut self) -> SimpleResult {
        let line = self.program[self.program_counter] as usize;
        self.advance_by_8_bytes();
        
        if line >= self.program.len() {
            return Err(Fault::InvalidJump);
        }

        if !self.infinity_flag {
            self.program_counter = line;
        }
        
        Ok(())
    }

    fn jumpoverflow_opcode(&mut self) -> SimpleResult {
        let line = self.program[self.program_counter] as usize;
        self.advance_by_8_bytes();
        
        if line >= self.program.len() {
            return Err(Fault::InvalidJump);
        }

        if self.overflow_flag {
            self.program_counter = line;
        }
        
        Ok(())
    }

    fn jumpnotoverflow_opcode(&mut self) -> SimpleResult {
        let line = self.program[self.program_counter] as usize;
        self.advance_by_8_bytes();
        
        if line >= self.program.len() {
            return Err(Fault::InvalidJump);
        }

        if !self.overflow_flag {
            self.program_counter = line;
        }
        
        Ok(())
    }

    fn jumpunderflow_opcode(&mut self) -> SimpleResult {
        let line = self.program[self.program_counter] as usize;
        self.advance_by_8_bytes();
        
        if line >= self.program.len() {
            return Err(Fault::InvalidJump);
        }

        if self.overflow_flag && self.sign_flag == Sign::Positive {
            self.program_counter = line;
        }
        
        Ok(())
    }

    fn jumpnotunderflow_opcode(&mut self) -> SimpleResult {
        let line = self.program[self.program_counter] as usize;
        self.advance_by_8_bytes();

        if line >= self.program.len() {
            return Err(Fault::InvalidJump);
        }

        if !self.overflow_flag || self.sign_flag == Sign::Negative {
            self.program_counter = line;
        }

        Ok(())
    }

    fn jumpnan_opcode(&mut self) -> SimpleResult {
        let line = self.program[self.program_counter] as usize;
        self.advance_by_8_bytes();

        if line >= self.program.len() {
            return Err(Fault::InvalidJump);
        }

        if self.nan_flag {
            self.program_counter = line;
        }

        Ok(())
    }

    fn jumpnotnan_opcode(&mut self) -> SimpleResult {
        let line = self.program[self.program_counter] as usize;
        self.advance_by_8_bytes();

        if line >= self.program.len() {
            return Err(Fault::InvalidJump);
        }

        if !self.nan_flag {
            self.program_counter = line;
        }

        Ok(())
    }

    fn jumpremainder_opcode(&mut self) -> SimpleResult {
        let line = self.program[self.program_counter] as usize;
        self.advance_by_8_bytes();

        if line >= self.program.len() {
            return Err(Fault::InvalidJump);
        }

        if self.remainder_64 != 0 || self.remainder_128 != 0 {
            self.program_counter = line;
        }

        Ok(())
    }

    fn jumpnotremainder_opcode(&mut self) -> SimpleResult {
        let line = self.program[self.program_counter] as usize;
        self.advance_by_8_bytes();

        if line >= self.program.len() {
            return Err(Fault::InvalidJump);
        }

        if self.remainder_64 == 0 && self.remainder_128 == 0 {
            self.program_counter = line;
        }

        Ok(())
    }

    fn call_opcode(&mut self) -> SimpleResult {
        let line = self.get_8_bytes() as usize;
        self.advance_by_8_bytes();

        if line >= self.program.len() {
            return Err(Fault::InvalidJump);
        }
        self.push_stack(&self.program_counter.to_le_bytes());
        self.program_counter = line;

        Ok(())
    }

    fn return_opcode(&mut self) -> SimpleResult {
        let line = self.pop_stack(64)?;
        self.program_counter = u64::from_le_bytes(line[..].try_into().unwrap()) as usize;
        Ok(())
    }

    fn pop_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        let value = self.pop_stack(size as usize)?;

        match size {
            8 => {
                check_register64!(register as usize);
                let mut bytes = [0];
                bytes[0] = value[0];
                self.registers_64[register as usize] = u8::from_le_bytes(bytes) as u64;
            },
            16 => {
                check_register64!(register as usize);
                let mut bytes = [0;2];
                bytes[0] = value[0];
                bytes[1] = value[1];
                
                self.registers_64[register as usize] = u16::from_le_bytes(bytes) as u64;
            },
            32 => {
                check_register64!(register as usize);
                let mut bytes = [0;4];
                bytes[0] = value[0];
                bytes[1] = value[1];
                bytes[2] = value[2];
                bytes[3] = value[3];
                
                self.registers_64[register as usize] = u32::from_le_bytes(bytes) as u64;
            },
            64 => {
                check_register64!(register as usize);
                let mut bytes = [0;8];
                bytes[0] = value[0];
                bytes[1] = value[1];
                bytes[2] = value[2];
                bytes[3] = value[3];
                bytes[4] = value[4];
                bytes[5] = value[5];
                bytes[6] = value[6];
                bytes[7] = value[7];
                
                self.registers_64[register as usize] = u64::from_le_bytes(bytes);
            },
            128 => {
                check_register128!(register as usize);
                let mut bytes = [0;16];
                bytes[0] = value[0];
                bytes[1] = value[1];
                bytes[2] = value[2];
                bytes[3] = value[3];
                bytes[4] = value[4];
                bytes[5] = value[5];
                bytes[6] = value[6];
                bytes[7] = value[7];
                bytes[8] = value[8];
                bytes[9] = value[9];
                bytes[10] = value[10];
                bytes[11] = value[11];
                bytes[12] = value[12];
                bytes[13] = value[13];
                bytes[14] = value[14];
                bytes[15] = value[15];
                
                self.registers_128[register as usize] = u128::from_le_bytes(bytes);
            },
            _ => return Err(Fault::InvalidSize),
        }

        Ok(())
    }

    fn push_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(register as usize);
                let value = (self.registers_64[register as usize] as u8).to_le_bytes();
                self.push_stack(&value);
            },
            16 => {
                check_register64!(register as usize);
                let value = (self.registers_64[register as usize] as u16).to_le_bytes();
                self.push_stack(&value);
            },
            32 => {
                check_register64!(register as usize);
                let value = (self.registers_64[register as usize] as u32).to_le_bytes();
                self.push_stack(&value);
            },
            64 => {
                check_register64!(register as usize);
                let value = (self.registers_64[register as usize] as u64).to_le_bytes();
                self.push_stack(&value);
            },
            128 => {
                check_register128!(register as usize);
                let value = self.registers_128[register as usize].to_le_bytes();
                self.push_stack(&value);
            },
            _ => return Err(Fault::InvalidSize),
        }
        Ok(())
    }
    
    fn popf_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        let value = self.pop_stack(size as usize)?;


        match size {
            32 => {
                check_registerF32!(register as usize);
                let mut bytes = [0;4];
                bytes[0] = value[0];
                bytes[1] = value[1];
                bytes[2] = value[2];
                bytes[3] = value[3];
                
                self.registers_f32[register as usize] = f32::from_le_bytes(bytes);
            },
            64 => {
                check_registerF64!(register as usize);
                let mut bytes = [0;8];
                bytes[0] = value[0];
                bytes[1] = value[1];
                bytes[2] = value[2];
                bytes[3] = value[3];
                bytes[4] = value[4];
                bytes[5] = value[5];
                bytes[6] = value[6];
                bytes[7] = value[7];
                
                self.registers_f64[register as usize] = f64::from_le_bytes(bytes);
            },
            _ => return Err(Fault::InvalidSize),
        }

        Ok(())
    }

    fn pushf_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        match size {
            32 => {
                check_registerF32!(register as usize);
                let value = self.registers_f32[register as usize].to_le_bytes();
                self.push_stack(&value);
            },
            64 => {
                check_registerF64!(register as usize);
                let value = self.registers_f64[register as usize].to_le_bytes();
                self.push_stack(&value);
            },
            _ => return Err(Fault::InvalidSize),
        }
        Ok(())
    }

    fn derefstack_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        let register = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        let address = self.get_8_bytes();
        self.advance_by_8_bytes();
        match size {
            8 => {
                if address >= self.stack_len()? as u64 {
                    return Err(Fault::InvalidAddress(address));
                }
                let register_bytes = &mut self.registers_64[register].to_le_bytes()[0..1];
                
                self.read_bytes_from_stack(address, 1, register_bytes)?;

                self.registers_64[register] = u8::from_le_bytes((*register_bytes).try_into().unwrap()) as u64;
            },
            16 => {
                if address >= self.stack_len()? as u64 {
                    return Err(Fault::InvalidAddress(address));
                }
                let register_bytes = &mut self.registers_64[register].to_le_bytes()[0..2];

                self.read_bytes_from_stack(address, 2, register_bytes)?;

                self.registers_64[register] = u16::from_le_bytes((*register_bytes).try_into().unwrap()) as u64;
            },
            32 => {
                if address >= self.stack_len()? as u64 {
                    return Err(Fault::InvalidAddress(address));
                }
                let register_bytes = &mut self.registers_64[register].to_le_bytes()[0..4];

                self.read_bytes_from_stack(address, 4, register_bytes)?;

                self.registers_64[register] = u32::from_le_bytes((*register_bytes).try_into().unwrap()) as u64;
            },
            64 => {
                if address >= self.stack_len()? as u64 {
                    return Err(Fault::InvalidAddress(address));
                }
                let register_bytes = &mut self.registers_64[register].to_le_bytes()[0..8];

                self.read_bytes_from_stack(address, 8, register_bytes)?;

                self.registers_64[register] = u64::from_le_bytes((*register_bytes).try_into().unwrap()) as u64;
            },
            128 => {
                if address >= self.stack_len()? as u64 {
                    return Err(Fault::InvalidAddress(address));
                }
                let register_bytes = &mut self.registers_128[register].to_le_bytes()[0..16];

                self.read_bytes_from_stack(address, 16, register_bytes)?;

                self.registers_128[register] = u128::from_le_bytes((*register_bytes).try_into().unwrap()) as u128;
            },
            _ => return Err(Fault::InvalidSize),
        }
        Ok(())
    }

    fn movestack_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        let address = self.get_8_bytes();
        self.advance_by_8_bytes();
        match size {
            8 => {
                let register = self.program[self.program_counter] as u8 as usize;
                check_register64!(register);
                self.advance_by_1_byte();

                if address >= self.stack_len()? as u64 {
                    return Err(Fault::InvalidAddress(address));
                }

                self.write_bytes_to_stack(address, 1, &self.registers_64[register].to_le_bytes()[0..1])?;
            },
            16 => {
                let register = self.program[self.program_counter] as u8 as usize;
                check_register64!(register);
                self.advance_by_1_byte();

                if address >= self.stack_len()? as u64 - 1 {
                    return Err(Fault::InvalidAddress(address));
                }

                self.write_bytes_to_stack(address, 2, &self.registers_64[register].to_le_bytes()[0..2])?;
            },
            32 => {
                let register = self.program[self.program_counter] as u8 as usize;
                check_register64!(register);
                self.advance_by_1_byte();

                if address >= self.stack_len()? as u64 - 3 {
                    return Err(Fault::InvalidAddress(address));
                }

                self.write_bytes_to_stack(address, 4, &self.registers_64[register].to_le_bytes()[0..4])?;
            },
            64 => {
                let register = self.program[self.program_counter] as u8 as usize;
                check_register64!(register);
                self.advance_by_1_byte();

                if address >= self.stack_len()? as u64 - 7 {
                    return Err(Fault::InvalidAddress(address));
                }

                self.write_bytes_to_stack(address, 8, &self.registers_64[register].to_le_bytes()[0..8])?;
            },
            128 => {
                let register = self.program[self.program_counter] as u8 as usize;
                check_register128!(register);
                self.advance_by_1_byte();

                if address >= self.stack_len()? as u64 - 15 {
                    return Err(Fault::InvalidAddress(address));
                }

                self.write_bytes_to_stack(address, 16, &self.registers_128[register].to_le_bytes()[0..16])?;
            },
            _ => return Err(Fault::InvalidSize),

        }
        Ok(())
    }

    fn derefregstack_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        let address_register = self.program[self.program_counter] as usize;
        check_register64!(address_register);
        self.advance_by_1_byte();
        let offset = i64::from_le_bytes(self.program[self.program_counter..self.program_counter + 8].try_into().unwrap());
        self.advance_by_8_bytes();

        let sign = if offset < 0 { -1 } else { 1 };
        let offset = offset.abs() as u64;
        let address = match sign {
            -1 => self.registers_64[address_register] - offset,
            1 => self.registers_64[address_register] + offset,
            _ => unreachable!(),
        };

        match size {
            8 => {
                check_register64!(register);

                if address >= self.stack_len()? as u64 - 1 {
                    return Err(Fault::InvalidAddress(address));
                }

                let register_bytes = &mut self.registers_64[register].to_le_bytes()[0..1];

                self.read_bytes_from_stack(address, 1, register_bytes)?;

                self.registers_64[register] = u8::from_le_bytes((*register_bytes).try_into().unwrap()) as u64;
            },
            16 => {
                check_register64!(register);

                if address >= self.stack_len()? as u64 - 1 {
                    return Err(Fault::InvalidAddress(address));
                }

                let register_bytes = &mut self.registers_64[register].to_le_bytes()[0..2];

                self.read_bytes_from_stack(address, 2, register_bytes)?;

                self.registers_64[register] = u16::from_le_bytes((*register_bytes).try_into().unwrap()) as u64;
            },
            32 => {
                check_register64!(register);

                if address >= self.stack_len()? as u64 - 3 {
                    return Err(Fault::InvalidAddress(address));
                }

                let register_bytes = &mut self.registers_64[register].to_le_bytes()[0..4];

                self.read_bytes_from_stack(address, 4, register_bytes)?;

                self.registers_64[register] = u32::from_le_bytes((*register_bytes).try_into().unwrap()) as u64;
            },
            64 => {
                check_register64!(register);

                if address >= self.stack_len()? as u64 - 7 {
                    return Err(Fault::InvalidAddress(address));
                }

                let register_bytes = &mut self.registers_64[register].to_le_bytes()[0..8];

                self.read_bytes_from_stack(address, 8, register_bytes)?;

                self.registers_64[register] = u64::from_le_bytes((*register_bytes).try_into().unwrap());
            },
            128 => {
                check_register128!(register);

                if address >= self.stack_len()? as u64 - 15 {
                    return Err(Fault::InvalidAddress(address));
                }

                let register_bytes = &mut self.registers_128[register].to_le_bytes()[0..16];

                self.read_bytes_from_stack(address, 16, register_bytes)?;

                self.registers_128[register] = u128::from_le_bytes((*register_bytes).try_into().unwrap());
            }
            _ => {
                return Err(Fault::InvalidSize);
            }

        }

        Ok(())
        
    }

    fn derefstackf_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        let register = self.program[self.program_counter] as u8 as usize;
        self.advance_by_1_byte();
        let address = self.program[self.program_counter] as u64;
        self.advance_by_8_bytes();
        match size {
            32 => {
                check_registerF32!(register);
                
                if address >= self.stack_len()? as u64 - 3 {
                    return Err(Fault::InvalidAddress(address));
                }

                let mut bytes = 0.0f32.to_le_bytes();

                self.read_bytes_from_stack(address, 4, &mut bytes)?;

                self.registers_f32[register] = f32::from_le_bytes(bytes);
            }
            64 => {
                check_registerF64!(register);
                
                if address >= self.stack_len()? as u64 - 7 {
                    return Err(Fault::InvalidAddress(address));
                }

                let mut bytes = 0.0f64.to_le_bytes();

                self.read_bytes_from_stack(address, 8, &mut bytes)?;

                self.registers_f64[register] = f64::from_le_bytes(bytes);
                
            }
            _ => return Err(Fault::InvalidSize),
        }
        Ok(())
    }

    fn movestackf_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as u8 as usize;
        self.advance_by_1_byte();
        let register = self.program[self.program_counter] as u8 as usize;
        self.advance_by_1_byte();
        match size {
            32 => {
                check_registerF32!(register as usize);
                let address = self.program[self.program_counter] as u64;
                self.advance_by_8_bytes();

                if address >= self.stack_len()? as u64 - 3 {
                    return Err(Fault::InvalidAddress(address));
                }

                let mut float_bytes = self.registers_f32[register as usize].to_le_bytes();

                self.read_bytes_from_stack(address, 4, &mut float_bytes)?;

                self.registers_f32[register as usize] = f32::from_le_bytes(float_bytes);
            },
            64 => {
                check_registerF64!(register as usize);
                let address = self.program[self.program_counter] as u64;
                self.advance_by_8_bytes();

                if address >= self.stack_len()? as u64 - 7 {
                    return Err(Fault::InvalidAddress(address));
                }

                let mut float_bytes = self.registers_f64[register as usize].to_le_bytes();

                self.read_bytes_from_stack(address, 8, &mut float_bytes)?;

                self.registers_f64[register as usize] = f64::from_le_bytes(float_bytes);
                
            },
            _ => return Err(Fault::InvalidSize),

        }
        Ok(())
    }

    fn derefregstackf_opcode(&mut self) -> SimpleResult {
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register = self.program[self.program_counter] as usize;
        self.advance_by_1_byte();
        let address_register = self.program[self.program_counter] as usize;
        check_register64!(address_register);
        self.advance_by_1_byte();

        let offset = i64::from_le_bytes(self.program[self.program_counter..self.program_counter + 8].try_into().unwrap());
        self.advance_by_8_bytes();

        let sign = if offset < 0 { -1 } else { 1 };
        let offset = offset.abs() as u64;
        let address = match sign {
            -1 => self.registers_64[address_register] - offset,
            1 => self.registers_64[address_register] + offset,
            _ => unreachable!(),
        };

        match size {
            32 => {
                check_registerF32!(register);

                if address >= self.stack_len()? as u64 - 3 {
                    return Err(Fault::InvalidAddress(address));
                }

                let mut bytes = 0.0f32.to_le_bytes();

                self.read_bytes_from_stack(address, 4, &mut bytes)?;

                self.registers_f32[register] = f32::from_le_bytes(bytes);
            },
            64 => {
                check_registerF64!(register);

                if address >= self.stack_len()? as u64 - 7 {
                    return Err(Fault::InvalidAddress(address));
                }

                let mut bytes = 0.0f64.to_le_bytes();

                self.read_bytes_from_stack(address, 8, &mut bytes)?;

                self.registers_f64[register] = f64::from_le_bytes(bytes);
            },
            _ => {
                return Err(Fault::InvalidSize);
            }

        }

        Ok(())
    }

    fn regmove_opcode(&mut self) -> SimpleResult {
        let register1 = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register2 = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        match size {
            8 | 16 | 32 | 64 => {
                check_register64!(register1 as usize, register2 as usize);

                self.registers_64[register1 as usize] = self.registers_64[register2 as usize];
            },
            128 => {
                check_register128!(register1 as usize, register2 as usize);

                self.registers_128[register1 as usize] = self.registers_128[register2 as usize];
            },
            _ => return Err(Fault::InvalidSize),

        }
        
        Ok(())
    }

    fn regmovef_opcode(&mut self) -> SimpleResult {
        let register1 = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let register2 = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let size = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        match size {
            32 => {
                check_registerF32!(register1 as usize, register2 as usize);

                self.registers_f32[register1 as usize] = self.registers_f32[register2 as usize];
            },
            64 => {
                check_registerF64!(register1 as usize, register2 as usize);

                self.registers_f64[register1 as usize] = self.registers_f64[register2 as usize];
            },
            _ => return Err(Fault::InvalidSize),

        }
        
        Ok(())
    }

    fn open_opcode(&mut self) -> SimpleResult {
        let pointer_reg = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let size_reg = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let flag_reg = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let fd_reg = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        check_register64!(pointer_reg as usize, size_reg as usize, flag_reg as usize, fd_reg as usize);

        let pointer = self.registers_64[pointer_reg as usize] as u64;
        let size = self.registers_64[size_reg as usize] as u64;
        let flags = self.registers_64[flag_reg as usize] as u8;

        let file_name = self.get_string(pointer, size)?;

        let message = Message::OpenFile(file_name, flags);

        self.send_message(message)?;

        let message = self.recv_message()?;

        match message {
            Message::FileDescriptor(fd) => {
                self.registers_64[fd_reg as usize] = fd;
            },
            Message::Error(fault) => {
                return Err(fault);
            },
            _ => {
                return Err(Fault::InvalidMessage);
            }
        }
        
        Ok(())
    }

    fn close_opcode(&mut self) -> SimpleResult {
        let fd_reg = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        check_register64!(fd_reg as usize);

        let fd = self.registers_64[fd_reg as usize];

        let message = Message::CloseFile(fd);

        self.send_message(message)?;

        let message = self.recv_message()?;

        match message {
            Message::Success => {
                return Ok(());
            },
            Message::Error(fault) => {
                return Err(fault);
            },
            _ => {
                return Err(Fault::InvalidMessage);
            }
        }
    }

    fn threadspawn_opcode(&mut self) -> SimpleResult {

        let program_counter_reg = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let thread_id_reg = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        check_register64!(program_counter_reg as usize, thread_id_reg as usize);

        let message = Message::SpawnThread(self.registers_64[program_counter_reg as usize] as u64);

        self.send_message(message)?;

        let message = self.recv_message()?;

        match message {
            Message::ThreadSpawned(thread_id) => {
                self.registers_64[thread_id_reg as usize] = thread_id as u64;
                self.threads.insert(thread_id);
            },
            Message::Error(fault) => {
                return Err(fault);
            },
            _ => {
                return Err(Fault::InvalidMessage);
            }
        }
        
        Ok(())
    }

    fn threadreturn_opcode(&mut self) -> SimpleResult {

        let message = Message::ThreadDone(0);

        self.send_message(message)?;

        let message = self.recv_message()?;

        match message {
            Message::Success => {
                return Ok(());
            },
            Message::Error(fault) => {
                return Err(fault);
            },
            _ => {
                return Err(Fault::InvalidMessage);
            }
        }
    }

    fn threadjoin_opcode(&mut self) -> SimpleResult {
        let thread_id_reg = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        check_register64!(thread_id_reg as usize);

        let core_id = self.registers_64[thread_id_reg as usize] as CoreId;

        let message = Message::JoinThread(core_id);

        self.send_message(message)?;
        
        let message = self.recv_message()?;

        match message {
            Message::Success => {
                return Ok(());
            },
            Message::Error(fault) => {
                return Err(fault);
            },
            _ => {
                return Err(Fault::InvalidMessage);
            }
        }
    }

    fn threaddetach_opcode(&mut self) -> SimpleResult {
        let thread_id_reg = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        check_register64!(thread_id_reg as usize);

        let core_id = self.registers_64[thread_id_reg as usize] as CoreId;

        self.remove_thread(core_id);

        Ok(())
    }

    fn foreigncall_opcode(&mut self) -> SimpleResult {
        let function_id_reg = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        check_register64!(function_id_reg as usize);

        let function_id = self.registers_64[function_id_reg as usize] as u64;

        let message = Message::GetForeignFunction(function_id);

        self.send_message(message)?;

        let message = self.recv_message()?;

        let (argument, function) = match message {
            Message::ForeignFunction(argument,function) => (argument, function),
            Message::Error(fault) => {
                return Err(fault);
            },
            _ => {
                return Err(Fault::InvalidMessage);
            }
        };

        function(self, argument)?;
        
        Ok(())
    }

    fn stackpointer_opcode(&mut self) -> SimpleResult {
        let stackptr_reg = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        check_register64!(stackptr_reg as usize);

        self.registers_64[stackptr_reg as usize] = self.stack_len()? as u64;

        Ok(())
    }

    fn malloc_opcode(&mut self) -> SimpleResult {
        let ptr_reg = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let size_reg = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        check_register64!(ptr_reg as usize, size_reg as usize);

        let size = self.registers_64[size_reg as usize];

        let message = Message::Malloc(size);

        self.send_message(message)?;

        let message = self.recv_message()?;

        match message {
            Message::MemoryPointer(ptr) => {
                self.registers_64[ptr_reg as usize] = ptr as u64;
            },
            Message::Error(fault) => {
                return Err(fault);
            },
            _ => {
                return Err(Fault::InvalidMessage);
            }
        }
        
        Ok(())
    }

    fn free_opcode(&mut self) -> SimpleResult {
        let ptr_reg = self.program[self.program_counter] as u8;

        check_register64!(ptr_reg as usize);

        let ptr = self.registers_64[ptr_reg as usize];

        let message = Message::Free(ptr);

        self.send_message(message)?;

        let message = self.recv_message()?;

        match message {
            Message::Success => {
                return Ok(());
            },
            Message::Error(fault) => {
                return Err(fault);
            },
            _ => {
                return Err(Fault::InvalidMessage);
            }
        }
    }

    fn realloc_opcode(&mut self) -> SimpleResult {
        let ptr_reg = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let size_reg = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        check_register64!(ptr_reg as usize, size_reg as usize);

        let ptr = self.registers_64[ptr_reg as usize];

        let size = self.registers_64[size_reg as usize];

        let message = Message::Realloc(ptr, size);

        self.send_message(message)?;

        let message = self.recv_message()?;

        match message {
            Message::MemoryPointer(ptr) => {
                self.registers_64[ptr_reg as usize] = ptr as u64;
            },
            Message::Error(fault) => {
                return Err(fault);
            },
            _ => {
                return Err(Fault::InvalidMessage);
            }
        }
        Ok(())
    }

    fn sleep_opcode(&mut self) -> SimpleResult {

        let time = self.get_8_bytes();
        self.advance_by_8_bytes();
        let scale = self.get_8_bytes();
        self.advance_by_8_bytes();

        thread::sleep(Duration::from_secs(time * scale));

        Ok(())
    }

    fn sleepreg_opcode(&mut self) -> SimpleResult {
        let time_reg = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let scale_reg = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        check_register64!(time_reg as usize, scale_reg as usize);

        let time = self.registers_64[time_reg as usize];
        let scale = self.registers_64[scale_reg as usize];

        thread::sleep(Duration::from_secs(time * scale));

        Ok(())
    }

    fn random_opcode(&mut self) -> SimpleResult {
        let size = self.get_1_byte();
        self.advance_by_1_byte();
        let reg = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        match size {
            8 => {
                check_register64!(reg as usize);
                let random = rand::random::<u8>();
                self.registers_64[reg as usize] = random as u64;
            },
            16 => {
                check_register64!(reg as usize);
                let random = rand::random::<u16>();
                self.registers_64[reg as usize] = random as u64;
            },
            32 => {
                check_register64!(reg as usize);
                let random = rand::random::<u32>();
                self.registers_64[reg as usize] = random as u64;
            },
            64 => {
                check_register64!(reg as usize);
                let random = rand::random::<u64>();
                self.registers_64[reg as usize] = random as u64;
            },
            128 => {
                check_register128!(reg as usize);
                let random = rand::random::<u128>();
                self.registers_128[reg as usize] = random;
            },
            _ => {
                return Err(Fault::InvalidSize);
            }
        }
        Ok(())
    }

    fn randomf_opcode(&mut self) -> SimpleResult {
        let size = self.get_1_byte();
        self.advance_by_1_byte();
        let reg = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        match size {
            32 => {
                check_registerF32!(reg as usize);
                let random = rand::random::<f32>();
                self.registers_f32[reg as usize] = random;
            },
            64 => {
                check_registerF64!(reg as usize);
                let random = rand::random::<f64>();
                self.registers_f64[reg as usize] = random;
            },
            _ => {
                return Err(Fault::InvalidSize);
            }
        }

        Ok(())
    }

    fn readbyte_opcode(&mut self) -> SimpleResult {
        let fd_reg = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let value_reg = self.program[self.program_counter] as u8;

        check_register64!(fd_reg as usize, value_reg as usize);

        let fd = self.registers_64[fd_reg as usize];

        let message = Message::ReadFile(fd, 1);

        self.send_message(message)?;

        let message = self.recv_message()?;

        match message {
            Message::FileData(data,_) => {
                self.registers_64[value_reg as usize] = data[0] as u64;
            },
            Message::Error(fault) => {
                return Err(fault);
            },
            _ => {
                return Err(Fault::InvalidMessage);
            }
        }

        Ok(())
    }

    fn read_opcode(&mut self) -> SimpleResult {
        let fd_reg = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let ptr_reg = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let amount_reg = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        check_register64!(fd_reg as usize, ptr_reg as usize, amount_reg as usize);

        let fd = self.registers_64[fd_reg as usize];
        let ptr = self.registers_64[ptr_reg as usize];
        let amount = self.registers_64[amount_reg as usize];

        let message = Message::ReadFile(fd, amount);

        self.send_message(message)?;

        let message = self.recv_message()?;

        match message {
            Message::FileData(bytes, amount) => {
                self.registers_64[amount_reg as usize] = amount as u64;
                loop {
                    match self.memory.try_write() {
                        Ok(mut memory) => {
                            for (i, byte) in bytes.iter().enumerate() {
                                memory[ptr as usize + i] = *byte;
                            }
                            break;
                        },
                        Err(TryLockError::WouldBlock) => continue,
                        Err(_) => return Err(Fault::CorruptedMemory),
                    }
                }
            },
            Message::Error(fault) => return Err(fault),
            _ => return Err(Fault::InvalidMessage),
        }

        Ok(())
    }

    fn readstack_opcode(&mut self) -> SimpleResult {
        let fd_reg = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let ptr_reg = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();
        let amount_reg = self.program[self.program_counter] as u8;
        self.advance_by_1_byte();

        check_register64!(fd_reg as usize, ptr_reg as usize, amount_reg as usize);

        let fd = self.registers_64[fd_reg as usize];
        let ptr = self.registers_64[ptr_reg as usize];
        let amount = self.registers_64[amount_reg as usize];

        let message = Message::ReadFile(fd, amount);

        self.send_message(message)?;

        let message = self.recv_message()?;

        match message {
            Message::FileData(bytes, amount) => {
                self.registers_64[amount_reg as usize] = amount as u64;
                self.write_bytes_to_stack(ptr, amount, &bytes)?;
            },
            Message::Error(fault) => return Err(fault),
            _ => return Err(Fault::InvalidMessage),
        }

        Ok(())
    }
}





#[cfg(test)]
mod tests {
    /*use super::*;
    use std::sync::mpsc::channel;

    #[test]
    fn test_addi() {
        let program = Arc::new(vec![6,0,64,0,1]);
        let memory = Arc::new(RwLock::new(Vec::new()));
        let (sender, receiver) = channel();
        let mut core = Core::new(memory, sender, receiver);
        core.add_program(program);

        core.registers_64[0] = 1;
        core.registers_64[1] = 2;

        core.run(0).unwrap();

        assert_eq!(core.registers_64[0] as i64, 3);
    }

    #[test]
    fn test_subi() {
        let program = vec![7,0,64,0,1];
        let memory = Arc::new(RwLock::new(Vec::new()));
        let (sender, receiver) = channel();
        let mut core = Core::new(memory, sender, receiver);
        core.add_program(Arc::new(program));

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
        let (sender, receiver) = channel();
        let mut core = Core::new(memory, sender, receiver);
        core.add_program(Arc::new(program));

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
        let (sender, receiver) = channel();
        let mut core = Core::new(memory, sender, receiver);
        core.add_program(Arc::new(program));

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
        let (sender, receiver) = channel();
        let mut core = Core::new(memory, sender, receiver);
        core.add_program(Arc::new(program));

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
        let (sender, receiver) = channel();
        let mut core = Core::new(memory, sender, receiver);
        core.add_program(Arc::new(program));

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
        let (sender, receiver) = channel();
        let mut core = Core::new(memory, sender, receiver);
        core.add_program(Arc::new(program));

        core.registers_64[0] = 1;
        core.registers_64[1] = 1;

        core.run(0).unwrap();

        assert_eq!(core.comparison_flag, Comparison::Equal);
    }

    #[test]
    fn test_lti() {
        let program = vec![12,0,64,0,1];
        let memory = Arc::new(RwLock::new(Vec::new()));
        let (sender, receiver) = channel();
        let mut core = Core::new(memory, sender, receiver);
        core.add_program(Arc::new(program));

        core.registers_64[0] = 1;
        core.registers_64[1] = 2;

        core.run(0).unwrap();

        assert_eq!(core.comparison_flag, Comparison::LessThan);
    }

    #[test]
    fn test_geqi() {
        let program = vec![15,0,64,0,1];
        let memory = Arc::new(RwLock::new(Vec::new()));
        let (sender, receiver) = channel();
        let mut core = Core::new(memory, sender, receiver);
        core.add_program(Arc::new(program));

        core.registers_64[0] = 2;
        core.registers_64[1] = 1;

        core.run(0).unwrap();

        assert_eq!(core.comparison_flag, Comparison::GreaterThanOrEqual);
    }

    #[test]
    fn test_addu() {
        let program = vec![16,0,128,0,1];
        let memory = Arc::new(RwLock::new(Vec::new()));
        let (sender, receiver) = channel();
        let mut core = Core::new(memory, sender, receiver);
        core.add_program(Arc::new(program));

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
        let (sender, receiver) = channel();
        let mut core = Core::new(memory, sender, receiver);
        core.add_program(Arc::new(program));
        
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
        let (sender, receiver) = channel();
        let mut core = Core::new(memory, sender, receiver);
        core.add_program(Arc::new(program));

        core.registers_64[0] = 1;
        core.registers_64[1] = 72;
        core.registers_64[2] = 105;
        core.registers_64[3] = 33;
        core.registers_64[4] = 10;

        
        core.run(0).unwrap();
        
        
    }

    #[test]
    fn test_hello_world() {
        let program = vec![146,0,0,1,2,149,0,0];
        let memory = vec![0, 104,101,108,108,111,32,119,111,114,108,100,10];
        let memory = Arc::new(RwLock::new(memory));
        let (sender, receiver) = channel();
        let mut core = Core::new(memory, sender, receiver);
        core.add_program(Arc::new(program));

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
        let (sender, receiver) = channel();
        let mut core = Core::new(memory, sender, receiver);
        core.add_program(Arc::new(program));

        core.run(0).unwrap();

        println!("{}", core.registers_f32[0]);
        println!("{:?}", 23.0f32.to_ne_bytes());
        println!("{}", f32::from_ne_bytes([0x00,0x00,0xb8,0x41]));
        println!("{}", f32::from_ne_bytes([0x41,0xb8,0x00,0x00]));

        assert_eq!(core.registers_f32[0], f32::from_ne_bytes([0x00,0x00,0xb8,0x41]));
    }

    #[test]
    fn test_jumps() {
        let program = vec![20,0, 8,0,2, 71,0, 26,0,0,0,0,0,0,0, 16,0,8,0,1, 70,0, 0,0,0,0,0,0,0,0];
        let memory = Arc::new(RwLock::new(vec![]));
        let (sender, receiver) = channel();
        let mut core = Core::new(memory, sender, receiver);
        core.add_program(Arc::new(program));

        core.registers_64[0] = 0;
        core.registers_64[1] = 2;
        core.registers_64[2] = 8;

        core.run(0).unwrap();

        assert_eq!(core.registers_64[0], 8);
        
    }*/


}

