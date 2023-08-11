

use std::collections::HashMap;
use std::sync::{RwLock, Arc};
use std::sync::TryLockError;

use crate::virtual_machine::Memory;
use crate::{Core, Byte, SimpleResult, Message, CoreResult, Collector, Pointer, GarbageCollectorCore};
use crate::core::MachineCore;


impl Core for GarbageCollector {
    fn run(&mut self, program_counter: usize) -> SimpleResult {
        self.core.program_counter = program_counter;

        let mut is_done = false;
        while !is_done {
            self.check_messages()?;
            
            is_done = self.collect()?;
        }
        Ok(())
    }

    fn run_once(&mut self) -> SimpleResult {
        self.collect()?;
        Ok(())
    }

    fn add_program(&mut self, program: Arc<Vec<Byte>>) {
        self.core.add_program(program);
    }

    fn add_channels(&mut self, machine_send: std::sync::mpsc::Sender<Message>, core_receive: std::sync::mpsc::Receiver<Message>) {
        self.core.add_channels(machine_send, core_receive);
    }
    fn send_message(&self, message: Message) -> SimpleResult {
        self.core.send_message(message)
    }

    fn recv_message(&mut self) -> CoreResult<Message> {
        self.core.recv_message()
    }

    fn check_messages(&mut self) -> SimpleResult {
        self.core.check_messages()
    }

    fn check_program_counter(&self) -> CoreResult<bool> {
        self.core.check_program_counter()
    }

    fn decode_opcode(&mut self) -> crate::instruction::Opcode {
        self.core.decode_opcode()
    }

    fn get_register_64<'input>(&'input mut self, register: usize) -> CoreResult<&'input mut u64> {
        self.core.get_register_64(register)
    }

    fn get_register_128<'input>(&'input mut self, register: usize) -> CoreResult<&'input mut u128> {
        self.core.get_register_128(register)
    }

    fn get_register_f32<'input>(&'input mut self, register: usize) -> CoreResult<&'input mut f32> {
        self.core.get_register_f32(register)
    }

    fn get_register_f64<'input>(&'input mut self, register: usize) -> CoreResult<&'input mut f64> {
        self.core.get_register_f64(register)
    }   
}

impl Collector for GarbageCollector {

    fn add_stack(&mut self, stack: Box<[Byte]>, offset_size: usize) {
        self.stack = stack;
        self.stack_offset_size = offset_size;
    }

    fn add_heap(&mut self, memory: Arc<RwLock<Memory>>) {
        self.heap = memory;
    }

    fn add_data_segment(&mut self, data:  Arc<Vec<Byte>>) {
        self.data_segment = data;
    }
}

impl GarbageCollectorCore for GarbageCollector {}

/// A garbage collector for the virtual machine.
pub struct GarbageCollector {
    /// This, with the Core trait, should allow us to do C style inheritance.
    core: MachineCore,
    data_segment: Arc<Vec<Byte>>,
    stack: Box<[Byte]>,
    stack_offset_size: usize,
    heap: Arc<RwLock<Memory>>,
    found_ptrs: HashMap<Pointer, bool>,
    found_flag: bool,
}

impl GarbageCollector {
    pub fn new() -> Self {
        Self {
            core: MachineCore::new(),
            data_segment: Arc::new(Vec::new()),
            stack: Box::new([]),
            stack_offset_size: 0,
            heap: Arc::new(RwLock::new(Memory::new())),
            found_ptrs: HashMap::new(),
            found_flag: true,
        }
    }

    fn collect(&mut self) -> CoreResult<bool> {

        let message = self.recv_message()?;

        let stack_pointers;

        match message {
            Message::StackPointers(pointers) => {stack_pointers = pointers;},
            _ => return Ok(true),
        }
        
        if self.core.program.len() != 0 {
            self.core.execute_instruction()?;
        }

        let mut memory;


        let memory_len;

        for (ptr, _) in memory.allocated_blocks.iter() {
            self.found_ptrs.insert(*ptr, !self.found_flag);
        }
        
        const POINTER_SIZE: usize = 8;
        const NULL_OFFSET: usize = 1;
        
        for stack in stacks {


            for i in (NULL_OFFSET..stack.len()).step_by(POINTER_SIZE) {

                let mut address = u64::from_le_bytes(stack[i..(i + POINTER_SIZE)].try_into().unwrap());

                while address != 0 && address < memory_len as u64 {
                    if memory.allocated_blocks.contains_key(&address) {
                        self.found_ptrs.insert(address, self.found_flag);
                    }

                    loop {
                        match memory.memory.try_read() {
                            Ok(memory) => {
                                address = u64::from_le_bytes(memory[address as usize..(address as usize + POINTER_SIZE)].try_into().unwrap());
                                break;
                            },
                            Err(TryLockError::WouldBlock) => continue,
                            Err(_) => panic!("Poisoned lock."),
                        }
                    }
                }
            }
        }

        let mut ptrs_to_remove = Vec::new();

        for (ptr, flag) in self.found_ptrs.iter() {
            if *flag != self.found_flag {
                ptrs_to_remove.push(*ptr);
                memory.free(*ptr);
            }

        }
        for ptr in ptrs_to_remove {
            self.found_ptrs.remove(&ptr);
        }

        self.found_flag = !self.found_flag;

        self.send_message(Message::Success)?;

        Ok(false)
    }
}

