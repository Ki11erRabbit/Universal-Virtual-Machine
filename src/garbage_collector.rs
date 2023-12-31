

use std::collections::HashMap;
use std::sync::{RwLock, Arc};

use log::{info, trace, error};

use crate::virtual_machine::Memory;
use crate::{Core, Byte, SimpleResult, Message, CoreResult, Collector, Pointer, GarbageCollectorCore, WholeStack, get_heap_len_panic, access, };
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

    fn wait_for_interrupt(&mut self) -> SimpleResult {
        self.core.wait_for_interrupt()
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

    fn add_stack(&mut self, stack: WholeStack) {
        self.stack = stack;
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
    stack: WholeStack,
    heap: Arc<RwLock<Memory>>,
    found_ptrs: HashMap<Pointer, bool>,
    found_flag: bool,
}

impl GarbageCollector {
    pub fn new() -> Self {
        Self {
            core: MachineCore::new(),
            data_segment: Arc::new(Vec::new()),
            stack: Arc::new(Vec::new()),
            heap: Arc::new(RwLock::new(Memory::new())),
            found_ptrs: HashMap::new(),
            found_flag: true,
        }
    }

    fn collect(&mut self) -> CoreResult<bool> {

        info!("Garbage Collector: Waiting for message to start collection.");
        let message = self.recv_message()?;

        let stack_pointers;

        match message {
            Message::StackPointers(pointers) => {stack_pointers = pointers;},
            _ => return Ok(true),
        }
        
        if self.core.data_segment.len() != 0 {
            self.core.execute_instruction()?;
        }

        let mut heap;

        access!(self.heap.try_write(), mem, {
            heap = mem;
            break;
        },{
            error!("Garbage Collector: Poisoned lock on heap.");
            panic!("Poisoned lock.");
        });
        
        for (ptr, _) in heap.allocated_blocks.iter() {
            self.found_ptrs.insert(*ptr, !self.found_flag);
        }

        let mut stacks = Vec::new();

        for stack in self.stack.iter() {

            access!(stack.try_read(), stack, {
                stacks.push(stack.clone());
                break;
            },{
                error!("Garbage Collector: Poisoned lock on stack.");
                panic!("Poisoned lock.");
            });
            
        }

        let data_segment_size = self.data_segment.len();
        let stack_len = stacks.iter().map(|stack| stack.len()).sum::<usize>();
        let heap_len;
        get_heap_len_panic!(heap.memory, heap_len);

        let memory_len = data_segment_size + stack_len + heap_len;
        
        const POINTER_SIZE: usize = 8;

        let mut num_bytes_collected = 0;
        
        for (stack, stack_pointer) in stacks.iter().zip(stack_pointers.iter()) {


            for i in (0..stack.len()).step_by(POINTER_SIZE) {

                let mut address = u64::from_le_bytes(stack[i..(i + POINTER_SIZE)].try_into().unwrap());

                while address != 0 && address < memory_len as u64 {
                    if heap.allocated_blocks.contains_key(&address) {
                        num_bytes_collected += heap.allocated_blocks.get(&address).unwrap();
                        self.found_ptrs.insert(address, self.found_flag);
                    }

                    access!(heap.memory.try_read(), memory, {
                        if address as usize + POINTER_SIZE > memory.len() {
                            address = 0;
                            break;
                        }
                        address = u64::from_le_bytes(memory[address as usize..(address as usize + POINTER_SIZE)].try_into().unwrap());
                    }, {
                        error!("Garbage Collector: Failed to read heap due to poisoned lock.");
                        panic!("Poisoned lock.");
                    });
                }
            }
        }

        let mut ptrs_to_remove = Vec::new();

        for (ptr, flag) in self.found_ptrs.iter() {
            if *flag != self.found_flag {
                ptrs_to_remove.push(*ptr);
                heap.free(*ptr);
            }

        }
        for ptr in ptrs_to_remove {
            self.found_ptrs.remove(&ptr);
        }

        self.found_flag = !self.found_flag;

        self.send_message(Message::Success)?;
        info!("Garbage Collector: Finished collection.");
        trace!("Garbage Collector: Freed {} bytes.", num_bytes_collected);
        Ok(false)
    }
}

