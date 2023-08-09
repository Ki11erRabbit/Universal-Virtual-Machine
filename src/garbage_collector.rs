

use std::sync::{RwLock, Arc};

use crate::{Core, Byte, SimpleResult, Message, CoreResult, GC, Collector};
use crate::core::MachineCore;


impl Core for GarbageCollector {
    fn run(&mut self, program_counter: usize) -> SimpleResult {

        Ok(())
    }

    fn run_once(&mut self) -> SimpleResult {
        Ok(())
    }

    fn add_program(&mut self, program: Arc<Vec<Byte>>) {
        self.core.add_program(program);
    }

    fn add_channels(&mut self, machine_send: std::sync::mpsc::Sender<Message>, core_receive: std::sync::mpsc::Receiver<Message>) {
        self.core.add_channels(machine_send, core_receive);
    }

    fn add_memory(&mut self, memory: Arc<RwLock<Vec<Byte>>>) {
        self.core.add_memory(memory);
    }

    fn send_message(&self, message: Message) -> SimpleResult {
        self.core.send_message(message)
    }

    fn recv_message(&self) -> CoreResult<Message> {
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

    fn set_gc(&mut self, gc: bool) {
        panic!("Cannot set garbage collector on garbage collector.");
    }

    fn get_stack(&self) -> Arc<RwLock<Vec<Byte>>> {
        panic!("Cannot get stack from garbage collector.");
    }
}

impl Collector for GarbageCollector {

    fn add_stacks(&mut self, stacks: Arc<RwLock<Vec<Option<Arc<RwLock<Vec<Byte>>>>>>>) {
        self.stacks = stacks;
    }
}

/// A garbage collector for the virtual machine.
pub struct GarbageCollector {
    /// This, with the Core trait, should allow us to do C style inheritance.
    core: MachineCore,
    /// A vector of stacks, this is so that we can have access to the stacks of the core.
    stacks: Arc<RwLock<Vec<Option<Arc<RwLock<Vec<Byte>>>>>>>,
}

impl GarbageCollector {
    pub fn new() -> Self {
        Self {
            core: MachineCore::new(),
            stacks: Arc::new(RwLock::new(Vec::new())),
        }
    }
}

