
use std::sync::{Arc, RwLock};
use std::thread::{self,JoinHandle};
use std::sync::atomic::AtomicU64;
use std::array::from_fn;


pub enum Fault {

}


pub struct Machine {
    heap: Arc<RwLock<Vec<u8>>>,
    cores: Vec<Arc<RwLock<Core>>>,
    core_threads: Vec<JoinHandle<Result<(),Fault>>>,
    program: Arc<RwLock<Vec<u8>>>,
}






pub struct Core {
    /* 64-bit registers */
    registers_64: [u64; 16],
    /* 128-bit registers */
    registers_128: [u128; 8],
    /* floating point registers */
    registers_f32: [f32; 8],
    registers_f64: [f64; 8],
    /* Atomic registers */
    registers_atomic_64: [AtomicU64; 8],
    /* flags */
    parity_flag: bool,
    zero_flag: bool,
    sign_flag: bool,
    /* other */
    program_counter: usize,
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
            stack: Vec::new(),
            pipeline: Vec::new(),
            program,
            memory,
        }
    }
}
