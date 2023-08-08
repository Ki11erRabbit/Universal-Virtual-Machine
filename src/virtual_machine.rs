use std::cell::RefCell;
use std::any::Any;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, RwLock, TryLockError};
use std::thread::{self,JoinHandle};
use std::sync::mpsc::{Sender,Receiver, channel};
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Write;
use std::time::{Instant, Duration};

use crate::{Pointer, CoreId, Byte, RegisterType, Message, Fault, ForeignFunction, ForeignFunctionArg, FileDescriptor};
use crate::binary::Binary;
use crate::core::Core;

/// Struct that contains options for the virtual machine
pub struct MachineOptions {
    /// The max number of cycles to run before resetting the cycle count
    pub cycle_size: usize,
    /// The nth cycle to join threads
    pub join_thread_cycle: usize,
    /// The nth cycle to defrag memory
    pub defrag_cycle: usize,
    /// The time to collect garbage
    /// None for no garbage collection
    /// Time is in minutes
    pub gc_time: Option<u64>,
}

/// A struct that represents our virtual machine
pub struct Machine {
    /// The options for the virtual machine
    options: MachineOptions,
    /// The memory of the virtual machine in little endian
    memory: Arc<RwLock<Vec<Byte>>>,
    /// A map of allocated blocks of memory that are in use
    allocated_blocks: HashMap<Pointer, usize>,
    /// A map of freed blocks of memory that are available for use
    available_blocks: HashMap<Pointer, usize>,
    /// The cores of the virtual machine that are not running
    cores: Vec<Core>,
    /// The thread handles for the running cores
    core_threads: Vec<Option<JoinHandle<Result<(),Fault>>>>,
    /// The program to run
    program: Option<Arc<Vec<Byte>>>,
    /// The entry point of the program for the program counter
    entry_point: Option<usize>,
    /// The channels for the cores to communicate with the machine's event loop
    channels: Rc<RefCell<Vec<Option<(Sender<Message>, Receiver<Message>)>>>>,
    /// The files that are open for the machine
    /// Unfortuately stdin doesn't implement Write so we have to create an offset for the file descriptors when writing
    files: Vec<Option<Box<dyn Write>>>,
    /// A map of child threads to their parent threads
    thread_children: HashMap<CoreId, CoreId>,
    /// A list of threads to join, this is set by the join instruction. This is so that we don't deadlock when trying to join a thread
    threads_to_join: Rc<RefCell<Vec<CoreId>>>,
    /// The id of the main thread
    main_thread_id: CoreId,
    /// A list of foriegn functions that can be called by the program
    foriegn_functions: Vec<(ForeignFunctionArg, ForeignFunction)>,
}

impl Machine {
    /// Creates a new machine without any cores with the default options
    pub fn new() -> Machine {
        Machine {
            options: MachineOptions {
                cycle_size: 1,
                join_thread_cycle: 0,
                defrag_cycle: 0,
                gc_time: Some(0),
            },
            memory: Arc::new(RwLock::new(Vec::new())),
            allocated_blocks: HashMap::new(),
            available_blocks: HashMap::new(),
            cores: Vec::new(),
            core_threads: Vec::new(),
            program: None,
            entry_point: None,
            channels: Rc::new(RefCell::new(Vec::new())),
            files: vec![Some(Box::new(std::io::stdout())), Some(Box::new(std::io::stderr()))],
            thread_children: HashMap::new(),
            threads_to_join: Rc::new(RefCell::new(Vec::new())),
            main_thread_id: 0,
            foriegn_functions: Vec::new(),
        }
    }

    /// Creates a new machine with the default options with a specified number of cores
    pub fn new_with_cores(core_count: usize) -> Machine {
        let mut machine = Machine::new();
        for _ in 0..core_count {
            machine.add_core();
        }
        machine
    }

    /// This is how we set new options for the machine
    pub fn set_options(&mut self, options: MachineOptions) {
        self.options = options;
    }

    /// This is how we add foriegn functions and their arguments to the machine
    pub fn add_function(&mut self, func_arg: ForeignFunctionArg, function: ForeignFunction) {
        self.foriegn_functions.push((func_arg, function));
    }

    /// This function allows us to specify the entry point of the program
    pub fn run_at(&mut self, program_counter: usize) {
        self.entry_point = Some(program_counter);
        self.run();
    }

    /// This function runs the main event loop of the machine
    /// We first spawn the main thread and then we run the main event loop
    /// We keep track of whether or not the main thread is completed. If it is, then we break out of the loop
    /// We keep track of the amount of cycles so that we can perform certain actions on certain nth cycles
    /// We always check for messages from the cores so they don't remain blocked for long.
    /// We also have options for joining joinable threads for the nth cycle and defragging memory for the nth cycle
    /// After the loop ends, we then clear the thread handles and the channels for the threads
    pub fn run(&mut self) {
        //TODO: change print to log
        let program_counter = self.entry_point.expect("Entry point not set");
        self.run_core(0, program_counter);
        let mut main_thread_done = false;
        let mut cycle_count = 0;
        let mut time = match self.options.gc_time {
            Some(_) => Some(Instant::now()),
            None => None,
        };
        loop {
            self.check_main_core(&mut main_thread_done);
            self.check_messages();
            if self.options.join_thread_cycle == cycle_count {
                self.join_joinable_threads();
            }
            if self.options.defrag_cycle == cycle_count {
                self.defrag_memory();
            }

            //TODO: check for commands from the threads to do various things like allocate more memory, etc.
            if main_thread_done {
                break;
            }

            if time.is_some() {
                let time = time.as_mut().unwrap();
                if time.elapsed().as_secs() >= Duration::from_secs(self.options.gc_time.unwrap() * 60).as_secs() {
                    //TODO: send signal to collect garbage
                    *time = Instant::now();
                }
            }

            cycle_count = (cycle_count + 1) % self.options.cycle_size;
        }
        self.core_threads.clear();
        self.channels.borrow_mut().clear();
    }

    /// This function checks to see if the main thread is done and if it is, then we join it and remove it from the channels
    fn check_main_core(&mut self, main_thread_done: &mut bool) {
        if self.core_threads[self.main_thread_id as usize].as_ref().expect("main core doesn't exist").is_finished() {
            *main_thread_done = true;
            self.join_thread(self.main_thread_id);

            self.channels.borrow_mut()[self.main_thread_id as usize] = None;
        }

    }

    /// This function checks to see if there are any threads that are able to be joined that were set to be joined.
    fn join_joinable_threads(&mut self) {
        let mut threads_joined = Vec::new();
        for thread_id in self.threads_to_join.borrow().iter() {
            if self.core_threads[*thread_id as usize].as_ref().expect("core doesn't exist").is_finished() {
                match self.core_threads[*thread_id as usize].take().expect("Already joined this core").join() {
                    Ok(result) => {
                        match result {
                            Ok(_) => eprintln!("Core {} finished", *thread_id),
                            Err(fault) => eprintln!("Core {} faulted with: {}", *thread_id, fault),
                        }

                        match self.thread_children.remove(thread_id) {
                            Some(parent_core_id) => {
                                let parent_channel = self.channels.borrow()[parent_core_id as usize].as_ref().expect("channel no longer exists").0.clone();
                                let message = Message::Success;
                                parent_channel.send(message).unwrap();
                            },
                            None => {},
                        }

                        self.channels.borrow_mut()[*thread_id as usize] = None;
                    },
                    Err(_) => eprintln!("Core {} panicked", *thread_id),
                }
                        threads_joined.push(*thread_id);
                    }
                }

        self.threads_to_join.borrow_mut().retain(|thread_id| !threads_joined.contains(thread_id));
    }

    /// This function merges available blocks that are next to each other
    fn defrag_memory(&mut self) {
        let mut blocks_to_update = Vec::new();
        for (ptr, size) in self.available_blocks.iter() {
            match self.allocated_blocks.get(&(*ptr + *size as u64)) {
                None => {},
                Some(_) => {
                    blocks_to_update.push((*ptr, *ptr + *size as u64));
                },
            }
        }

        for (ptr, next_ptr) in blocks_to_update.iter().rev() {
            let size = self.available_blocks.remove(next_ptr).expect("block no longer exists");
            let ptr_size = self.available_blocks.get(ptr).expect("block no longer exists");
            self.available_blocks.insert(*ptr, ptr_size + size);

        }

    }

    /// This function goes through the channels and checks to see if there are any messages to be processed
    fn check_messages(&mut self) {
        let mut core_id = 0;
        self.channels.clone().borrow().iter().for_each(|channels| {
            if channels.is_none() {
                core_id += 1;
                return;
            }
            let (send, recv) = channels.as_ref().expect("channel no longer exists");
            match recv.try_recv() {
                Ok(message) => {
                    match message {
                        Message::OpenFile(filename, flag) => {
                            let message = self.open_file(filename, flag);
                            send.send(message).unwrap();
                        },
                        Message::WriteFile(fd, data) => {
                            let message = self.write_file(fd, data);
                            send.send(message).unwrap();
                        },
                        Message::CloseFile(fd) => {
                            let message = self.close_file(fd);
                            send.send(message).unwrap();
                        },
                        Message::Flush(fd) => {
                            let message = self.flush(fd);
                            send.send(message).unwrap();
                        },
                        Message::SpawnThread(program_counter) => {
                            let (message, child_id) = self.thread_spawn(program_counter);
                            self.thread_children.insert(child_id, core_id as u8);
                            send.send(message).unwrap();
                        },
                        Message::ThreadDone(_) => {
                            send.send(Message::Success).unwrap();
                        },
                        Message::JoinThread(thread_id) => {
                            self.threads_to_join.borrow_mut().push(thread_id);
                        },
                        Message::DetachThread(thread_id) => {

                            self.core_threads[thread_id as usize].take().expect("Already joined this core");

                            send.send(Message::Success).unwrap();
                        },
                        Message::GetForeignFunction(function_id) => {
                            let (arg, func) = self.foriegn_functions[function_id as usize].clone();
                            send.send(Message::ForeignFunction(arg, func)).unwrap();
                        },
                        Message::Malloc(size) => {
                            let message = self.malloc(size);
                            send.send(message).unwrap();
                        },
                        Message::Free(ptr) => {
                            let message = self.free(ptr);
                            send.send(message).unwrap();
                        },
                        Message::Realloc(ptr, size) => {
                            let message = self.realloc(ptr, size);
                            send.send(message).unwrap();
                        },
                        _ => unimplemented!(),

                    }

                },
                Err(_) => {},
            }

           core_id += 1;
        });

    }

    /// This function is used to allocate memory when the right message is passed in
    /// We try to acquire a lock on the memory and if we can't we try again until we do.
    fn malloc(&mut self, size: u64) -> Message {
        loop {
            match self.memory.try_write() {
                Ok(mut memory) => {

                    let mut block = Vec::new();
                    let mut new_size = 0;
                    let mut new_ptr = 0;
                    for (ptr, block_size) in self.available_blocks.iter() {
                        if *block_size >= size as usize {
                            new_size = *block_size - size as usize;
                            new_ptr = *ptr + size;


                            block.push(*ptr);

                            break;
                        }
                    }

                    if !block.is_empty() {
                        if new_size > 0 {
                            self.available_blocks.insert(new_ptr, new_size);
                        }
                        for ptr in block {
                            self.available_blocks.remove(&ptr);
                            return Message::MemoryPointer(ptr);
                        }
                    }
                    
                    let new_size = memory.len() + size as usize;

                    let ptr = memory.len() as u64;

                    if new_size > memory.len() {
                        memory.resize(new_size, 0);
                    }

                    self.allocated_blocks.insert(ptr, size as usize);

                    return Message::MemoryPointer(ptr);
                }
                Err(TryLockError::WouldBlock) => {
                    continue;
                },
                Err(_) => {
                    panic!("Memory lock poisoned");
                }
            }
        }
    }

    /// This function is used to reallocate the memory of a pointer to a larger or smaller size
    fn realloc(&mut self, ptr: Pointer, size: u64) -> Message {
        if size == 0 {
            return self.free(ptr);
        }

        let old_size = *self.allocated_blocks.get(&ptr).expect("Reallocating Bad Pointer") as u64;
        let mut ret_ptr = None;
        let mut free_ptr = None;
        loop {
            match self.memory.try_write() {
                Ok(mut memory) => {

                    if old_size < size {
                        let try_address = old_size + ptr ;

                        // If we are not already allocated and not free
                        if !self.allocated_blocks.contains_key(&try_address) && !self.available_blocks.contains_key(&try_address) {
                            let difference = size - old_size;

                            let mem_size = memory.len();

                            memory.resize(mem_size + difference as usize, 0);

                            return Message::MemoryPointer(ptr);
                        }// TODO: add in check for if we already have blocks allocated
                        else {

                            let mem_size = memory.len();

                            let new_ptr = mem_size as u64;

                            memory.resize(mem_size + size as usize, 0);

                            let old_memory = memory[ptr as usize..(ptr + old_size) as usize].to_vec();

                            for (index, byte) in old_memory.iter().enumerate() {
                                memory[new_ptr as usize + index] = *byte;
                            }

                            self.allocated_blocks.insert(new_ptr, size as usize);
                            self.allocated_blocks.remove(&ptr);
                            self.available_blocks.insert(ptr, old_size as usize);

                            free_ptr = Some(ptr);
                            ret_ptr = Some(new_ptr);
                            break;
                        }
                    }
                    else {
                        let difference = old_size - size;

                        let mem_size = memory.len();

                        self.allocated_blocks.insert(ptr, size as usize);
                        self.available_blocks.insert(ptr + size, difference as usize);

                        return Message::MemoryPointer(ptr);
                    }
                }
                Err(TryLockError::WouldBlock) => {
                    continue;
                },
                Err(_) => {
                    panic!("Memory lock poisoned");
                }
            }
        }
        if let Some(ptr) = free_ptr {
            self.free(ptr);
        }
        if let Some(ptr) = ret_ptr {
            return Message::MemoryPointer(ptr);
        }
        Message::Error(Fault::InvalidRealloc)
    }

    /// This function take a pointer and frees the memory associated with it
    fn free(&mut self, ptr: Pointer) -> Message {
        if !self.allocated_blocks.contains_key(&ptr) {
            return Message::Error(Fault::InvalidFree);
        }
        self.available_blocks.insert(ptr, self.allocated_blocks.get(&ptr).unwrap().clone());
        self.allocated_blocks.remove(&ptr);
        Message::Success
    }

    /// This function will mark a thread for joining and then wait to send the right message back to block the calling thread
    fn join_thread(&mut self, thread_id: CoreId) {
        let core_id = thread_id as usize;

        match self.core_threads[core_id].take().expect("Already joined this core").join() {
            Ok(result) => {
                match result {
                    Ok(_) => eprintln!("Core {} finished", core_id),
                    Err(fault) => eprintln!("Core {} faulted with: {}", core_id, fault),
                }
            },
            Err(_) => eprintln!("Core {} panicked", core_id),
        }
    }

    /// This function writes bytes to a file based on the file descriptor
    fn write_file(&mut self, fd: FileDescriptor, data: Vec<u8>) -> Message {
        let fd = fd - 1;
        if fd as usize >= self.files.len() {
            return Message::Error(Fault::InvalidFileDescriptor);
        }
        let file = match &mut self.files[fd as usize] {
            Some(file) => file,
            None => return Message::Error(Fault::InvalidFileDescriptor),
        };
        match file.write(&data) {
            Ok(_) => Message::Success,
            Err(_) => Message::Error(Fault::FileWriteError),
        }
    }

    /// This function will flush the file based on the file descriptor
    fn flush(&mut self, fd: FileDescriptor) -> Message {
        let fd = fd as usize - 1;
        if fd as usize >= self.files.len() {
            return Message::Error(Fault::InvalidFileDescriptor);
        }
        let file = match &mut self.files[fd as usize] {
            Some(file) => file,
            None => return Message::Error(Fault::InvalidFileDescriptor),
        };
        match file.flush() {
            Ok(_) => Message::Success,
            Err(_) => Message::Error(Fault::FileWriteError),
        }
    }

    /// This function will close the file based on the file descriptor
    /// Trying to remove stdout or stderr will result in weird behavior
    fn close_file(&mut self, fd: FileDescriptor) -> Message {
        let fd = fd - 1;
        if fd as usize >= self.files.len() {
            return Message::Error(Fault::InvalidFileDescriptor);
        }

        self.files[fd as usize] = None;
        Message::Success
    }

    /// This function will open a file based on the filename and the flag
    /// The flag is just a char that is either 'r', 'w', 'a', or 't'
    fn open_file(&mut self, filename: Vec<u8>, flag: u8) -> Message {
        let mut file = OpenOptions::new();
        let file = file.create(true);
        let file = match flag as char {
            't' => file.truncate(true),
            'a' => file.append(true),
            'w' => file.write(true),
            'r' | _ => file.read(true),
        };

        let filename = String::from_utf8(filename);
        match filename {
            Ok(filename) => {
                match file.open(filename) {
                    Ok(file) => {
                        self.files.push(Some(Box::new(file)));
                        Message::FileDescriptor(self.files.len() as FileDescriptor)
                    },
                    Err(_) => Message::Error(Fault::FileOpenError),
                }
            },
            Err(_) => Message::Error(Fault::InvalidMessage),
        }
    }

    /// This function will run a core at the given program counter
    fn thread_spawn(&mut self, program_counter: u64) -> (Message, u8) {
        if self.cores.len() == 0 {
            self.add_core();
        }
        self.run_core_threaded(0, program_counter as usize);
        let core_id = self.core_threads.len() - 1;
        (Message::ThreadSpawned(core_id as u8), core_id as u8)
    }

    /// This function adds a core to the machine
    pub fn add_core(&mut self) {
        //TODO: Make it so that we don't panic from trying to add another core while running
        let (core_sender, core_receiver) = channel();
        let (machine_sender, machine_receiver) = channel();
        self.channels.borrow_mut().push(Some((core_sender, machine_receiver)));
        let core = Core::new(self.memory.clone(), machine_sender, core_receiver,);
        self.cores.push(core);
    }

    /// This function will run a core at a given program counter in a new thread
    fn run_core_threaded(&mut self, core: usize, program_counter: usize) {
        let mut core = self.cores.remove(core);
        let mut program = (**self.program.as_ref().expect("Program Somehow not set").clone()).to_vec();
        let new_pc = program.len();
        program.push(109);
        program.push(0);
        program.extend_from_slice(&program_counter.to_le_bytes());
        program.push(162);
        program.push(0);
        let program = Arc::new(program);
        core.add_program(program);
        let core_thread = {
            thread::spawn(move || {
                core.run(new_pc)
            })
        };
        self.core_threads.push(Some(core_thread));
    }

    /// This function is for running the first core
    pub fn run_core(&mut self, core: usize, program_counter: usize) {
        let mut core = self.cores.remove(core);
        core.set_main_thread();
        core.add_program(self.program.as_ref().expect("Program Not set").clone());
        let core_thread = {
            thread::spawn(move || {
                core.run(program_counter)
            })
        };
        self.core_threads.push(Some(core_thread));
    }

    /// This function will add a program to the machine
    pub fn add_program(&mut self, program: Vec<Byte>) {
        self.program = Some(Arc::new(program));
    }

    /// This function will get the total number of cores
    pub fn core_count(&self) -> usize {
        self.core_threads.len()
    }

    /// This function will load a binary into the machine
    /// A binary is just a struct that contains the program, data segment, and entry point
    pub fn load_binary(&mut self, binary: &Binary) {
        self.program = Some(Arc::new(binary.program().clone()));
        self.memory.write().unwrap().extend_from_slice(&binary.data_segment());
        self.entry_point = Some(binary.entry_address());
    }



}

#[cfg(test)]
mod tests {
    use crate::assembler::generate_binary;
    use std::io::Read;
    use std::time::SystemTime;

    use super::*;

    #[test]
    fn test_file_hello_world() {
        let input = "File{
.string \"hello.txt\"}
Msg{
.string \"Hello World!\"}
main{
move 64, $0, File
move 64, $1, 9u64
move 8, $2, 119
move 64, $3, Msg
move 64, $4, 12u64
open $0, $1, $2, $5
write $5, $3, $4
flush $5
close $5
ret
}
";
        let binary = generate_binary(input, "test").unwrap();

        let mut machine = Machine::new();

        machine.load_binary(&binary);

        machine.add_core();

        machine.run();

        assert_eq!(machine.files.len(), 2);
        let mut file = File::open("hello.txt").unwrap();

        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();

        assert_eq!(contents, "Hello World!");
    }

    #[test]
    fn test_dp_fibonacci() {
        let input = "array{
.u32 0u32
.u32 1u32
.u32 0u32
.u32 0u32
.u32 0u32
.u32 0u32
.u32 0u32
.u32 0u32
.u32 0u32
.u32 0u32
.u32 0u32
.u32 0u32
.u32 0u32
.u32 0u32
.u32 0u32
.u32 0u32
.u32 0u32
.u32 0u32
.u32 0u32}
main{
move 64, $0, array ; pointer to first element
move 64, $1, 4u64
addu 64, $0, $1
addu 64, $0, $1 ; pointer is now in 3rd element
move 32, $3, 10u32; end condition 
move 32, $4, 2u32; i variable
move 32, $10, 1u32
}
loop{
equ 32, $3, $4
jumpeq end
move 32, $5, $0, -4i64
move 32, $6, $0, -8i64
addu 32, $5, $6
move 32, $0, $5
addu 64, $0, $1
addu 32, $4, $10
jump loop
}
end{
ret}
";

        let now = SystemTime::now();
        let binary = generate_binary(input, "test").unwrap();

        let mut machine = Machine::new();

        machine.load_binary(&binary);

        machine.add_core();

        machine.run();

        println!("Time: {:?}", now.elapsed().unwrap());

        assert_eq!(machine.memory.read().unwrap()[1..], [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 5, 0, 0, 0, 8, 0, 0, 0, 13, 0, 0, 0, 21, 0, 0, 0, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_recursive_fibonacci() {
        /*let input = "number{
.u32 0u32}
fibonacci{
move 32, $5, 1u32
lequ 32, $5, $1
jumpeq before-end
push 32, $1
subu 32, $1, $5
call fibonacci
pop 32, $1
subu 32, $1, $5
subu 32, $1, $5
push 32, $0
call fibonacci
pop 32, $1
addu 32, $0, $1
jump end}
before-end{
move $0, $1, 32}
end{
ret}
main{
move 32, $1, 5u32
call fibonacci
move 64, $3, number
move 32, $3, $0
ret
}
";*/
        let input = "number{
.u32 0u32}
fibonacci{
move 32, $2, 1u32
lequ 32, $1, $2
jumpgt rec
move $0, $1, 32
jump end
}
rec{
push 32, $1
subu 32, $1, $2
call fibonacci
pop 32, $1
push 32, $0
move 32, $2, 2u32
subu 32, $1, $2
call fibonacci
pop 32, $1
addu 32, $0, $1}
end{
ret}
main{
move 32, $1, 9u32
call fibonacci
move 64, $3, number
move 32, $3, $0
ret
}
";
        let now = SystemTime::now();
        let binary = generate_binary(input, "test").unwrap();

        let mut machine = Machine::new();

        machine.load_binary(&binary);

        //println!("{}", binary.assembly());
        //println!("{}", binary.program_with_count());

        machine.add_core();

        machine.run();

        println!("Time: {:?}", now.elapsed().unwrap());

        assert_eq!(machine.memory.read().unwrap()[1..5], [34, 0, 0, 0]);
        
    }

    #[test]
    fn test_multicore() {
        let input = "counter{
.u64 0u64}
count{
move 64, $0, counter
move 64, $1, 0u64
move 64, $2, 10u64
}
loop{
move 64, $3, 1u64
equ 64, $1, $2
jumpeq end
addu 64, $1, $3
jump loop
}
end{
move 64, $0, $1
move 8, $10, 68
move 64, $11, 1i64
writebyte $11, $10
flush $11
ret}
main{
move 64, $0, count
threadspawn $0, $1
threadspawn $0, $2
threadjoin $1
threadjoin $2
ret}
";

        let binary = generate_binary(input, "test").unwrap();

        let mut machine = Machine::new();

        println!("{}", binary.assembly());
        println!("{}", binary.program_with_count());

        machine.load_binary(&binary);


        machine.add_core();
        machine.add_core();
        machine.add_core();

        machine.run();


        assert_eq!(machine.memory.read().unwrap()[1..9], [10, 0, 0, 0, 0, 0, 0, 0]);


    }


    fn simple_mutation(core: &mut Core, _args: Option<Arc<RwLock<dyn Any + Send + Sync>>>) -> Result<(),Fault> {

        let reg = core.get_register_64(0)?;

        *reg += 10;


        Ok(())
    }

    #[test]
    fn test_foreign_function() {
        let input = "mem{
.u64 0u64}
main{
move 64, $0, 35u64
foreign $1
move 64, $1, mem
move 64, $1, $0
ret}";

        let binary = generate_binary(input, "test").unwrap();

        let mut machine = Machine::new();

        machine.add_function(None, simple_mutation);

        machine.load_binary(&binary);

        machine.add_core();

        machine.run();

        assert_eq!(machine.memory.read().unwrap()[1..9], [45, 0, 0, 0, 0, 0, 0, 0]);

    }

    fn complex_mutation(core: &mut Core, args: Option<Arc<RwLock<dyn Any + Send + Sync>>>) -> Result<(),Fault> {

        let reg = core.get_register_64(0)?;

        let binding = args.unwrap();
        let mut binding = binding.write().unwrap();
        let value = binding.downcast_mut::<u64>().unwrap();

        *value += *reg;


        Ok(())
    }

    #[test]
    fn test_foreign_function_with_arg() {
        let input = "mem{
.u64 0u64}
main{
move 64, $0, 35u64
move 64, $1, mem
move 64, $2, 0u64
foreign $2
move 64, $1, $0
ret}";

        let binary = generate_binary(input, "test").unwrap();

        let mut machine = Machine::new();

        let argument: Option<Arc<RwLock<dyn Any + Send + Sync>>> = Some(Arc::new(RwLock::new(10u64)));

        machine.add_function(argument.clone(), complex_mutation);
        
        machine.load_binary(&binary);

        machine.add_core();

        machine.run();

        assert_eq!(machine.memory.read().unwrap()[1..9], [35, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(argument.unwrap().read().unwrap().downcast_ref::<u64>().unwrap(), &45u64);

    }

    #[test]
    fn test_allocation() {
        let input = "
main{
move 64, $0, 64u64
malloc $1, $0
move 64, $2, 10u64
move 64, $1, $2
ret}
";
        let binary = generate_binary(input, "test").unwrap();

        let mut machine = Machine::new();

        machine.load_binary(&binary);

        machine.add_core();

        machine.run();

        assert_eq!(machine.memory.read().unwrap()[1..9], [10, 0, 0, 0, 0, 0, 0, 0]);

    }

    #[test]
    fn test_free_allocation() {
        let input = "
main{
move 64, $0, 64u64
malloc $1, $0
move 64, $2, 10u64
move 64, $1, $2
free $1
ret}
";
        let binary = generate_binary(input, "test").unwrap();

        let mut machine = Machine::new();

        machine.load_binary(&binary);

        machine.add_core();

        machine.run();

        assert_eq!(machine.memory.read().unwrap()[1..9], [10, 0, 0, 0, 0, 0, 0, 0]);

    }

    #[test]
    fn test_reallocation() {
        let input = "
main{
move 64, $0, 8u64
malloc $1, $0
move 64, $2, 10u64
move 64, $1, $2
move 64, $0, 16u64
realloc $1, $0
move 64, $2, 8u64
addu 64, $1, $2
move 64, $1, $2
ret}
";
        let binary = generate_binary(input, "test").unwrap();

        let mut machine = Machine::new();

        println!("{}", binary.assembly());
        println!("{}", binary.program_with_count());
        
        machine.load_binary(&binary);

        machine.add_core();

        machine.run();

        println!("{:?}", machine.memory.read().unwrap());

        assert_eq!(machine.memory.read().unwrap()[1..17], [10, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0 ,0, 0, 0, 0, 0]);

    }

}
