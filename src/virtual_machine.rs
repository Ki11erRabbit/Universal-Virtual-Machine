use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, RwLock};
use std::thread::{self,JoinHandle};
use std::sync::mpsc::{Sender,Receiver, channel};
use std::fs::OpenOptions;
use std::io::{Write, Read, self};
use std::time::{Instant, Duration};

#[cfg(not(test))]
use log::{trace, info, error};
#[cfg(test)]
use std::{println as trace, println as info, println as error};


use crate::core::MachineCore;
use crate::garbage_collector::GarbageCollector;
use crate::{Pointer, CoreId, Byte, Message, Fault, ForeignFunction, ForeignFunctionArg, FileDescriptor, Core, SimpleResult, GarbageCollectorCore, RegCore, Collector, ReadWrite, WholeStack, get_heap_len_panic, Registers, access_mut};
use crate::binary::Binary;

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
    /// The program for the garbage collector to run
    pub gc_program: Option<Arc<Vec<Byte>>>,
    /// The size of the stack in kilobytes
    pub stack_size: usize,
    /// A scaler value for the stack size
    pub stack_scale: usize,
}

impl MachineOptions {
    pub fn calculate_stack_size(&self) -> usize {
        self.stack_size * self.stack_scale * 1024
    }
}

#[derive(Debug)]
pub struct Memory {
    /// The memory of the virtual machine in little endian
    pub memory: Arc<RwLock<Vec<Byte>>>,
    /// A map of allocated blocks of memory that are in use
    pub allocated_blocks: HashMap<Pointer, usize>,
    /// A map of freed blocks of memory that are available for use
    pub available_blocks: HashMap<Pointer, usize>,
}

impl Memory {

    pub fn new() -> Self {
        Memory {
            memory: Arc::new(RwLock::new(Vec::new())),
            allocated_blocks: HashMap::new(),
            available_blocks: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        let x;
        get_heap_len_panic!(self.memory,x);
        x
    }
    
    /// This function is used to allocate memory when the right message is passed in
    /// We try to acquire a lock on the memory and if we can't we try again until we do.
    pub fn malloc(&mut self, size: u64) -> Message {

        access_mut!(self.memory.try_write(), memory, {

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
        },{
            panic!("Memory lock poisoned");
        });
    }

    /// This function is used to reallocate the memory of a pointer to a larger or smaller size
    pub fn realloc(&mut self, ptr: Pointer, size: u64) -> Message {
        if size == 0 {
            return self.free(ptr);
        }

        let old_size = *self.allocated_blocks.get(&ptr).expect("Reallocating Bad Pointer") as u64;
        let mut ret_ptr = None;
        let mut free_ptr = None;
        access_mut!(self.memory.try_write(), memory, {
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
                    
                    //let mem_size = memory.len();
                    
                    self.allocated_blocks.insert(ptr, size as usize);
                    self.available_blocks.insert(ptr + size, difference as usize);

                    return Message::MemoryPointer(ptr);
                }
        },{
            panic!("Memory lock poisoned");
        });
        if let Some(ptr) = free_ptr {
            self.free(ptr);
        }
        if let Some(ptr) = ret_ptr {
            return Message::MemoryPointer(ptr);
        }
        Message::Error(Fault::InvalidRealloc)
    }

    /// This function take a pointer and frees the memory associated with it
    pub fn free(&mut self, ptr: Pointer) -> Message {
        if !self.allocated_blocks.contains_key(&ptr) {
            return Message::Error(Fault::InvalidFree);
        }
        self.available_blocks.insert(ptr, self.allocated_blocks.get(&ptr).unwrap().clone());
        self.allocated_blocks.remove(&ptr);
        Message::Success
    }

    /// This function merges available blocks that are next to each other
    fn defrag_memory(&mut self) {
        let mut blocks_to_update = Vec::new();
        for (ptr, size) in self.available_blocks.iter() {
            match self.available_blocks.get(&(*ptr + *size as u64)) {
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
}

/// A struct that represents our virtual machine
pub struct Machine {
    /// The options for the virtual machine
    options: MachineOptions,
    data_segment: Arc<Vec<Byte>>,
    stack: WholeStack,
    /// The memory of the virtual machine in little endian
    heap: Arc<RwLock<Memory>>,
    /// The cores of the virtual machine that are not running
    cores: Vec<Option<Box<dyn RegCore + Send>>>,
    /// The thread handles for the running cores
    core_threads: Vec<Option<JoinHandle<SimpleResult>>>,
    /// The program to run
    //program: Option<Arc<Vec<Byte>>>,
    ///// The entry point of the program for the program counter
    entry_point: Option<usize>,
    /// The channels for the cores to communicate with the machine's event loop
    channels: Rc<RefCell<Vec<Option<(Sender<Message>, Receiver<Message>)>>>>,
    /// The files that are open for the machine
    /// Unfortuately stdin doesn't implement Write so we have to create an offset for the file descriptors when writing
    files: Vec<Option<Box<dyn ReadWrite>>>,
    /// A map of child threads to their parent threads
    thread_children: HashMap<CoreId, CoreId>,
    /// A list of threads to join, this is set by the join instruction. This is so that we don't deadlock when trying to join a thread
    threads_to_join: Rc<RefCell<Vec<CoreId>>>,
    /// The id of the main thread
    main_thread_id: CoreId,
    /// A list of foriegn functions that can be called by the program
    foriegn_functions: Vec<(ForeignFunctionArg, ForeignFunction)>,
    gc: Option<Result<Box<dyn GarbageCollectorCore + Send>, JoinHandle<SimpleResult>>>,
    gc_channels: Option<(Sender<Message>, Receiver<Message>)>,
}

impl Machine {
    /// Creates a new machine without any cores with the default options
    pub fn new() -> Machine {
        Machine {
            options: MachineOptions {
                cycle_size: 1,
                join_thread_cycle: 0,
                defrag_cycle: 0,
                gc_time: None,
                gc_program: None,
                stack_size: 1,
                stack_scale: 512,
            },
            data_segment: Arc::new(Vec::new()),
            stack: Arc::new(Vec::new()),
            heap: Arc::new(RwLock::new(Memory::new())),
            cores: Vec::new(),
            core_threads: Vec::new(),
            //program: None,
            entry_point: None,
            channels: Rc::new(RefCell::new(Vec::new())),
            files: vec![None, None, None],
            thread_children: HashMap::new(),
            threads_to_join: Rc::new(RefCell::new(Vec::new())),
            main_thread_id: 0,
            foriegn_functions: Vec::new(),
            gc: None,
            gc_channels: None,
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

        if self.options.gc_time.is_some() {
            self.add_gc_core();
        }
        
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
        info!("Machine: Running machine");
        trace!("Machine: Stack size: {}", self.options.calculate_stack_size() * self.cores.len());
        let stack: WholeStack = Arc::new(vec![Arc::new(RwLock::new(vec![0; self.options.calculate_stack_size()].into_boxed_slice())); self.cores.len()]);

        self.stack = stack;

        let program_counter = self.entry_point.expect("Entry point not set");
        self.run_core(0, program_counter);
        if self.options.gc_time.is_some() {
            self.run_gc();
        }
        let mut main_thread_done = false;
        let mut cycle_count = 0;
        let mut time = match self.options.gc_time {
            Some(_) => Some(Instant::now()),
            None => None,
        };
        loop {
            self.check_main_core(&mut main_thread_done);
            self.check_messages();
            self.replenish_cores();
            if self.options.join_thread_cycle == cycle_count {
                self.join_joinable_threads();
            }
            if self.options.defrag_cycle == cycle_count {
                self.heap.write().unwrap().defrag_memory();
            }

            if main_thread_done {
                info!("Main thread done");
                break;
            }

            if time.is_some() {
                let time = time.as_mut().unwrap();
                if time.elapsed().as_secs() >= Duration::from_secs(self.options.gc_time.unwrap() * 60).as_secs() {
                    info!("Starting garbage collection");
                    let message = Message::CollectGarbage;
                    let mut stack_pointers = Vec::new();

                    for pair in self.channels.borrow().iter() {
                        if let Some((sender, reciever)) = pair {
                            sender.send(message.clone()).unwrap();
                            loop {
                                match reciever.recv().unwrap() {
                                    Message::StackPointer(stack_pointer) => {
                                        stack_pointers.push(stack_pointer);
                                        break;
                                    },
                                    msg => sender.send(Message::RetryMessage(Box::new(msg))).unwrap(),
                                }
                            }
                        }
                    }

                    let message = Message::StackPointers(stack_pointers);
                    
                    self.gc_channels.as_ref().unwrap().0.send(message).unwrap();

                    let message = self.gc_channels.as_ref().unwrap().1.recv().unwrap();

                    match message {
                        Message::Success => {
                            trace!("Garbage collection successful");
                        },
                        _ => {
                            error!("Machine: Unexpected message from garbage collector");
                            panic!("Unexpected message from garbage collector")
                        },
                    }

                    for pair in self.channels.borrow().iter() {
                        if let Some((sender, _)) = pair {
                            sender.send(Message::Success).unwrap();
                        }
                    }
                    
                    *time = Instant::now();
                }
            }

            cycle_count = (cycle_count + 1) % self.options.cycle_size;
        }
        self.core_threads.clear();
        self.channels.borrow_mut().clear();
    }

    fn replenish_cores(&mut self) {
        let mut cores_to_add = Vec::new();
        for i in 0..self.cores.len() {
            if self.cores[i].is_none() && self.core_threads[i].is_none() {
                info!("Machine: Replenishing core {}", i);
                cores_to_add.push(i);
            }
        }
        for core in cores_to_add {
            self.add_core_at(core);
        }
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
        info!("Joining joinable threads");
        let mut threads_joined = Vec::new();
        for thread_id in self.threads_to_join.borrow().iter() {
            if self.core_threads[*thread_id as usize].as_ref().expect("core doesn't exist").is_finished() {
                match self.core_threads[*thread_id as usize].take().expect("Already joined this core").join() {
                    Ok(result) => {
                        match result {
                            Ok(_) => info!("Core {} finished", *thread_id),
                            Err(fault) => error!("Core {} faulted with: {}", *thread_id, fault),
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
                    Err(_) => error!("Core {} panicked", *thread_id),
                }
                        threads_joined.push(*thread_id);
                    }
                }

        self.threads_to_join.borrow_mut().retain(|thread_id| !threads_joined.contains(thread_id));
    }


    /// This function goes through the channels and checks to see if there are any messages to be processed
    fn check_messages(&mut self) {
        info!("Machine: Checking messages from cores");
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
                        Message::SpawnThread(program_counter, registers) => {
                            let (message, child_id) = self.thread_spawn(program_counter, registers);
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

                            access_mut!(self.heap.try_write(), memory, {
                                let message = memory.malloc(size);
                                send.send(message).unwrap();
                                break;
                            }, {
                                error!("Machine: Malloc failed due to Memory Corruption due to a poisoned lock");
                                panic!("Memory Corrupted")
                            });
                            
                        },
                        Message::Free(ptr) => {
                            
                            access_mut!(self.heap.try_write(), memory, {
                                let message = memory.free(ptr);
                                send.send(message).unwrap();
                                break;
                            }, {
                                error!("Machine: Free failed due to Memory Corruption due to a poisoned lock");
                                panic!("Memory Corrupted")
                            });
                        },
                        Message::Realloc(ptr, size) => {
                            access_mut!(self.heap.try_write(), memory, {
                                let message = memory.realloc(ptr, size);
                                send.send(message).unwrap();
                                break;
                            }, {
                                error!("Machine: Realloc failed due to Memory Corruption due to a poisoned lock");
                                panic!("Memory Corrupted")
                            });
                        },
                        Message::ReadFile(fd, size) => {
                            let message = self.read_file(fd, size as usize);
                            send.send(message).unwrap();
                        },
                        message => {
                            error!("Machine: Message {:?}, not implemented", message);
                            unimplemented!()
                        },

                    }

                },
                Err(_) => {},
            }

           core_id += 1;
        });

    }


    /// This function will mark a thread for joining and then wait to send the right message back to block the calling thread
    fn join_thread(&mut self, thread_id: CoreId) {
        info!("Machine: Joining thread {}", thread_id);
        let core_id = thread_id as usize;

        match self.core_threads[core_id].take().expect("Already joined this core").join() {
            Ok(result) => {
                match result {
                    Ok(_) => info!("Machine: Core {} finished", core_id),
                    Err(fault) => error!("Machine: Core {} faulted with: {}", core_id, fault),
                }
            },
            Err(_) => error!("Machine: Core {} panicked", core_id),
        }
    }

    fn read_file(&mut self, fd: FileDescriptor, amount: usize) -> Message {
        info!("Machine: Reading {} bytes from file {}", amount, fd);
        if fd as usize >= self.files.len() {
            return Message::Error(Fault::InvalidFileDescriptor);
        }

        match fd {
            0 => {
                let mut stdin = io::stdin();
                let mut buffer: Vec<u8> = vec![0; amount];
                match stdin.read(&mut buffer) {
                    Ok(size) => {
                        return Message::FileData(buffer, size as u64);
                    },
                    Err(_) => return Message::Error(Fault::FileReadError),
                }
            },
            1 => return Message::Error(Fault::InvalidFileDescriptor),
            2 => return Message::Error(Fault::InvalidFileDescriptor),
            _ => {},
        }

        let file = match &mut self.files[fd as usize] {
            Some(file) => file,
            None => return Message::Error(Fault::InvalidFileDescriptor),
        };

        let mut buffer: Vec<u8> = vec![0; amount];

        match file.read(&mut buffer) {
            Ok(size) => {
                return Message::FileData(buffer, size as u64);
            },
            Err(_) => return Message::Error(Fault::FileReadError),
        }
    }

    /// This function writes bytes to a file based on the file descriptor
    fn write_file(&mut self, fd: FileDescriptor, data: Vec<u8>) -> Message {
        info!("Machine: Writing to file {}", fd);
        trace!("Machine: Writing {} to file {}", String::from_utf8(data.clone()).unwrap(), fd);
        if fd as usize >= self.files.len() {
            return Message::Error(Fault::InvalidFileDescriptor);
        }

        match fd {
            0 => return Message::Error(Fault::InvalidFileDescriptor),
            1 => {
                let mut stdout = io::stdout();
                match stdout.write(&data) {
                    Ok(_) => return Message::Success,
                    Err(_) => return Message::Error(Fault::FileWriteError),
                }
            },
            2 => {
                let mut stderr = io::stderr();
                match stderr.write(&data) {
                    Ok(_) => return Message::Success,
                    Err(_) => return Message::Error(Fault::FileWriteError),
                }
            },
            _ => {},
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
        info!("Machine: Flushing file: {}", fd);
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
        info!("Machine: Closing file: {}", fd);
        if fd as usize >= self.files.len() {
            return Message::Error(Fault::InvalidFileDescriptor);
        }

        self.files[fd as usize] = None;
        Message::Success
    }

    /// This function will open a file based on the filename and the flag
    /// The flag is just a char that is either 'r', 'w', 'a', or 't'
    fn open_file(&mut self, filename: Vec<u8>, flag: u8) -> Message {
        info!("Machine: Opening file as fd: {}", self.files.len());
        trace!("Machine: Opening file: {} with flag: {}", String::from_utf8(filename.clone()).unwrap(), flag as char);
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
                        Message::FileDescriptor((self.files.len() - 1) as FileDescriptor)
                    },
                    Err(_) => Message::Error(Fault::FileOpenError),
                }
            },
            Err(_) => Message::Error(Fault::InvalidMessage),
        }
    }

    /// This function will run a core at the given program counter
    fn thread_spawn(&mut self, program_counter: u64, registers: Registers) -> (Message, u8) {
        info!("Machine: Spawning thread");
        if self.cores.len() == 0 {
            self.add_core();
        }
        self.run_core_threaded(0, program_counter as usize, registers);
        let core_id = self.core_threads.len() - 1;
        (Message::ThreadSpawned(core_id as u8), core_id as u8)
    }

    /// This function adds a core to the machine
    pub fn add_core(&mut self) {
        info!("Machine: Adding core");
        let (core_sender, core_receiver) = channel();
        let (machine_sender, machine_receiver) = channel();
        self.channels.borrow_mut().push(Some((core_sender, machine_receiver)));
        let mut core = Box::new(MachineCore::new());
        core.add_data_segment(self.data_segment.clone());
        core.add_heap(self.heap.read().unwrap().memory.clone());
        core.add_channels(machine_sender, core_receiver);
        self.cores.push(Some(core));
        self.core_threads.push(None);
    }

    pub fn add_core_at(&mut self, index: usize) {
        if index >= self.cores.len() {
            error!("Machine: Tried adding a core at an index that is out of bounds");
            panic!("Index out of bounds");
        }

        if !self.cores[index].is_none() && !self.core_threads[index].is_none() {
            return;
        }
        
        let (core_sender, core_receiver) = channel();
        let (machine_sender, machine_receiver) = channel();
        self.channels.borrow_mut()[index].replace((core_sender, machine_receiver));
        let mut core = Box::new(MachineCore::new());
        core.add_data_segment(self.data_segment.clone());
        core.add_heap(self.heap.read().unwrap().memory.clone());
        core.add_channels(machine_sender, core_receiver);
        self.cores[index].replace(core);
    }

    /// This function adds a garbage collector core to the machine
    fn add_gc_core(&mut self) {
        let (core_sender, core_receiver) = channel();
        let (machine_sender, machine_receiver) = channel();
        self.gc_channels = Some((core_sender, machine_receiver));
        let mut core = Box::new(GarbageCollector::new());
        core.add_heap(self.heap.clone());
        core.add_channels(machine_sender, core_receiver);
        self.gc = Some(Ok(core));
    }

    /// This function will run a core at a given program counter in a new thread
    fn run_core_threaded(&mut self, core_index: usize, program_counter: usize,registers: Registers) {
        info!("Machine: Running core {} in a new thread", core_index);
        let mut core = self.cores[core_index].take().unwrap();
        /*let mut program = (**self.program.as_ref().expect("Program Somehow not set").clone()).to_vec();
        let new_pc = program.len();
        program.push(109);
        program.push(0);
        program.extend_from_slice(&program_counter.to_le_bytes());
        program.push(162);
        program.push(0);
        let program = Arc::new(program);
        core.add_program(program);*/

        self.stack[core_index].write().expect("Stack blocked for some reason")[0..8].copy_from_slice(&self.data_segment.len().to_le_bytes());

        core.add_stack(self.stack.clone(), core_index);

        core.set_registers(registers);
        
        let core_thread = {
            thread::spawn(move || {
                core.run(program_counter)
            })
        };
        self.core_threads[core_index].replace(core_thread);
    }

    /// This function is for running the first core
    pub fn run_core(&mut self, core_index: usize, program_counter: usize) {
        info!("Machine: Running core {} in a new thread", core_index);
        let mut core = self.cores[core_index].take().unwrap();
        //core.add_program(self.program.as_ref().expect("Program Not set").clone());

        core.add_stack(self.stack.clone(), core_index);
        
        let core_thread = {
            thread::spawn(move || {
                core.run(program_counter)
            })
        };
        self.core_threads[core_index].replace(core_thread);
    }

    /// This function runs the garbage collector
    pub fn run_gc(&mut self) {
        info!("Machine: Running GC");
        let gc = self.gc.take().unwrap();

        let (core_sender, core_receiver) = channel();
        let (machine_sender, machine_receiver) = channel();

        

        match gc {
            Ok(mut gc) => {
                if self.options.gc_program.is_some() {
                    gc.add_program(self.options.gc_program.clone().unwrap());
                }
                gc.add_data_segment(self.data_segment.clone());
                gc.add_heap(self.heap.clone());
                gc.add_stack(self.stack.clone());
                gc.add_channels(machine_sender, core_receiver);

                self.gc_channels = Some((core_sender, machine_receiver));
                
                let gc_thread = {
                    thread::spawn(move || {
                        gc.run(0)
                    })
                };
                self.gc = Some(Err(gc_thread));
            },
            Err(_) => {
                error!("Machine: Tried running GC when it should already be running");
                panic!("GC already running")
            },
            
        }

    }

    /// This function will add a program to the machine
    pub fn add_program(&mut self, program: Vec<Byte>) {
        info!("Machine: Adding program");
        //self.program = Some(Arc::new(program));
    }

    /// This function will get the total number of cores
    pub fn core_count(&self) -> usize {
        self.core_threads.len()
    }

    /// This function will load a binary into the machine
    /// A binary is just a struct that contains the program, data segment, and entry point
    pub fn load_binary(&mut self, binary: &Binary) {
        info!("Machine: Loading binary");
        let segment = Arc::new(binary.program().clone());
        //let segment = binary.data_segment().clone();
        //self.data_segment = Arc::new(segment);
        self.data_segment = segment;
        self.entry_point = Some(binary.entry_address());
    }



}

#[cfg(test)]
mod tests {
    use crate::assembler::generate_binary;
    use std::io::Read;
    use std::time::SystemTime;
    use std::fs::File;

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

        assert_eq!(machine.files.len(), 4);
        let mut file = File::open("hello.txt").unwrap();

        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();

        assert_eq!(contents, "Hello World!");
    }

    #[test]
    fn test_dp_fibonacci() {
        let input = "
main{
move 32, $0, 0u32 ; first element
move 32, $1, 1u32 ; second element
add 32, $2, 0u32 ; temp
move 64, $3, 2u64 ; counter
}
loop{
eq 64, $3, 10u64
jumpeq end
move 32, $2, 0u32
add 32, $2, $0
add 32, $2, $1
move $0, $1, 32
move $1, $2, 32
add 64, $3, 1u64
jump loop
}
end{
move 64, $2, 4u64
malloc $4, $2
move 32, $4, $1
ret}
";

        let now = SystemTime::now();
        let binary = generate_binary(input, "test").unwrap();

        let mut machine = Machine::new();

        machine.load_binary(&binary);

        println!("{}", binary.program_with_count());

        machine.add_core();

        machine.run();

        println!("Time: {:?}", now.elapsed().unwrap());

        assert_eq!(machine.heap.read().unwrap().len(), 4);
        assert_eq!(machine.heap.read().unwrap().memory.read().unwrap()[0..], [34,0,0,0]);

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
        let input = "
fibonacci{
move 32, $2, 1u32
leq 32, $1, $2
jumpgt rec
move $0, $1, 32
jump end
}
rec{
push 32, $1
sub 32, $1, $2
call fibonacci
pop 32, $1
push 32, $0
move 32, $2, 2u32
sub 32, $1, $2
call fibonacci
pop 32, $1
add 32, $0, $1}
end{
ret}
main{
move 32, $1, 9u32
call fibonacci
move 64, $4, 4u64
malloc $3, $4
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

        assert_eq!(machine.heap.read().unwrap().memory.read().unwrap()[..], [34, 0, 0, 0]);
        
    }

    #[test]
    //TODO: Fix this test by adding a way to pass arguments into threads
    fn test_multicore() {
        let input = "
count{
move 64, $1, 0u64
move 64, $2, 10u64
}
loop{
move 64, $3, 1u64
eq 64, $1, $2
jumpeq end
add 64, $1, $3
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
move 64, $0, 8u64
malloc $0, $1
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


        assert_eq!(machine.heap.read().unwrap().memory.read().unwrap()[..], [10, 0, 0, 0, 0, 0, 0, 0]);


    }


    fn simple_mutation(core: &mut dyn Core, _args: ForeignFunctionArg) -> SimpleResult {

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
move 64, $5, 8u64
malloc $1, $5
move 64, $1, $0
ret}";

        let binary = generate_binary(input, "test").unwrap();

        let mut machine = Machine::new();

        machine.add_function(None, Arc::new(simple_mutation));

        machine.load_binary(&binary);

        machine.add_core();

        machine.run();

        assert_eq!(machine.heap.read().unwrap().memory.read().unwrap()[..], [45, 0, 0, 0, 0, 0, 0, 0]);

    }

    fn complex_mutation(core: &mut dyn Core, args: ForeignFunctionArg) -> SimpleResult {
        
        let reg = core.get_register_64(0)?;

        let binding = args.unwrap();
        let mut binding = binding.write().unwrap();
        let value = binding.downcast_mut::<u64>().unwrap();

        *value += *reg;


        Ok(())
    }

    #[test]
    fn test_foreign_function_with_arg() {
        let input = "
main{
move 64, $0, 35u64
move 64, $5, 8u64
malloc $1, $5
move 64, $2, 0u64
foreign $2
move 64, $1, $0
ret}";

        let binary = generate_binary(input, "test").unwrap();

        let mut machine = Machine::new();

        let argument: ForeignFunctionArg = Some(Arc::new(RwLock::new(10u64)));

        machine.add_function(argument.clone(), Arc::new(complex_mutation));
        
        machine.load_binary(&binary);

        machine.add_core();

        machine.run();

        assert_eq!(machine.heap.read().unwrap().memory.read().unwrap()[..], [35, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(argument.unwrap().read().unwrap().downcast_ref::<u64>().unwrap(), &45u64);

    }

    #[test]
    fn test_allocation() {
        let input = "
main{
move 64, $0, 8u64
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

        assert_eq!(machine.heap.read().unwrap().memory.read().unwrap()[..], [10, 0, 0, 0, 0, 0, 0, 0]);

    }

    #[test]
    fn test_free_allocation() {
        let input = "
main{
move 64, $0, 8u64
malloc $1, $0
move 64, $2, 10u64
move 64, $1, $2
free $1
ret}
";
        let binary = generate_binary(input, "test").unwrap();

        let mut machine = Machine::new();

        machine.load_binary(&binary);

        println!("{}", binary.assembly());
        println!("{}", binary.program_with_count());

        machine.add_core();

        machine.run();

        assert_eq!(machine.heap.read().unwrap().memory.read().unwrap()[..], [10, 0, 0, 0, 0, 0, 0, 0]);

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
add 64, $1, $2
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

        println!("{:?}", machine.heap.read().unwrap().memory.read().unwrap());

        assert_eq!(machine.heap.read().unwrap().memory.read().unwrap()[..], [10, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0 ,0, 0, 0, 0, 0]);

    }

    #[test]
    fn test_garbage_collection() {
        let input = "
main{
move 64, $0, 8u64
malloc $1, $0
sleep 30u64, 1u64
malloc $1, $0
sleep 35u64, 1u64
malloc $1, $0
ret}
";
        let binary = generate_binary(input, "test").unwrap();

        let mut machine = Machine::new();
        machine.load_binary(&binary);

        machine.add_core();
        let options = MachineOptions {
            cycle_size: 1,
            join_thread_cycle: 0,
            defrag_cycle: 0,
            gc_time: Some(1),
            gc_program: None,
            stack_size: 1,
            stack_scale: 512,
            
        };

        machine.set_options(options);

        machine.run();

        println!("{:?}", machine.heap.read().unwrap().memory.read().unwrap());

        assert_eq!(machine.heap.read().unwrap().memory.read().unwrap()[..], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_garbage_colection_save() {
        let input = "
main{
move 64, $0, 8u64
malloc $1, $0
push 64, $1
sleep 70u64, 1u64
malloc $1, $0
pop 64, $1
ret}
";
        let binary = generate_binary(input, "test").unwrap();

        let mut machine = Machine::new();
        machine.load_binary(&binary);

        machine.add_core();
        let options = MachineOptions {
            cycle_size: 1,
            join_thread_cycle: 0,
            defrag_cycle: 0,
            gc_time: Some(1),
            gc_program: None,
            stack_size: 1,
            stack_scale: 512,
        };

        machine.set_options(options);

        println!("ASSEMBLY: {}", binary.assembly());
        println!("PROGRAM: \n{}", binary.program_with_count());
        

        machine.run();

        println!("{:?}", machine.heap.read().unwrap().memory.read().unwrap());

        assert_eq!(machine.heap.read().unwrap().memory.read().unwrap()[..], [0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_arbitrary_call() {
        
        let input = "
fibonacci{
move 32, $2, 1u32
leq 32, $1, $2
jumpgt rec
move $0, $1, 32
jump end
}
rec{
push 32, $1
sub 32, $1, $2
move 64, $10, fibonacci
call $10
pop 32, $1
push 32, $0
move 32, $2, 2u32
sub 32, $1, $2
call $10
pop 32, $1
add 32, $0, $1}
end{
ret}
main{
move 32, $1, 9u32
call fibonacci
move 64, $4, 4u64
malloc $3, $4
move 32, $3, $0
ret
}
";
        let binary = generate_binary(input, "test").unwrap();

        let mut machine = Machine::new();

        machine.load_binary(&binary);

        machine.add_core();

        machine.run();

        assert_eq!(machine.heap.read().unwrap().memory.read().unwrap()[..], [34, 0, 0, 0]);

        

    }

}
