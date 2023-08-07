#![allow(arithmetic_overflow)]

use std::cell::RefCell;
use std::any::Any;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, RwLock};
use std::thread::{self,JoinHandle};
use std::sync::mpsc::{Sender,Receiver, channel};
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Write;

use crate::{Pointer, CoreId, Byte, RegisterType, Message, Fault, ForeignFunction, ForeignFunctionArg};
use crate::binary::Binary;
use crate::core::Core;



pub struct Machine {
    memory: Arc<RwLock<Vec<Byte>>>,
    cores: Vec<Core>,
    core_threads: Vec<Option<JoinHandle<Result<(),Fault>>>>,
    program: Option<Arc<Vec<Byte>>>,
    entry_point: Option<usize>,
    channels: Rc<RefCell<Vec<Option<(Sender<Message>, Receiver<Message>)>>>>,
    files: Vec<Box<dyn Write>>,
    thread_children: HashMap<CoreId, CoreId>,
    threads_to_join: Rc<RefCell<Vec<CoreId>>>,
    main_thread_id: CoreId,
    foriegn_functions: Vec<(ForeignFunctionArg, ForeignFunction)>,
}

impl Machine {

    pub fn new() -> Machine {
        Machine {
            memory: Arc::new(RwLock::new(Vec::new())),
            cores: Vec::new(),
            core_threads: Vec::new(),
            program: None,
            entry_point: None,
            channels: Rc::new(RefCell::new(Vec::new())),
            files: vec![Box::new(std::io::stdout()), Box::new(std::io::stderr())],
            thread_children: HashMap::new(),
            threads_to_join: Rc::new(RefCell::new(Vec::new())),
            main_thread_id: 0,
            foriegn_functions: Vec::new(),
        }
    }

    pub fn new_with_cores(core_count: usize) -> Machine {
        let mut machine = Machine::new();
        for _ in 0..core_count {
            machine.add_core();
        }
        machine
    }

    pub fn add_function(&mut self, func_arg: Option<Arc<RwLock<dyn Any + Send + Sync>>>, function: fn(&mut Core, Option<Arc<RwLock<dyn Any + Send + Sync>>>) -> Result<(),Fault>) {
        self.foriegn_functions.push((func_arg, Arc::new(function)));
    }

    pub fn run_at(&mut self, program_counter: usize) {
        self.entry_point = Some(program_counter);
        self.run();
    }
    
    pub fn run(&mut self) {
        //TODO: change print to log
        let program_counter = self.entry_point.expect("Entry point not set");
        self.run_core(0, program_counter);
        let mut main_thread_done = false;
        loop {
            self.check_main_core(&mut main_thread_done);
            self.check_messages();
            self.join_joinable_threads();

            //TODO: check for commands from the threads to do various things like allocate more memory, etc.
            if main_thread_done {
                break;
            }
        }
        self.core_threads.clear();
        self.channels.borrow_mut().clear();
    }

    fn check_main_core(&mut self, main_thread_done: &mut bool) {
        if self.core_threads[self.main_thread_id as usize].as_ref().expect("main core doesn't exist").is_finished() {
            *main_thread_done = true;
            self.join_thread(self.main_thread_id);

            self.channels.borrow_mut()[self.main_thread_id as usize] = None;
        }

    }

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

                            //self.join_thread(thread_id);

                            self.threads_to_join.borrow_mut().push(thread_id);

                            //send.send(Message::Success).unwrap();
                        },
                        Message::DetachThread(thread_id) => {

                            self.core_threads[thread_id as usize].take().expect("Already joined this core");

                            send.send(Message::Success).unwrap();
                        },
                        Message::GetForeignFunction(function_id) => {
                            let (arg, func) = self.foriegn_functions[function_id as usize].clone();
                            send.send(Message::ForeignFunction(arg, func)).unwrap();
                        },
                        
                        _ => unimplemented!(),

                    }

                },
                Err(_) => {},
            }

           core_id += 1;
        });

    }

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

    fn write_file(&mut self, fd: i64, data: Vec<u8>) -> Message {
        let fd = fd as usize - 1;
        if fd >= self.files.len() {
            return Message::Error(Fault::InvalidFileDescriptor);
        }
        let file = &mut self.files[fd];
        match file.write(&data) {
            Ok(_) => Message::Success,
            Err(_) => Message::Error(Fault::FileWriteError),
        }
    }

    fn flush(&mut self, fd: i64) -> Message {
        let fd = fd as usize - 1;
        if fd >= self.files.len() {
            return Message::Error(Fault::InvalidFileDescriptor);
        }
        let file = &mut self.files[fd];
        match file.flush() {
            Ok(_) => Message::Success,
            Err(_) => Message::Error(Fault::FileWriteError),
        }
    }

    fn close_file(&mut self, fd: i64) -> Message {
        let fd = fd as usize - 1;
        if fd >= self.files.len() {
            return Message::Error(Fault::InvalidFileDescriptor);
        }

        self.files.remove(fd);
        Message::Success
    }

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
                        self.files.push(Box::new(file));
                        Message::FileDescriptor(self.files.len() as i64)
                    },
                    Err(_) => Message::Error(Fault::FileOpenError),
                }
            },
            Err(_) => Message::Error(Fault::InvalidMessage),
        }
    }

    fn thread_spawn(&mut self, program_counter: u64) -> (Message, u8) {
        if self.cores.len() == 0 {
            self.add_core();
        }
        self.run_core_threaded(0, program_counter as usize);
        let core_id = self.core_threads.len() - 1;
        (Message::ThreadSpawned(core_id as u8), core_id as u8)
    }

    pub fn add_core(&mut self) {
        //TODO: Make it so that we don't panic from trying to add another core while running
        let (core_sender, core_receiver) = channel();
        let (machine_sender, machine_receiver) = channel();
        self.channels.borrow_mut().push(Some((core_sender, machine_receiver)));
        let core = Core::new(self.memory.clone(), machine_sender, core_receiver,);
        self.cores.push(core);
    }

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

    pub fn add_program(&mut self, program: Vec<u8>) {
        self.program = Some(Arc::new(program));

        self.memory.write().unwrap().extend_from_slice(&self.program.as_ref().unwrap()[0..]);
        
    }

    pub fn core_count(&self) -> usize {
        self.core_threads.len()
    }

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

        assert_eq!(machine.memory.read().unwrap()[..], [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 5, 0, 0, 0, 8, 0, 0, 0, 13, 0, 0, 0, 21, 0, 0, 0, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
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

        assert_eq!(machine.memory.read().unwrap()[..], [0, 0, 0, 34]);
        
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


        assert_eq!(machine.memory.read().unwrap()[..], [0, 0, 0, 0, 0, 0, 0, 10]);


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

        assert_eq!(machine.memory.read().unwrap()[..], [0, 0, 0, 0, 0, 0, 0, 45]);

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
foreign $1
move 64, $1, $0
ret}";

        let binary = generate_binary(input, "test").unwrap();

        let mut machine = Machine::new();

        let argument: Option<Arc<RwLock<dyn Any + Send + Sync>>> = Some(Arc::new(RwLock::new(10u64)));

        machine.add_function(argument.clone(), complex_mutation);

        machine.load_binary(&binary);

        machine.add_core();

        machine.run();

        assert_eq!(machine.memory.read().unwrap()[..], [0, 0, 0, 0, 0, 0, 0, 35]);
        assert_eq!(argument.unwrap().read().unwrap().downcast_ref::<u64>().unwrap(), &45u64);

    }


}
