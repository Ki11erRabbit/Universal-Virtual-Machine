# Universal Virtual Machine

This is a register based virtual machine that I designed to be a library to help those who are making a programming language. This library provides a few tools to help with that goal.
1. The virtual machine itself
2. An assembler
3. A disassembler

## Architecture
This virtual machine was designed with parallel processing in mind and as a result, all "cores" run in their own threads.

Despite each core running in its own thread, the memory is shared between all cores. As a result there is some loss in performance if the cores try to read and write to the stack/heap at the same time.

We can interact with foreign code by registering a function with the virtual machine. That way, the FFI is done through Rust functions.
