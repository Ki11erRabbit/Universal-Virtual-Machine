use std::collections::HashMap;

use crate::instruction::Opcode;

use crate::Byte;

#[derive(Debug, PartialEq)]
pub struct Binary {
    shebang: String,
    header_size: usize,
    program_offset: usize,
    program_size: usize,
    data_segment_size: usize,
    entry_address: usize,
    program: Vec<Byte>,
    data_segment: Vec<Byte>,
    section_names: Vec<String>,
    section_addresses: Vec<usize>,
}



enum DeserializeState {
    Shebang,
    HeaderSize,
    ProgramOffset,
    ProgramSize,
    DataSegmentSize,
    EntryAddress,
    Program,
    DataSegment,
    SectionNames(usize),
    SectionAddresses,
}

impl Binary {
    pub fn new(shebang: &str,
               entry_address: usize,
               program: Vec<Byte>,
               data_segment: Vec<Byte>,
               section_names: Vec<String>,
               section_addresses: Vec<usize>) -> Binary {
        // however many bytes the shebang is
        // 8 bytes for the header size
        // 8 bytes for the program offset
        // 8 bytes for the program size
        // 8 bytes for the data segment size
        // 8 bytes for the entry address
        let header_size = shebang.len() + 8 + 8 + 8 + 8 + 8;
          Binary {
              shebang: shebang.to_string(),
              header_size,
              program_offset: header_size,
              program_size: program.len(),
              data_segment_size: data_segment.len(),
              entry_address,
              program,
              data_segment,
              section_names,
              section_addresses,
          }
    }

    pub fn program_with_count(&self) -> String {
        let mut program = " 0:".to_string();
        let mut count = 0;
        let mut line = 0;
        
        for byte in &self.program {
            program.push_str(&format!("{}", byte));
            program.push(' ');
            count += 1;
            if count == 16 {
                line += count;
                program.push('\n');
                program.push_str(&format!("{}:", line));
                count = 0;
            }
        }
        program
    }

    pub fn data_segment(&self) -> Vec<Byte> {
        self.data_segment.clone()
    }

    pub fn program(&self) -> Vec<Byte> {
        self.program.clone()
    }

    pub fn entry_address(&self) -> usize {
        self.entry_address
    }


    pub fn serialize(&self) -> Vec<Byte> {
        let mut binary = Vec::new();
        binary.extend(self.shebang.as_bytes());
        binary.extend(&self.header_size.to_le_bytes());
        binary.extend(&self.program_offset.to_le_bytes());
        binary.extend(&self.program_size.to_le_bytes());
        binary.extend(&self.data_segment_size.to_le_bytes());
        binary.extend(&self.entry_address.to_le_bytes());
        binary.extend(&self.program.clone());
        binary.extend(&self.data_segment.clone());
        for name in &self.section_names {
            binary.extend(name.as_bytes());
            binary.push(0);
        }
        binary.push(0);
        binary.push(0);
        for address in &self.section_addresses {
            binary.extend(&address.to_le_bytes());
        }
        binary
    }

    pub fn deserialize(bin: Vec<Byte>) -> Binary {
        let mut buffer = Vec::new();
        let mut shebang = String::new();
        let mut entry_address = 0;
        let mut program_size = 0;
        let mut data_segment_size = 0;
        let mut program = Vec::new();
        let mut data_segment = Vec::new();
        let mut section_names = Vec::new();
        let mut section_addresses = Vec::new();

        let mut state = DeserializeState::Shebang;
        for byte in bin {
            match state {
                DeserializeState::Shebang => {
                    buffer.push(byte);
                    if byte  == '\n' as u8 {
                        shebang = String::from_utf8(buffer.clone()).unwrap();
                        buffer.clear();
                        state = DeserializeState::HeaderSize;
                    }
                },
                DeserializeState::HeaderSize => {
                    buffer.push(byte);
                    if buffer.len() == 8 {
                        buffer.clear();
                        state = DeserializeState::ProgramOffset;
                    }
                },
                DeserializeState::ProgramOffset => {
                    buffer.push(byte);
                    if buffer.len() == 8 {
                        buffer.clear();
                        state = DeserializeState::ProgramSize;
                    }
                },
                DeserializeState::ProgramSize => {
                    buffer.push(byte);
                    if buffer.len() == 8 {
                        program_size = usize::from_le_bytes(buffer.clone().try_into().unwrap());
                        buffer.clear();
                        state = DeserializeState::DataSegmentSize;
                    }
                },
                DeserializeState::DataSegmentSize => {
                    buffer.push(byte);
                    if buffer.len() == 8 {
                        data_segment_size = usize::from_le_bytes(buffer.clone().try_into().unwrap());
                        buffer.clear();
                        state = DeserializeState::EntryAddress;
                    }
                },
                DeserializeState::EntryAddress => {
                    buffer.push(byte);
                    if buffer.len() == 8 {
                        entry_address = usize::from_le_bytes(buffer.clone().try_into().unwrap());
                        buffer.clear();
                        state = DeserializeState::Program;
                    }
                },
                DeserializeState::Program => {
                    buffer.push(byte);
                    if buffer.len() == program_size {
                        program = buffer.clone();
                        buffer.clear();
                        state = DeserializeState::DataSegment;
                    }
                },
                DeserializeState::DataSegment => {
                    buffer.push(byte);
                    if buffer.len() == data_segment_size {
                        data_segment = buffer.clone();
                        buffer.clear();
                        state = DeserializeState::SectionNames(0);
                    }
                },
                DeserializeState::SectionNames(zero_count) => {
                    buffer.push(byte);
                    if byte == 0 && zero_count == 0 {
                        buffer.pop();
                        section_names.push(String::from_utf8(buffer.clone()).unwrap());
                        buffer.clear();
                        state = DeserializeState::SectionNames(1);
                    }
                    else if byte == 0 && zero_count == 2 {
                        buffer.clear();
                        state = DeserializeState::SectionAddresses;
                    }
                    else if byte == 0 {
                        buffer.clear();
                        state = DeserializeState::SectionNames(zero_count + 1);
                    }
                    else {
                        state = DeserializeState::SectionNames(0);
                    }
                },
                DeserializeState::SectionAddresses => {
                    buffer.push(byte);

                    if buffer.len() == 8 {
                        section_addresses.push(usize::from_le_bytes(buffer.clone().try_into().unwrap()));
                        buffer.clear();
                    }
                }

            }
        }

       Binary::new(&shebang, entry_address, program, data_segment, section_names, section_addresses)
    }


    pub fn assembly(&self) -> String {
        let mut read_head = 0;
        let mut assembly = String::new();

        let mut section_positions = HashMap::new();

        for (name,address) in self.section_names.iter().zip(self.section_addresses.iter()) {
            section_positions.insert(address, name);
        }

        match section_positions.get(&read_head) {
            Some(name) => {
                assembly.push_str(&format!("{}{{\n", name));
            },
            None => {}
        }

        while read_head < self.program.len() {
            
            let opcode = Opcode::from(self.program[read_head] as u16);
            read_head += 2;

            use Opcode::*;
            match opcode {
                Halt => {
                    assembly.push_str("halt\n");
                },
                NoOp => {
                    assembly.push_str("noop\n");
                },
                DeRefReg => {
                    assembly.push_str("move ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let reg1 = self.program[read_head];
                    read_head += 1;
                    let reg2 = self.program[read_head];
                    read_head += 1;
                    let offset = i64::from_le_bytes(self.program[read_head..read_head + 8].try_into().unwrap());
                    read_head += 8;

                    assembly.push_str(&format!("{}, ${}, ${}, {}i64\n", size, reg1, reg2, offset));
                },
                DeRef => {
                    assembly.push_str("move ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;
                    let address = usize::from_le_bytes(self.program[read_head..read_head + 8].try_into().unwrap());
                    read_head += 8;

                    assembly.push_str(&format!("{}, ${}, #{}\n", size, reg, address));
                },
                Move => {
                    assembly.push_str("move ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let address_reg = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("{}, ${}, ${}\n", size, address_reg, reg));
                },
                Set => {
                    assembly.push_str("move ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;
                    match &size {
                        8 => {
                            let number = u8::from_le_bytes(self.program[read_head..read_head + 1].try_into().unwrap());
                            read_head += 1;
                            assembly.push_str(&format!("{}, ${}, {}u8\n", size, reg, number));
                        },
                        16 => {
                            let number = u16::from_le_bytes(self.program[read_head..read_head + 2].try_into().unwrap());
                            read_head += 2;
                            assembly.push_str(&format!("{}, ${}, {}u16\n", size, reg, number));
                        },
                        32 => {
                            let number = u32::from_le_bytes(self.program[read_head..read_head + 4].try_into().unwrap());
                            read_head += 4;
                            assembly.push_str(&format!("{}, ${}, {}u32\n", size, reg, number));
                        },
                        64 => {
                            let number = u64::from_le_bytes(self.program[read_head..read_head + 8].try_into().unwrap());
                            read_head += 8;
                            assembly.push_str(&format!("{}, ${}, {}u64\n", size, reg, number));
                        },
                        128 => {
                            let number = u128::from_le_bytes(self.program[read_head..read_head + 16].try_into().unwrap());
                            read_head += 16;
                            assembly.push_str(&format!("{}, ${}, {}u128\n", size, reg, number));
                        },
                        _ => panic!("Invalid size"),
                    }

                },
                RegMove => {
                    assembly.push_str("move ");
                    let reg1 = self.program[read_head];
                    read_head += 1;
                    let reg2 = self.program[read_head];
                    read_head += 1;
                    let size = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("${}, ${}, {}\n", reg1, reg2, size));
                },
                Add | Sub | Mul | Div | Eq | Neq | Lt | Gt | Leq | Geq => {
                    match opcode {
                        Add => assembly.push_str("add "),
                        Sub => assembly.push_str("sub "),
                        Mul => assembly.push_str("mul "),
                        Div => assembly.push_str("div "),
                        Eq => assembly.push_str("eq "),
                        Neq => assembly.push_str("neq "),
                        Lt => assembly.push_str("lt "),
                        Gt => assembly.push_str("gt "),
                        Leq => assembly.push_str("leq "),
                        Geq => assembly.push_str("geq "),
                        _ => panic!("Invalid opcode"),
                    }

                    let size = self.program[read_head];
                    read_head += 1;
                    let reg1 = self.program[read_head];
                    read_head += 1;
                    let reg2 = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("{}, ${}, ${}\n", size, reg1, reg2));
                },
                AddC | SubC | MulC | DivC | EqC | NeqC | LtC | GtC | LeqC | GeqC => {
                    match opcode {
                        AddC => assembly.push_str("add "),
                        SubC => assembly.push_str("sub "),
                        MulC => assembly.push_str("mul "),
                        DivC => assembly.push_str("div "),
                        EqC => assembly.push_str("eq "),
                        NeqC => assembly.push_str("neq "),
                        LtC => assembly.push_str("lt "),
                        GtC => assembly.push_str("gt "),
                        LeqC => assembly.push_str("leq "),
                        GeqC => assembly.push_str("geq "),
                        _ => panic!("Unvalid opcode"),
                    }

                    let size = self.program[read_head];
                    read_head += 1;
                    let reg1 = self.program[read_head];
                    read_head += 1;

                    match size {
                        8 => {
                            let const_ = u8::from_le_bytes(self.program[read_head..read_head + 1].try_into().unwrap());
                            read_head += 1;
                            assembly.push_str(&format!("{}, ${}, {}u8\n", size, reg1, const_));
                        },
                        16 => {
                            let const_ = u16::from_le_bytes(self.program[read_head..read_head + 2].try_into().unwrap());
                            read_head += 2;
                            assembly.push_str(&format!("{}, ${}, {}u16\n", size, reg1, const_));
                        },
                        32 => {
                            let const_ = u32::from_le_bytes(self.program[read_head..read_head + 4].try_into().unwrap());
                            read_head += 4;
                            assembly.push_str(&format!("{}, ${}, {}u32\n", size, reg1, const_));
                        },
                        64 => {
                            let const_ = u64::from_le_bytes(self.program[read_head..read_head + 8].try_into().unwrap());
                            read_head += 8;
                            assembly.push_str(&format!("{}, ${}, {}u64\n", size, reg1, const_));
                        },
                        128 => {
                            let const_ = u128::from_le_bytes(self.program[read_head..read_head + 16].try_into().unwrap());
                            read_head += 16;
                            assembly.push_str(&format!("{}, ${}, {}u128\n", size, reg1, const_));
                        },
                        _ => panic!("Invalid size"),
                    }
                },
                DeRefRegF => {
                    assembly.push_str("movef ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let reg1 = self.program[read_head];
                    read_head += 1;
                    let reg2 = self.program[read_head];
                    read_head += 1;
                    let offset = i64::from_le_bytes(self.program[read_head..read_head + 8].try_into().unwrap());
                    read_head += 8;

                    assembly.push_str(&format!("{}, ${}, ${}, {}i64\n", size, reg1, reg2, offset));
                },
                DeRefF => {
                    assembly.push_str("movef ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;
                    let address = usize::from_le_bytes(self.program[read_head..read_head + 8].try_into().unwrap());
                    read_head += 8;

                    assembly.push_str(&format!("{}, ${}, #{}\n", size, reg, address));
                },
                MoveF => {
                    assembly.push_str("movef ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let address_reg = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("{}, ${}, ${}\n", size, address_reg, reg));
                },
                SetF => {
                    assembly.push_str("movef ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;
                    let num_bytes = match &size {
                        32 => 4,
                        64 => 8,
                        _ => panic!("Invalid size"),
                    };

                    let number = f64::from_le_bytes(self.program[read_head..read_head + num_bytes].try_into().unwrap());
                    read_head += num_bytes;

                    assembly.push_str(&format!("{}, ${}, {}f{}\n", size, reg, number, size));
                },
                RegMoveF => {
                    assembly.push_str("movef ");
                    let reg1 = self.program[read_head];
                    read_head += 1;
                    let reg2 = self.program[read_head];
                    read_head += 1;
                    let size = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("${}, ${}, {}\n", reg1, reg2, size));
                },
                AddF | SubF | MulF | DivF | EqF | NeqF | LtF | GtF | LeqF | GeqF => {
                    match opcode {
                        AddF => assembly.push_str("addf "),
                        SubF => assembly.push_str("subf "),
                        MulF => assembly.push_str("mulf "),
                        DivF => assembly.push_str("divf "),
                        EqF => assembly.push_str("eqf "),
                        NeqF => assembly.push_str("neqf "),
                        LtF => assembly.push_str("ltf "),
                        GtF => assembly.push_str("gtf "),
                        LeqF => assembly.push_str("leqf "),
                        GeqF => assembly.push_str("geqf "),
                        _ => panic!("Unvalid opcode"),
                    }

                    let size = self.program[read_head];
                    read_head += 1;
                    let reg1 = self.program[read_head];
                    read_head += 1;
                    let reg2 = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("{}, ${}, ${}\n", size, reg1, reg2));
                },
                AddFC | SubFC | MulFC | DivFC | EqFC | NeqFC | LtFC | GtFC | LeqFC | GeqFC => {
                    match opcode {
                        AddFC => assembly.push_str("addc "),
                        SubFC => assembly.push_str("subc "),
                        MulFC => assembly.push_str("mulf "),
                        DivFC => assembly.push_str("divf "),
                        EqFC => assembly.push_str("eqf "),
                        NeqFC => assembly.push_str("neqf "),
                        LtFC => assembly.push_str("ltf "),
                        GtFC => assembly.push_str("gtf "),
                        LeqFC => assembly.push_str("leqf "),
                        GeqFC => assembly.push_str("geqf "),
                        _ => panic!("Unvalid opcode"),
                    }

                    let size = self.program[read_head];
                    read_head += 1;
                    let reg1 = self.program[read_head];
                    read_head += 1;

                    match size {
                        32 => {
                            let num_bytes = 4;
                            let number = f32::from_le_bytes(self.program[read_head..read_head + num_bytes].try_into().unwrap());
                            read_head += num_bytes;

                            assembly.push_str(&format!("{}, ${}, {}f{}\n", size, reg1, number, size));
                        },
                        64 => {
                            let num_bytes = 8;
                            let number = f64::from_le_bytes(self.program[read_head..read_head + num_bytes].try_into().unwrap());
                            read_head += num_bytes;

                            assembly.push_str(&format!("{}, ${}, {}f{}\n", size, reg1, number, size));
                        },
                        _ => panic!("Invalid size"),
                    }
                },
                And | Or | Xor => {
                    match opcode {
                        And => assembly.push_str("and "),
                        Or => assembly.push_str("or "),
                        Xor => assembly.push_str("xor "),
                        _ => panic!("Unvalid opcode"),
                    }

                    let size = self.program[read_head];
                    read_head += 1;
                    let reg1 = self.program[read_head];
                    read_head += 1;
                    let reg2 = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("{}, ${}, ${}\n", size, reg1, reg2));
                },
                AndC | OrC | XorC => {
                    match opcode {
                        AndC => assembly.push_str("and "),
                        OrC => assembly.push_str("or "),
                        XorC => assembly.push_str("xor "),
                        _ => panic!("Unvalid opcode"),
                    }

                    let size = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;
                    match size {
                        8 => {
                            let num_bytes = 1;
                            let number = u8::from_le_bytes(self.program[read_head..read_head + num_bytes].try_into().unwrap());
                            read_head += num_bytes;

                            assembly.push_str(&format!("{}, ${}, {}u8\n", size, reg, number));
                        },
                        16 => {
                            let num_bytes = 2;
                            let number = u16::from_le_bytes(self.program[read_head..read_head + num_bytes].try_into().unwrap());
                            read_head += num_bytes;

                            assembly.push_str(&format!("{}, ${}, {}u16\n", size, reg, number));
                        },
                        32 => {
                            let num_bytes = 4;
                            let number = u32::from_le_bytes(self.program[read_head..read_head + num_bytes].try_into().unwrap());
                            read_head += num_bytes;

                            assembly.push_str(&format!("{}, ${}, {}u32\n", size, reg, number));
                        },
                        64 => {
                            let num_bytes = 8;
                            let number = u64::from_le_bytes(self.program[read_head..read_head + num_bytes].try_into().unwrap());
                            read_head += num_bytes;

                            assembly.push_str(&format!("{}, ${}, {}u64\n", size, reg, number));
                        },
                        128 => {
                            let num_bytes = 16;
                            let number = u128::from_le_bytes(self.program[read_head..read_head + num_bytes].try_into().unwrap());
                            read_head += num_bytes;

                            assembly.push_str(&format!("{}, ${}, {}u128\n", size, reg, number));
                        },
                        _ => panic!("Invalid size"),
                    }
                },
                Not => {
                    assembly.push_str("not ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("{}, ${}\n", size, reg));
                },
                ShiftLeft => {
                    assembly.push_str("shl ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let reg1 = self.program[read_head];
                    read_head += 1;
                    let reg2 = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("{}, ${}, ${}\n", size, reg1, reg2));
                },
                ShiftRight => {
                    assembly.push_str("shr ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let reg1 = self.program[read_head];
                    read_head += 1;
                    let reg2 = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("{}, ${}, ${}\n", size, reg1, reg2));
                },
                ShiftLeftC => {
                    assembly.push_str("shl ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;
                    let amount = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("{}, ${}, {}u8\n", size, reg, amount));
                },
                ShiftRightC => {
                    assembly.push_str("shr ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;
                    let amount = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("{}, ${}, {}u8\n", size, reg, amount));
                },
                Jump | JumpEq | JumpNeq | JumpLt | JumpGt | JumpLeq | JumpGeq | JumpZero | JumpNotZero | JumpNeg |
                JumpPos | JumpEven | JumpOdd |  JumpInfinity | JumpNotInfinity |
                JumpOverflow | JumpUnderflow | JumpNotOverflow | JumpNotUnderflow | JumpNaN | JumpNotNaN |
                JumpRemainder | JumpNotRemainder => {
                    match opcode {
                        Jump => assembly.push_str("jump "),
                        JumpEq => assembly.push_str("jumpeq "),
                        JumpNeq => assembly.push_str("jumpneq "),
                        JumpLt => assembly.push_str("jumplt "),
                        JumpGt => assembly.push_str("jumpgt "),
                        JumpLeq => assembly.push_str("jumpleq "),
                        JumpGeq => assembly.push_str("jumpgeq "),
                        JumpZero => assembly.push_str("jumpzero "),
                        JumpNotZero => assembly.push_str("jumpnzero "),
                        JumpNeg => assembly.push_str("jumpneg "),
                        JumpPos => assembly.push_str("jumppos "),
                        JumpEven => assembly.push_str("jumpeven "),
                        JumpOdd => assembly.push_str("jumpodd "),
                        JumpInfinity => assembly.push_str("jumpinf "),
                        JumpNotInfinity => assembly.push_str("jumpninf "),
                        JumpOverflow => assembly.push_str("jumpoverflow "),
                        JumpUnderflow => assembly.push_str("jumpunderflow "),
                        JumpNotOverflow => assembly.push_str("jumpnoverflow "),
                        JumpNotUnderflow => assembly.push_str("jumpnunderflow "),
                        JumpNaN => assembly.push_str("jumpnan "),
                        JumpNotNaN => assembly.push_str("jumpnnan "),
                        JumpRemainder => assembly.push_str("jumprmndr "),
                        JumpNotRemainder => assembly.push_str("jumpnrmndr "),
                        _ => panic!("Unvalid opcode"),
                    }

                    let address = u64::from_le_bytes(self.program[read_head..read_head + 8].try_into().unwrap());
                    read_head += 8;

                    assembly.push_str(&format!("#{}\n", address));
                },
                JumpBack | JumpForward => {
                    match opcode {
                        JumpBack => assembly.push_str("jumpback "),
                        JumpForward => assembly.push_str("jumpforward "),
                        _ => panic!("Unvalid opcode"),
                    }

                    let offset = u64::from_le_bytes(self.program[read_head..read_head + 8].try_into().unwrap());
                    read_head += 8;

                    assembly.push_str(&format!("{}u64\n", offset));
                },
                CallArb => {
                    assembly.push_str("callarb ");

                    let reg = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("${}\n", reg));
                },
                Call => {
                    assembly.push_str("call ");

                    let address = u64::from_le_bytes(self.program[read_head..read_head + 8].try_into().unwrap());
                    read_head += 8;

                    assembly.push_str(&format!("#{}\n", address));
                },
                Return => {
                    assembly.push_str("ret\n");
                },
                Pop => {
                    assembly.push_str("pop ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("{}, ${}\n", size, reg));
                },
                Push => {
                    assembly.push_str("push ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("{}, ${}\n", size, reg));
                },
                PopF => {
                    assembly.push_str("popf ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("{}, ${}\n", size, reg));
                },
                PushF => {
                    assembly.push_str("pushf ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("{}, ${}\n", size, reg));
                },
                Malloc => {
                    assembly.push_str("malloc ");
                    let reg1 = self.program[read_head];
                    read_head += 1;
                    let reg2 = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("${}, ${}\n", reg1, reg2));
                },
                Free => {
                    assembly.push_str("free ");
                    let reg = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("${}\n", reg));
                },
                Realloc => {
                    assembly.push_str("realloc ");
                    let reg1 = self.program[read_head];
                    read_head += 1;
                    let reg2 = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("${}, ${}\n", reg1, reg2));
                },
                ReadByte => {
                    assembly.push_str("readbyte ");
                    let reg1 = self.program[read_head];
                    read_head += 1;
                    let reg2 = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("${}, ${}\n", reg1, reg2));
                },
                Read => {
                    assembly.push_str("read ");
                    let reg1 = self.program[read_head];
                    read_head += 1;
                    let reg2 = self.program[read_head];
                    read_head += 1;
                    let reg3 = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("${}, ${}, ${}\n", reg1, reg2, reg3));
                },
                WriteByte => {
                    assembly.push_str("writebyte ");
                    let reg1 = self.program[read_head];
                    read_head += 1;
                    let reg2 = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("${}, ${}\n", reg1, reg2));
                },
                Write => {
                    assembly.push_str("write ");
                    let reg1 = self.program[read_head];
                    read_head += 1;
                    let reg2 = self.program[read_head];
                    read_head += 1;
                    let reg3 = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("${}, ${}, ${}\n", reg1, reg2, reg3));
                },
                Open => {
                    assembly.push_str("open ");
                    let reg1 = self.program[read_head];
                    read_head += 1;
                    let reg2 = self.program[read_head];
                    read_head += 1;
                    let reg3 = self.program[read_head];
                    read_head += 1;
                    let reg4 = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("${}, ${}, ${}, ${}\n", reg1, reg2, reg3, reg4));
                },
                Close => {
                    assembly.push_str("close ");
                    let reg = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("${}\n", reg));
                },
                Flush => {
                    assembly.push_str("flush ");
                    let reg = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("${}\n", reg));
                },
                ThreadSpawn => {
                    assembly.push_str("threadspawn ");
                    let reg = self.program[read_head];
                    read_head += 1;
                    let id_reg = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("${}, ${}\n", reg, id_reg));
                },
                Remainder => {
                    assembly.push_str("rmndr ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;
                    

                    assembly.push_str(&format!("{}, ${}\n", size, reg));
                },
                Clear => {
                    assembly.push_str("clear\n");
                },
                AddFI | SubFI | MulFI | DivFI | AddIF | SubIF | MulIF | DivIF  => {
                    match opcode {
                        AddFI => assembly.push_str("addfi "),
                        SubFI => assembly.push_str("subfi "),
                        MulFI => assembly.push_str("mulfi "),
                        DivFI => assembly.push_str("divfi "),
                        AddIF => assembly.push_str("addif "),
                        SubIF => assembly.push_str("subif "),
                        MulIF => assembly.push_str("mulif "),
                        DivIF => assembly.push_str("divif "),
                        _ => {}
                    }
                    
                    let size = self.program[read_head];
                    read_head += 1;
                    let reg1 = self.program[read_head];
                    read_head += 1;
                    let reg2 = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("{}, ${}, ${}\n", size, reg1, reg2));
                },
                AddFIC | SubFIC | MulFIC | DivFIC => {
                    match opcode {
                        AddFIC => assembly.push_str("addfi "),
                        SubFIC => assembly.push_str("subfi "),
                        MulFIC => assembly.push_str("mulfi "),
                        DivFIC => assembly.push_str("divfi "),
                        _ => {}
                    }
                    
                    let size = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;
                    match size {
                        32 => {
                            let constant = u32::from_le_bytes(self.program[read_head..read_head + 4].try_into().unwrap());
                            read_head += 4;
                            assembly.push_str(&format!("{}, ${}, {}u32\n", size, reg, constant));
                        },
                        64 => {
                            let constant = u64::from_le_bytes(self.program[read_head..read_head + 8].try_into().unwrap());
                            read_head += 8;
                            assembly.push_str(&format!("{}, ${}, {}u64\n", size, reg, constant));
                        },
                        _ => panic!("Invalid size for instruction"),
                    }
                },
                AddIFC | SubIFC | MulIFC | DivIFC  => {
                    match opcode {
                        AddIFC => assembly.push_str("addif "),
                        SubIFC => assembly.push_str("subif "),
                        MulIFC => assembly.push_str("mulif "),
                        DivIFC => assembly.push_str("divif "),
                        _ => {}
                    }
                    let size = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;

                    match size {
                        32 => {
                            let constant = f32::from_le_bytes(self.program[read_head..read_head + 4].try_into().unwrap());
                            read_head += 4;
                            assembly.push_str(&format!("{}, ${}, {}f32\n", size, reg, constant));
                        },
                        64 => {
                            let constant = f64::from_le_bytes(self.program[read_head..read_head + 8].try_into().unwrap());
                            read_head += 8;
                            assembly.push_str(&format!("{}, ${}, {}f64\n", size, reg, constant));
                        },
                        _ => panic!("Invalid size for instruction"),
                    }

                }
                ThreadReturn => assembly.push_str("threadret"),
                ThreadJoin => {
                    assembly.push_str("threadjoin ");
                    let reg = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("${}\n", reg));
                },
                ThreadDetach => {
                    assembly.push_str("threaddetach ");
                    let reg = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("${}\n", reg));
                },
                StackPointer => {
                    assembly.push_str("stackptr ");
                    let reg = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("${}\n", reg));
                },
                ForeignCall => {
                    assembly.push_str("foriegn ");
                    let reg = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("${}\n", reg));
                },
                Sleep => {
                    assembly.push_str("sleep ");
                    let num = u64::from_le_bytes(self.program[read_head..read_head + 8].try_into().unwrap());
                    read_head += 8;
                    let scale = u64::from_le_bytes(self.program[read_head..read_head + 8].try_into().unwrap());
                    read_head += 8;

                    assembly.push_str(&format!("{}u64, {}u64\n", num, scale));
                },
                SleepReg => {
                    assembly.push_str("sleep ");
                    let reg = self.program[read_head];
                    read_head += 1;
                    let scale_reg = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("${}, ${}\n", reg, scale_reg));
                },
                Random => {
                    assembly.push_str("rand ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("{}, ${}\n", size, reg));
                },
                RandomF => {
                    assembly.push_str("randf ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("{}, ${}\n", size, reg));
                },
                Reset => {
                    assembly.push_str("reset\n");
                },
                StrLen => {
                    assembly.push_str("strlen ");
                    let reg1 = self.program[read_head];
                    read_head += 1;
                    let reg2 = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("${}, ${}\n", reg1, reg2));
                },

                
                Illegal => {
                    assembly.push_str("illegal\n");
                },

            }
            
            
            match section_positions.get(&read_head) {
                Some(name) => {
                    assembly.push_str(&format!("}}\n{}{{\n", name));
                },
                None => {}
            }
        }
        assembly.push_str("}");
        
        assembly
    }
    
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembler::generate_binary;
    

    #[test]
    fn test_hello_world() {
        let input = "LC{
 .string \"Hello, world!\"}
main{
move 64, $0, 1u64
move 64, $1, LC
move 64, $2, 13u64
write $0, $1, $2
flush $0
ret}";
        let binary = generate_binary(input, "vm").unwrap();

        //println!("{:?}", binary.serialize());

        let deserialized = Binary::deserialize(binary.serialize());

        //println!("{:?}", deserialized);

        println!("{}", deserialized.assembly());

        assert_eq!(deserialized, binary);
    }


}
