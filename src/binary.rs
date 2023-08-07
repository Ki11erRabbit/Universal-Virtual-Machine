use std::collections::HashMap;

use crate::instruction::Opcode;

use crate::{Byte,Pointer};

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
        let mut program = String::new();
        let mut count = 0;
        for byte in &self.program {
            program.push_str(&format!("{}", byte));
            program.push(' ');
            count += 1;
            if count == 16 {
                program.push('\n');
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
                AddI | SubI | MulI | DivI | EqI | NeqI | LtI | GtI | LeqI | GeqI => {
                    match opcode {
                        AddI => assembly.push_str("addi "),
                        SubI => assembly.push_str("subi "),
                        MulI => assembly.push_str("muli "),
                        DivI => assembly.push_str("divi "),
                        EqI => assembly.push_str("eqi "),
                        NeqI => assembly.push_str("neqi "),
                        LtI => assembly.push_str("lti "),
                        GtI => assembly.push_str("gti "),
                        LeqI => assembly.push_str("leqi "),
                        GeqI => assembly.push_str("geqi "),
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
                AddU | SubU | MulU | DivU | EqU | NeqU | LtU | GtU | LeqU | GeqU => {
                    match opcode {
                        AddU => assembly.push_str("addu "),
                        SubU => assembly.push_str("subu "),
                        MulU => assembly.push_str("mulu "),
                        DivU => assembly.push_str("divu "),
                        EqU => assembly.push_str("equ "),
                        NeqU => assembly.push_str("nequ "),
                        LtU => assembly.push_str("ltu "),
                        GtU => assembly.push_str("gtu "),
                        LeqU => assembly.push_str("lequ "),
                        GeqU => assembly.push_str("gequ "),
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
                DeRefRegA => {
                    assembly.push_str("movea ");
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
                DeRefA => {
                    assembly.push_str("movea ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;
                    let address = usize::from_le_bytes(self.program[read_head..read_head + 8].try_into().unwrap());
                    read_head += 8;

                    assembly.push_str(&format!("{}, ${}, #{}\n", size, reg, address));
                },
                MoveA => {
                    assembly.push_str("movea ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let address_reg = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("{}, ${}, ${}\n", size, address_reg, reg));
                },
                SetA => {
                    assembly.push_str("movea ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;
                    let num_bytes = match &size {
                        8 => 1,
                        16 => 2,
                        32 => 4,
                        64 => 8,
                        128 => 16,
                        _ => panic!("Invalid size"),
                    };

                    let number = u128::from_le_bytes(self.program[read_head..read_head + num_bytes].try_into().unwrap());
                    read_head += num_bytes;

                    assembly.push_str(&format!("{}, ${}, {}u{}\n", size, reg, number, size));
                },
                RegMoveA => {
                    assembly.push_str("movea ");
                    let reg1 = self.program[read_head];
                    read_head += 1;
                    let reg2 = self.program[read_head];
                    read_head += 1;
                    let size = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("${}, ${}, {}\n", reg1, reg2, size));
                },
                AddAI | SubAI | MulAI | DivAI | EqAI | NeqAI | LtAI | GtAI | LeqAI | GeqAI => {
                    match opcode {
                        AddAI => assembly.push_str("addai "),
                        SubAI => assembly.push_str("subai "),
                        MulAI => assembly.push_str("mulai "),
                        DivAI => assembly.push_str("divai "),
                        EqAI => assembly.push_str("eqai "),
                        NeqAI => assembly.push_str("neqai "),
                        LtAI => assembly.push_str("ltai "),
                        GtAI => assembly.push_str("gtai "),
                        LeqAI => assembly.push_str("leqai "),
                        GeqAI => assembly.push_str("geqai "),
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
                AddAU | SubAU | MulAU | DivAU | EqAU | NeqAU | LtAU | GtAU | LeqAU | GeqAU => {
                    match opcode {
                        AddAU => assembly.push_str("addau "),
                        SubAU => assembly.push_str("subau "),
                        MulAU => assembly.push_str("mulau "),
                        DivAU => assembly.push_str("divau "),
                        EqAU => assembly.push_str("eqau "),
                        NeqAU => assembly.push_str("neqau "),
                        LtAU => assembly.push_str("ltau "),
                        GtAU => assembly.push_str("gtau "),
                        LeqAU => assembly.push_str("leqau "),
                        GeqAU => assembly.push_str("geqau "),
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
                PopA => {
                    assembly.push_str("popa ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("{}, ${}\n", size, reg));
                },
                PushA => {
                    assembly.push_str("pusha ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("{}, ${}\n", size, reg));
                },
                DeRefRegStack => {
                    assembly.push_str("movestack ");
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
                DeRefStack => {
                    assembly.push_str("movestack ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;
                    let address = usize::from_le_bytes(self.program[read_head..read_head + 8].try_into().unwrap());
                    read_head += 8;

                    assembly.push_str(&format!("{}, ${}, #{}\n", size, reg, address));
                },
                MoveStack => {
                    assembly.push_str("movestack ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let address_reg = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("{}, ${}, ${}\n", size, address_reg, reg));
                },
                DeRefRegStackF => {
                    assembly.push_str("movestackf ");
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
                DeRefStackF => {
                    assembly.push_str("movestackf ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;
                    let address = usize::from_le_bytes(self.program[read_head..read_head + 8].try_into().unwrap());
                    read_head += 8;

                    assembly.push_str(&format!("{}, ${}, #{}\n", size, reg, address));
                },
                MoveStackF => {
                    assembly.push_str("movestackf ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let address_reg = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("{}, ${}, ${}\n", size, address_reg, reg));
                },
                DeRefRegStackA => {
                    assembly.push_str("movestacka ");
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
                DeRefStackA => {
                    assembly.push_str("movestacka ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;
                    let address = usize::from_le_bytes(self.program[read_head..read_head + 8].try_into().unwrap());
                    read_head += 8;

                    assembly.push_str(&format!("{}, ${}, #{}\n", size, reg, address));
                },
                MoveStackA => {
                    assembly.push_str("movestacka ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let address_reg = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("{}, ${}, ${}\n", size, address_reg, reg));
                },
                DeRefRegStackC => {
                    assembly.push_str("movecstack ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let core_reg = self.program[read_head];
                    read_head += 1;
                    let reg1 = self.program[read_head];
                    read_head += 1;
                    let reg2 = self.program[read_head];
                    read_head += 1;
                    let offset = i64::from_le_bytes(self.program[read_head..read_head + 8].try_into().unwrap());
                    read_head += 8;

                    assembly.push_str(&format!("{}, ${}, ${}, ${}, {}i64\n", size, core_reg, reg1, reg2, offset));
                },
                DeRefStackC => {
                    assembly.push_str("movecstack ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let core_reg = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;
                    let address = usize::from_le_bytes(self.program[read_head..read_head + 8].try_into().unwrap());
                    read_head += 8;

                    assembly.push_str(&format!("{}, ${}, ${}, #{}\n", size, core_reg, reg, address));
                },
                MoveStackC => {
                    assembly.push_str("movecstack ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let core_reg = self.program[read_head];
                    read_head += 1;
                    let address_reg = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("{}, ${}, ${}, ${}\n", size, core_reg, address_reg, reg));
                },
                DeRefRegStackCF => {
                    assembly.push_str("movecstackf ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let core_reg = self.program[read_head];
                    read_head += 1;
                    let reg1 = self.program[read_head];
                    read_head += 1;
                    let reg2 = self.program[read_head];
                    read_head += 1;
                    let offset = i64::from_le_bytes(self.program[read_head..read_head + 8].try_into().unwrap());
                    read_head += 8;

                    assembly.push_str(&format!("{}, ${}, ${}, ${}, {}i64\n", size, core_reg, reg1, reg2, offset));
                },
                DeRefStackCF => {
                    assembly.push_str("movecstackf ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let core_reg = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;
                    let address = usize::from_le_bytes(self.program[read_head..read_head + 8].try_into().unwrap());
                    read_head += 8;

                    assembly.push_str(&format!("{}, ${}, ${}, #{}\n", size, core_reg, reg, address));
                },
                MoveStackCF => {
                    assembly.push_str("movecstackf ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let core_reg = self.program[read_head];
                    read_head += 1;
                    let address_reg = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("{}, ${}, ${}, ${}\n", size, core_reg, address_reg, reg));
                },
                DeRefRegStackCA => {
                    assembly.push_str("movecstacka ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let core_reg = self.program[read_head];
                    read_head += 1;
                    let reg1 = self.program[read_head];
                    read_head += 1;
                    let reg2 = self.program[read_head];
                    read_head += 1;
                    let offset = i64::from_le_bytes(self.program[read_head..read_head + 8].try_into().unwrap());
                    read_head += 8;

                    assembly.push_str(&format!("{}, ${}, ${}, ${}, {}i64\n", size, core_reg, reg1, reg2, offset));
                },
                DeRefStackCA => {
                    assembly.push_str("movecstacka ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let core_reg = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;
                    let address = usize::from_le_bytes(self.program[read_head..read_head + 8].try_into().unwrap());
                    read_head += 8;

                    assembly.push_str(&format!("{}, ${}, ${}, #{}\n", size, core_reg,reg, address));
                },
                MoveStackCA => {
                    assembly.push_str("movecstacka ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let core_reg = self.program[read_head];
                    read_head += 1;
                    let address_reg = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;

                    assembly.push_str(&format!("{}, ${}, ${}, ${}\n", size, core_reg, address_reg, reg));
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
                    assembly.push_str("remainder ");
                    let size = self.program[read_head];
                    read_head += 1;
                    let reg = self.program[read_head];
                    read_head += 1;
                    

                    assembly.push_str(&format!("{}, ${}\n", size, reg));
                },
                Clear => {
                    assembly.push_str("clear\n");
                },
                AddFI | SubFI | MulFI | DivFI | AddIF | SubIF | MulIF | DivIF | AddUF | SubUF |
                MulUF | DivUF => {
                    match opcode {
                        AddFI => assembly.push_str("addfi "),
                        SubFI => assembly.push_str("subfi "),
                        MulFI => assembly.push_str("mulfi "),
                        DivFI => assembly.push_str("divfi "),
                        AddIF => assembly.push_str("addif "),
                        SubIF => assembly.push_str("subif "),
                        MulIF => assembly.push_str("mulif "),
                        DivIF => assembly.push_str("divif "),
                        AddUF => assembly.push_str("adduf "),
                        SubUF => assembly.push_str("subuf "),
                        MulUF => assembly.push_str("muluf "),
                        DivUF => assembly.push_str("divuf "),
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
