
#[derive(Debug, PartialEq)]
pub struct Binary {
    shebang: String,
    header_size: usize,
    program_offset: usize,
    program_size: usize,
    entry_address: usize,
    program: Vec<u8>,
    section_names: Vec<String>,
    section_addresses: Vec<usize>,
}



impl Binary {
    pub fn new(shebang: &str,
                entry_address: usize,
                program: Vec<u8>,
                section_names: Vec<String>,
               section_addresses: Vec<usize>) -> Binary {
        let header_size = shebang.len() + 8 + 8 + 8;
          Binary {
                shebang: shebang.to_string(),
                header_size,
                program_offset: header_size,
                program_size: program.len(),
                entry_address,
                program,
                section_names,
                section_addresses,
          }
    }


    pub fn serialize(&self) -> Vec<u8> {
        let mut binary = Vec::new();
        binary.extend(self.shebang.as_bytes());
        binary.extend(&self.header_size.to_le_bytes());
        binary.extend(&self.program_offset.to_le_bytes());
        binary.extend(&self.program_size.to_le_bytes());
        binary.extend(&self.entry_address.to_le_bytes());
        binary.extend(&self.program.clone());
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

    pub fn deserialize(bin: Vec<u8>) -> Binary {
        let mut buffer = Vec::new();
        let mut shebang = String::new();
        let mut entry_address = 0;
        let mut program_size = 0;
        let mut program = Vec::new();
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

       Binary::new(&shebang, entry_address, program, section_names, section_addresses)
    }
    
}

enum DeserializeState {
    Shebang,
    HeaderSize,
    ProgramOffset,
    ProgramSize,
    EntryAddress,
    Program,
    SectionNames(usize),
    SectionAddresses,
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
        let binary = generate_binary(input).unwrap();

        //println!("{:?}", binary.serialize());

        let deserialized = Binary::deserialize(binary.serialize());

        //println!("{:?}", deserialized);

        assert_eq!(deserialized, binary);
    }


}
