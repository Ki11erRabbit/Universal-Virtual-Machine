use chumsky::prelude::*;
use std::collections::HashMap;


#[derive(Debug, Clone)]
enum Number {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    I128(i128),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    U128(u128),
    F32(f32),
    F64(f64),
}

impl Number {
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            Number::I8(i) => i.to_le_bytes().to_vec(),
            Number::I16(i) => i.to_le_bytes().to_vec(),
            Number::I32(i) => i.to_le_bytes().to_vec(),
            Number::I64(i) => i.to_le_bytes().to_vec(),
            Number::I128(i) => i.to_le_bytes().to_vec(),
            Number::U8(i) => i.to_le_bytes().to_vec(),
            Number::U16(i) => i.to_le_bytes().to_vec(),
            Number::U32(i) => i.to_le_bytes().to_vec(),
            Number::U64(i) => i.to_le_bytes().to_vec(),
            Number::U128(i) => i.to_le_bytes().to_vec(),
            Number::F32(i) => i.to_le_bytes().to_vec(),
            Number::F64(i) => i.to_le_bytes().to_vec(),
        }
    }
}

#[derive(Debug, Clone)]
enum Ast {
    Instruction(String, Vec<Ast>),
    Register(u8),
    Number(Number),
    Address(u64),
    Label(String),
    Comment(String),
    String(String),
    MemorySet(String, Box<Ast>),
    Labelled(Box<Ast>, Vec<Ast>),
    File(Vec<Ast>),
}

fn instruction_parser() -> impl Parser<char, Ast, Error = Simple<char>> {
    let inst_without_comment = choice((
        op_parser(),
        memory_parser(),
    ))
        .padded()
        .map(|i| i);

    let inst_with_comment = choice((
        op_parser(),
        memory_parser(),
    ))
        .then_ignore(comment_parser().padded())
        .map(|i| i);

    let instruction = choice((
        inst_with_comment,
        inst_without_comment,
    ));

    instruction
}

fn memory_parser() -> impl Parser<char, Ast, Error = Simple<char>> {
    let numbers = choice((
            just(".u8").to("u8"),
            just(".u16").to("u16"),
            just(".u32").to("u32"),
            just(".u64").to("u64"),
            just(".u128").to("u128"),
            just(".i8").to("i8"),
            just(".i16").to("i16"),
            just(".i32").to("i32"),
            just(".i64").to("i64"),
            just(".i128").to("i128"),
            just(".f32").to("f32"),
            just(".f64").to("f64"),
        )).padded()
        .then(number_parser().padded())
        .map(|(t, v)| Ast::MemorySet(t.to_owned(), Box::new(v)));

    let strings = just(".string").to("string").padded()
        .ignore_then(string_parser().padded())
        .map(|s| Ast::MemorySet("string".to_owned(), Box::new(s)));

    choice((
        numbers,
        strings,
    )).labelled("memory parser")
        

}

fn op_parser() -> impl Parser<char, Ast, Error = Simple<char>> {
    let instruction = one_of("abcdefghijklmnopqrstuvwxyz").repeated().at_least(3).padded()
        .then(choice((register_parser(), label_parser(), number_parser(), address_parser())).padded()
            .separated_by(just(",").padded()))
        .map(|(chars, args)| Ast::Instruction(chars.into_iter().collect(), args));


    instruction
}

fn register_parser() -> impl Parser<char, Ast, Error = Simple<char>> {
    let register = just("$")
        .ignore_then(raw_number_parser())
        .map(|n| match n {
            Number::U64(i) => Ast::Register(i as u8),
            Number::I64(i) => Ast::Register(i as u8),
            Number::U32(i) => Ast::Register(i as u8),
            Number::I32(i) => Ast::Register(i as u8),
            Number::U16(i) => Ast::Register(i as u8),
            Number::I16(i) => Ast::Register(i as u8),
            Number::U8(i) => Ast::Register(i as u8),
            Number::I8(i) => Ast::Register(i as u8),
            _ => panic!("Invalid register"),
        });

    register
}

fn address_parser() -> impl Parser<char, Ast, Error = Simple<char>> {
    let address = just("#")
        .ignore_then(raw_number_parser())
        .map(|n| match n {
            Number::U8(i) => Ast::Address(i as u64),
            Number::U16(i) => Ast::Address(i as u64),
            Number::U32(i) => Ast::Address(i as u64),
            Number::U64(i) => Ast::Address(i as u64),
            Number::U128(i) => Ast::Address(i as u64),
            _ => panic!("Invalid address"),
        });

    address
}

fn label_parser() -> impl Parser<char, Ast, Error = Simple<char>> {
    let cant_start_with = none_of("0123456789 \n\t\r'\"\\,()[]{}@;:");
    let cant_contain = none_of(" \n\t\r'\"\\,()[]{}@;:");



    let normal = cant_start_with
        .then(cant_contain.repeated())
        .map(|(c, s)| format!("{}{}", c, s.iter().collect::<String>()));

    
    normal
        .map(|s| Ast::Label(s))
        .padded()
        .labelled("label parser")
}

fn comment_parser() -> impl Parser<char, Ast, Error = Simple<char>> {
    let comment = just(";")
        .ignore_then(none_of("\n\r").repeated().padded())
        .map(|chars| Ast::Comment(chars.into_iter().collect()));

    comment
}



fn string_parser() -> impl Parser<char, Ast, Error = Simple<char>> {

    let escape = just::<char, char, Simple<char>>('\\')
        .then(one_of("\"\\nrt "))
        .map(|(_, c)| match c {
            '"' => '"',
            '\\' => '\\',
            'n' => '\n',
            'r' => '\r',
            't' => '\t',
            ' ' => ' ',
            _ => unreachable!()
        });

    let string_char = none_of("\"\\")
        .or(escape);

    let string = just('"')
        .ignore_then(string_char.repeated())
        .then_ignore(just('"'))
        //.then_ignore(end())
        .map(|s| Ast::String(s.iter().collect()))
        .padded()
        .labelled("string parser");
    

    string
}
    

fn raw_number_parser() -> impl Parser<char, Number, Error = Simple<char>> {

    let dec = text::int(10).map(|i: String| i);
    let hex = text::int(16).map(|i: String| i);
    let oct = text::int(8).map(|i: String| i);
    let bin = text::int(2).map(|i: String| i);

    let sign = choice((
        just("+").to("+"),
        just("-").to("-"),
    ));

    let hex_prefix = choice((
        just("0x").to("0x"),
        just("0X").to("0x"),
    ));

    let oct_prefix = choice((
        just("0o").to("0o"),
        just("0O").to("0o"),
    ));

    let bin_prefix = choice((
        just("0b").to("0b"),
        just("0B").to("0b"),
    ));

    let int_suffix = choice((
        just("i8").to("i8"),
        just("I8").to("i8"),
        just("u8").to("u8"),
        just("U8").to("u8"),
        just("i16").to("i16"),
        just("I16").to("i16"),
        just("u16").to("u16"),
        just("U16").to("u16"),
        just("i32").to("i32"),
        just("I32").to("i32"),
        just("u32").to("u32"),
        just("U32").to("u32"),
        just("i64").to("i64"),
        just("I64").to("i64"),
        just("u64").to("u64"),
        just("U64").to("u64"),
        just("i128").to("i128"),
        just("I128").to("i128"),
        just("u128").to("u128"),
        just("U128").to("u128"),
        just("f32").to("f32"),//converts to float
        just("F32").to("f32"),//converts to float
        just("f64").to("f64"),//converts to float
        just("F64").to("f64"),//converts to float
        just("").to(""),
    ));

    let dec_point = just(".").to(".");
    let exp_mark = choice((just("e").to("e"), just("E").to("e")));

    let s_dec_int = sign
        .then(dec)
        .map(|(s, i)| s.to_owned() + &i);

    let u_dec_int = dec
        .map(|i| i);

    let hex_int = sign
        .then(hex_prefix.map(|p| p))
        .then(hex)
        .map(|((s, p), i)| s.to_owned() + &p + &i);

    let oct_int = sign
        .then(oct_prefix.map(|p| p))
        .then(oct)
        .map(|((s, p), i)| s.to_owned() + &p + &i);

    let bin_int = sign
        .then(bin_prefix.map(|p| p))
        .then(bin)
        .map(|((s, p), i)| s.to_owned() + &p + &i);
    

    let integer = choice((s_dec_int, u_dec_int, hex_int, oct_int, bin_int))
        .then(int_suffix.map(|s| s))
        .map(|(i, s)| match s {
            "i8" => Number::I8(i.parse::<i8>().unwrap()),
            "u8" => Number::U8(i.parse::<u8>().unwrap()),
            "i16" => Number::I16(i.parse::<i16>().unwrap()),
            "u16" => Number::U16(i.parse::<u16>().unwrap()),
            "i32" => Number::I32(i.parse::<i32>().unwrap()),
            "u32" => Number::U32(i.parse::<u32>().unwrap()),
            "i64" => Number::I64(i.parse::<i64>().unwrap()),
            "u64" => Number::U64(i.parse::<u64>().unwrap()),
            "i128" => Number::I128(i.parse::<i128>().unwrap()),
            "u128" => Number::U128(i.parse::<u128>().unwrap()),
            "f32" => Number::F32(i.parse::<f32>().unwrap()),
            "f64" => Number::F64(i.parse::<f64>().unwrap()),
            _ => Number::U8(i.parse::<u8>().unwrap()),
        });
    

    let float_suffix = choice((
        just("f32").to("f32"),//converts to float
        just("F32").to("f32"),//converts to float
        just("f64").to("f64"),//converts to float
        just("F64").to("f64"),//converts to float
        just("").to(""),
    ));
    
    let exp = exp_mark
        .then(sign.map(|s| s))
        .then(dec.map(|i| i))
        .map(|((e, s), i)| e.to_owned() + &s + &i);

    let float_exp = sign
        .then(dec.map(|i| i))
        .then(dec_point.map(|p| p.to_string()))
        .then(dec.map(|i| i))
        .then(exp.map(|e| e))
        .map(|((((s, i), p), i2), e)| (s.to_owned() + &i + &p + &i2 + &e).parse::<f64>().unwrap());


    let float_wo_exp = sign
        .then(dec)
        .then(dec_point.map(|p| p.to_string()))
        .then(dec)
        .map(|(((s, p), i), i3)| (s.to_owned() + &p + &i + &i3).parse::<f64>().unwrap());

    let float = choice((float_exp, float_wo_exp))
        .then(float_suffix.map(|s| s))
        .map(|(f, s)| match s {
            "f32" => Number::F32(f as f32),
            "f64" => Number::F64(f),
            _ => Number::F64(f),
        });
        

    let number = choice((float, integer))
        .labelled("raw number parser");

        
    number
}

fn number_parser() -> impl Parser<char, Ast, Error = Simple<char>> {
    raw_number_parser().map(|n| Ast::Number(n))
        .labelled("number parser")
}



fn labelled_parser() -> impl Parser<char, Ast, Error = Simple<char>> {
    let parser = label_parser()
        .then(instruction_parser().repeated().at_least(1)
              .delimited_by(just("{"), just("}")))
        .map(|(l, i)| Ast::Labelled(Box::new(l), i))
        .labelled("block parser").padded();

    parser
}

fn file_parser() -> impl Parser<char, Ast, Error = Simple<char>> {
    let parser = labelled_parser().repeated().padded()
        .map(|i| Ast::File(i))
        .labelled("file parser");

    parser
}

#[derive(Debug, Clone)]
enum MoveOps {
    Number,
    Register,
    Address,
}

fn parse_arithmetic(opcode: Vec<u8>, args: &Vec<Ast>) -> Result<Vec<u8>, String> {
    let mut bytes = opcode;
    
    for arg in args {
        match arg {
            Ast::Number(n) => {
                bytes.append(&mut n.to_bytes());
            },
            Ast::Register(r) => {
                bytes.push(*r);
            },
            _ => return Err("Only registers are allowed in this op".to_owned()),
        }
    }

    Ok(bytes)
}

fn parse_jump<F>(opcode: Vec<u8>, args: &Vec<Ast>, pos_setter: F) -> Result<Vec<u8>, String>
where
    F: Fn(String) -> u64,{
    let mut bytes = opcode;
    
    for arg in args {
        match arg {
            Ast::Number(n) => {
                bytes.append(&mut n.to_bytes());
            },
            Ast::Label(l) => {
                bytes.append(&mut pos_setter(l.to_owned()).to_le_bytes().to_vec());
            },
            _ => return Err("Only numbers are allowed in this op".to_owned()),
        }
    }

    Ok(bytes)
}

/// This function is used to parse an instruction into an opcode
fn parse_instruction<F>(ast: &Ast, pos_setter: F) -> Result<Vec<u8>, String>
where
    F: Fn(String) -> u64, {
    match ast {
        Ast::Instruction(name, args) => {
            match name.as_str() {
                "halt" => Ok(vec![0x00]),
                "noop" => Ok(vec![0x01]),
                "move" => {
                    let mut bytes: Vec<u8> = Vec::new();
                    let mut ops: Vec<MoveOps> = Vec::new();
                    for arg in args {
                        match arg {
                            Ast::Number(n) => {
                                bytes.append(&mut n.to_bytes());
                                ops.push(MoveOps::Number);
                            },
                            Ast::Register(r) => {
                                bytes.push(*r);
                                ops.push(MoveOps::Register);
                            },
                            Ast::Label(l) => {
                                let pos = pos_setter(l.to_owned());
                                bytes.append(&mut pos.to_le_bytes().to_vec());
                                ops.push(MoveOps::Number);
                            },
                            Ast::Address(a) => {
                                bytes.append(&mut a.to_le_bytes().to_vec());
                                ops.push(MoveOps::Address);
                            },
                            _ => return Err("Expected number, register or label".to_owned()),
                        }
                    }

                    match ops.as_slice() {
                        //move opcode
                        [MoveOps::Number, MoveOps::Address, MoveOps::Register] => {
                            let mut temp = vec![4,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        //set opcode
                        [MoveOps::Number, MoveOps::Register, MoveOps::Number] => {
                            let mut temp = vec![5,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        //regmove opcode
                        [MoveOps::Number, MoveOps::Register, MoveOps::Register] => {
                            let mut temp = vec![159,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        //deref opcode
                        [MoveOps::Number, MoveOps::Register, MoveOps::Address] => {
                            let mut temp = vec![3,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        //derefreg
                        [MoveOps::Number, MoveOps::Register, MoveOps::Register, MoveOps::Number] => {
                            let mut temp = vec![2,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        _ => return Err(format!("Invalid arguments for move: {:?}",ops)),
                    }
                },
                "addi" => return parse_arithmetic(vec![6,0], args),
                "subi" => return parse_arithmetic(vec![7,0], args),
                "muli" => return parse_arithmetic(vec![8,0], args),
                "divi" => return parse_arithmetic(vec![9,0], args),
                "eqi" => return parse_arithmetic(vec![10,0], args),
                "neqi" => return parse_arithmetic(vec![11,0], args),
                "lti" => return parse_arithmetic(vec![12,0], args),
                "gti" => return parse_arithmetic(vec![13,0], args),
                "leqi" => return parse_arithmetic(vec![14,0], args),
                "geqi" => return parse_arithmetic(vec![15,0], args),
                "addu" => return parse_arithmetic(vec![16,0], args),
                "subu" => return parse_arithmetic(vec![17,0], args),
                "mulu" => return parse_arithmetic(vec![18,0], args),
                "divu" => return parse_arithmetic(vec![19,0], args),
                "equ" => return parse_arithmetic(vec![20,0], args),
                "nequ" => return parse_arithmetic(vec![21,0], args),
                "ltu" => return parse_arithmetic(vec![22,0], args),
                "gtu" => return parse_arithmetic(vec![23,0], args),
                "lequ" => return parse_arithmetic(vec![24,0], args),
                "gequ" => return parse_arithmetic(vec![25,0], args),
                "movef" => {
                    let mut bytes: Vec<u8> = Vec::new();
                    let mut ops: Vec<MoveOps> = Vec::new();
                    for arg in args {
                        match arg {
                            Ast::Number(n) => {
                                bytes.append(&mut n.to_bytes());
                                ops.push(MoveOps::Number);
                            },
                            Ast::Register(r) => {
                                bytes.push(*r);
                                ops.push(MoveOps::Register);
                            },
                            Ast::Label(l) => {
                                let pos = pos_setter(l.to_owned());
                                bytes.append(&mut pos.to_le_bytes().to_vec());
                                ops.push(MoveOps::Number);
                            },
                            Ast::Address(a) => {
                                bytes.append(&mut a.to_le_bytes().to_vec());
                                ops.push(MoveOps::Address);
                            },
                            _ => return Err("Expected number, register or label".to_owned()),
                        }
                    }

                    match ops.as_slice() {
                        //move opcode
                        [MoveOps::Number, MoveOps::Address, MoveOps::Register] => {
                            let mut temp = vec![28,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        //set opcode
                        [MoveOps::Number, MoveOps::Register, MoveOps::Number] => {
                            let mut temp = vec![29,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        //regmove opcode
                        [MoveOps::Number, MoveOps::Register, MoveOps::Register] => {
                            let mut temp = vec![160,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        //deref opcode
                        [MoveOps::Number, MoveOps::Register, MoveOps::Address] => {
                            let mut temp = vec![27,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        //derefreg
                        [MoveOps::Number, MoveOps::Register, MoveOps::Register, MoveOps::Number] => {
                            let mut temp = vec![26,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        _ => return Err("Invalid arguments for move".to_owned()),
                    }
                },
                "addf" => return parse_arithmetic(vec![30,0], args),
                "subf" => return parse_arithmetic(vec![31,0], args),
                "mulf" => return parse_arithmetic(vec![32,0], args),
                "divf" => return parse_arithmetic(vec![33,0], args),
                "eqf" => return parse_arithmetic(vec![34,0], args),
                "neqf" => return parse_arithmetic(vec![35,0], args),
                "ltf" => return parse_arithmetic(vec![36,0], args),
                "gtf" => return parse_arithmetic(vec![37,0], args),
                "leqf" => return parse_arithmetic(vec![38,0], args),
                "geqf" => return parse_arithmetic(vec![39,0], args),
                "movea" => {
                    let mut bytes: Vec<u8> = Vec::new();
                    let mut ops: Vec<MoveOps> = Vec::new();
                    for arg in args {
                        match arg {
                            Ast::Number(n) => {
                                bytes.append(&mut n.to_bytes());
                                ops.push(MoveOps::Number);
                            },
                            Ast::Register(r) => {
                                bytes.push(*r);
                                ops.push(MoveOps::Register);
                            },
                            Ast::Label(l) => {
                                let pos = pos_setter(l.to_owned());
                                bytes.append(&mut pos.to_le_bytes().to_vec());
                                ops.push(MoveOps::Number);
                            },
                            Ast::Address(a) => {
                                bytes.append(&mut a.to_le_bytes().to_vec());
                                ops.push(MoveOps::Address);
                            },
                            _ => return Err("Expected number, register or label".to_owned()),
                        }
                    }

                    match ops.as_slice() {
                        //move opcode
                        [MoveOps::Number, MoveOps::Address, MoveOps::Register] => {
                            let mut temp = vec![42,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        //set opcode
                        [MoveOps::Number, MoveOps::Register, MoveOps::Number] => {
                            let mut temp = vec![43,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        //regmove opcode
                        [MoveOps::Number, MoveOps::Register, MoveOps::Register] => {
                            let mut temp = vec![161,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        //deref opcode
                        [MoveOps::Number, MoveOps::Register, MoveOps::Address] => {
                            let mut temp = vec![41,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        //derefreg
                        [MoveOps::Number, MoveOps::Register, MoveOps::Register, MoveOps::Number] => {
                            let mut temp = vec![40,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        _ => return Err("Invalid arguments for move".to_owned()),
                    }
                },
                "addai" => return parse_arithmetic(vec![44,0], args),
                "subai" => return parse_arithmetic(vec![45,0], args),
                "mulai" => return parse_arithmetic(vec![46,0], args),
                "divai" => return parse_arithmetic(vec![47,0], args),
                "eqai" => return parse_arithmetic(vec![48,0], args),
                "neqai" => return parse_arithmetic(vec![49,0], args),
                "ltai" => return parse_arithmetic(vec![50,0], args),
                "gtai" => return parse_arithmetic(vec![51,0], args),
                "leqai" => return parse_arithmetic(vec![52,0], args),
                "geqai" => return parse_arithmetic(vec![53,0], args),
                "addau" => return parse_arithmetic(vec![54,0], args),
                "subau" => return parse_arithmetic(vec![55,0], args),
                "mulau" => return parse_arithmetic(vec![56,0], args),
                "divau" => return parse_arithmetic(vec![57,0], args),
                "eqau" => return parse_arithmetic(vec![58,0], args),
                "neqau" => return parse_arithmetic(vec![59,0], args),
                "ltau" => return parse_arithmetic(vec![60,0], args),
                "gtau" => return parse_arithmetic(vec![61,0], args),
                "leqau" => return parse_arithmetic(vec![62,0], args),
                "geqau" => return parse_arithmetic(vec![63,0], args),
                "and" => return parse_arithmetic(vec![64,0], args),
                "or" => return parse_arithmetic(vec![65,0], args),
                "xor" => return parse_arithmetic(vec![66,0], args),
                "not" => return parse_arithmetic(vec![67,0], args),
                "shl" => return parse_arithmetic(vec![68,0], args),
                "shr" => return parse_arithmetic(vec![69,0], args),
                "jump" => return parse_jump(vec![70,0], args, pos_setter),
                "jumpeq" => return parse_jump(vec![71,0], args, pos_setter),
                "jumpneq" => return parse_jump(vec![72,0], args, pos_setter),
                "jumplt" => return parse_jump(vec![73,0], args, pos_setter),
                "jumpgt" => return parse_jump(vec![74,0], args, pos_setter),
                "jumpleq" => return parse_jump(vec![75,0], args, pos_setter),
                "jumpgeq" => return parse_jump(vec![76,0], args, pos_setter),
                "jumpzero" => return parse_jump(vec![77,0], args, pos_setter),
                "jumpnzero" => return parse_jump(vec![78,0], args, pos_setter),
                "jumpneg" => return parse_jump(vec![79,0], args, pos_setter),
                "jumppos" => return parse_jump(vec![80,0], args, pos_setter),
                "jumpeven" => return parse_jump(vec![81,0], args, pos_setter),
                "jumpodd" => return parse_jump(vec![82,0], args, pos_setter),
                "jumpback" => return parse_jump(vec![83,0], args, pos_setter),
                "jumpforward" => return parse_jump(vec![84,0], args, pos_setter),
                "jumpinf" => return parse_jump(vec![85,0], args, pos_setter),
                "jumpninf" => return parse_jump(vec![86,0], args, pos_setter),
                "jumpoverflow" => return parse_jump(vec![87,0], args, pos_setter),
                "jumpunderflow" => return parse_jump(vec![88,0], args, pos_setter),
                "jumpnoverflow" => return parse_jump(vec![89,0], args, pos_setter),
                "jumpnunderflow" => return parse_jump(vec![90,0], args, pos_setter),
                "jumpnan" => return parse_jump(vec![91,0], args, pos_setter),
                "jumpnnan" => return parse_jump(vec![92,0], args, pos_setter),
                "jumprmdr" => return parse_jump(vec![93,0], args, pos_setter),
                "jumpnrmdr" => return parse_jump(vec![94,0], args, pos_setter),
                "call" => return parse_jump(vec![109,0], args, pos_setter),
                "ret" => return Ok(vec![110,0]),
                "pop" => {
                    let mut bytes: Vec<u8> = Vec::new();
                    let mut ops: Vec<MoveOps> = Vec::new();
                    for arg in args {
                        match arg {
                            Ast::Number(n) => {
                                bytes.append(&mut n.to_bytes());
                                ops.push(MoveOps::Number);
                            },
                            Ast::Register(r) => {
                                bytes.push(*r);
                                ops.push(MoveOps::Register);
                            },
                            _ => return Err("Expected number or register".to_owned()),
                        }
                    }

                    match ops.as_slice() {
                        [MoveOps::Number, MoveOps::Register] => {
                            let mut temp = vec![112,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        _ => return Err("Invalid arguments for pop".to_owned()),
                    }
                },
                "push" => {
                    let mut bytes: Vec<u8> = Vec::new();
                    let mut ops: Vec<MoveOps> = Vec::new();
                    for arg in args {
                        match arg {
                            Ast::Number(n) => {
                                bytes.append(&mut n.to_bytes());
                                ops.push(MoveOps::Number);
                            },
                            Ast::Register(r) => {
                                bytes.push(*r);
                                ops.push(MoveOps::Register);
                            },
                            _ => return Err("Expected number or register".to_owned()),
                        }
                    }

                    match ops.as_slice() {
                        [MoveOps::Number, MoveOps::Register] => {
                            let mut temp = vec![112,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        _ => return Err("Invalid arguments for push".to_owned()),
                    }
                },
                "popf" => {
                    let mut bytes: Vec<u8> = Vec::new();
                    let mut ops: Vec<MoveOps> = Vec::new();
                    for arg in args {
                        match arg {
                            Ast::Number(n) => {
                                bytes.append(&mut n.to_bytes());
                                ops.push(MoveOps::Number);
                            },
                            Ast::Register(r) => {
                                bytes.push(*r);
                                ops.push(MoveOps::Register);
                            },
                            _ => return Err("Expected number or register".to_owned()),
                        }
                    }

                    match ops.as_slice() {
                        [MoveOps::Number, MoveOps::Register] => {
                            let mut temp = vec![113,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        _ => return Err("Invalid arguments for popf".to_owned()),
                    }
                },
                "pushf" => {
                    let mut bytes: Vec<u8> = Vec::new();
                    let mut ops: Vec<MoveOps> = Vec::new();
                    for arg in args {
                        match arg {
                            Ast::Number(n) => {
                                bytes.append(&mut n.to_bytes());
                                ops.push(MoveOps::Number);
                            },
                            Ast::Register(r) => {
                                bytes.push(*r);
                                ops.push(MoveOps::Register);
                            },
                            _ => return Err("Expected number or register".to_owned()),
                        }
                    }

                    match ops.as_slice() {
                        [MoveOps::Number, MoveOps::Register] => {
                            let mut temp = vec![114,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        _ => return Err("Invalid arguments for pushf".to_owned()),
                    }
                },
                "popa" => {
                    let mut bytes: Vec<u8> = Vec::new();
                    let mut ops: Vec<MoveOps> = Vec::new();
                    for arg in args {
                        match arg {
                            Ast::Number(n) => {
                                bytes.append(&mut n.to_bytes());
                                ops.push(MoveOps::Number);
                            },
                            Ast::Register(r) => {
                                bytes.push(*r);
                                ops.push(MoveOps::Register);
                            },
                            _ => return Err("Expected number or register".to_owned()),
                        }
                    }

                    match ops.as_slice() {
                        [MoveOps::Number, MoveOps::Register] => {
                            let mut temp = vec![115,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        _ => return Err("Invalid arguments for popa".to_owned()),
                    }
                },
                "pusha" => {
                    let mut bytes: Vec<u8> = Vec::new();
                    let mut ops: Vec<MoveOps> = Vec::new();
                    for arg in args {
                        match arg {
                            Ast::Number(n) => {
                                bytes.append(&mut n.to_bytes());
                                ops.push(MoveOps::Number);
                            },
                            Ast::Register(r) => {
                                bytes.push(*r);
                                ops.push(MoveOps::Register);
                            },
                            _ => return Err("Expected number or register".to_owned()),
                        }
                    }

                    match ops.as_slice() {
                        [MoveOps::Number, MoveOps::Register] => {
                            let mut temp = vec![116,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        _ => return Err("Invalid arguments for pushf".to_owned()),
                    }
                },
                "movestack" => {
                    let mut bytes: Vec<u8> = Vec::new();
                    let mut ops: Vec<MoveOps> = Vec::new();
                    for arg in args {
                        match arg {
                            Ast::Number(n) => {
                                bytes.append(&mut n.to_bytes());
                                ops.push(MoveOps::Number);
                            },
                            Ast::Register(r) => {
                                bytes.push(*r);
                                ops.push(MoveOps::Register);
                            },
                            Ast::Label(l) => {
                                let pos = pos_setter(l.to_owned());
                                bytes.append(&mut pos.to_le_bytes().to_vec());
                                ops.push(MoveOps::Number);
                            },
                            Ast::Address(a) => {
                                bytes.append(&mut a.to_le_bytes().to_vec());
                                ops.push(MoveOps::Address);
                            },
                            _ => return Err("Expected number, register or label".to_owned()),
                        }
                    }

                    match ops.as_slice() {
                        //move opcode
                        [MoveOps::Number, MoveOps::Address, MoveOps::Register] => {
                            let mut temp = vec![119,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        //deref opcode
                        [MoveOps::Number, MoveOps::Register, MoveOps::Address] => {
                            let mut temp = vec![118,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        //derefreg
                        [MoveOps::Number, MoveOps::Register, MoveOps::Register, MoveOps::Number] => {
                            let mut temp = vec![117,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        _ => return Err("Invalid arguments for movestack".to_owned()),
                    }
                },
                "movestackf" => {
                    let mut bytes: Vec<u8> = Vec::new();
                    let mut ops: Vec<MoveOps> = Vec::new();
                    for arg in args {
                        match arg {
                            Ast::Number(n) => {
                                bytes.append(&mut n.to_bytes());
                                ops.push(MoveOps::Number);
                            },
                            Ast::Register(r) => {
                                bytes.push(*r);
                                ops.push(MoveOps::Register);
                            },
                            Ast::Label(l) => {
                                let pos = pos_setter(l.to_owned());
                                bytes.append(&mut pos.to_le_bytes().to_vec());
                                ops.push(MoveOps::Number);
                            },
                            Ast::Address(a) => {
                                bytes.append(&mut a.to_le_bytes().to_vec());
                                ops.push(MoveOps::Address);
                            },
                            _ => return Err("Expected number, register or label".to_owned()),
                        }
                    }

                    match ops.as_slice() {
                        //move opcode
                        [MoveOps::Number, MoveOps::Address, MoveOps::Register] => {
                            let mut temp = vec![122,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        //deref opcode
                        [MoveOps::Number, MoveOps::Register, MoveOps::Address] => {
                            let mut temp = vec![121,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        //derefreg
                        [MoveOps::Number, MoveOps::Register, MoveOps::Register, MoveOps::Number] => {
                            let mut temp = vec![120,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        _ => return Err("Invalid arguments for movestackf".to_owned()),
                    }
                },
                "movestacka" => {
                    let mut bytes: Vec<u8> = Vec::new();
                    let mut ops: Vec<MoveOps> = Vec::new();
                    for arg in args {
                        match arg {
                            Ast::Number(n) => {
                                bytes.append(&mut n.to_bytes());
                                ops.push(MoveOps::Number);
                            },
                            Ast::Register(r) => {
                                bytes.push(*r);
                                ops.push(MoveOps::Register);
                            },
                            Ast::Label(l) => {
                                let pos = pos_setter(l.to_owned());
                                bytes.append(&mut pos.to_le_bytes().to_vec());
                                ops.push(MoveOps::Number);
                            },
                            Ast::Address(a) => {
                                bytes.append(&mut a.to_le_bytes().to_vec());
                                ops.push(MoveOps::Address);
                            },
                            _ => return Err("Expected number, register or label".to_owned()),
                        }
                    }

                    match ops.as_slice() {
                        //move opcode
                        [MoveOps::Number, MoveOps::Address, MoveOps::Register] => {
                            let mut temp = vec![125,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        //deref opcode
                        [MoveOps::Number, MoveOps::Register, MoveOps::Address] => {
                            let mut temp = vec![124,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        //derefreg
                        [MoveOps::Number, MoveOps::Register, MoveOps::Register, MoveOps::Number] => {
                            let mut temp = vec![123,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        _ => return Err("Invalid arguments for movestacka".to_owned()),
                    }
                },
                "movecstack" => {
                    let mut bytes: Vec<u8> = Vec::new();
                    let mut ops: Vec<MoveOps> = Vec::new();
                    for arg in args {
                        match arg {
                            Ast::Number(n) => {
                                bytes.append(&mut n.to_bytes());
                                ops.push(MoveOps::Number);
                            },
                            Ast::Register(r) => {
                                bytes.push(*r);
                                ops.push(MoveOps::Register);
                            },
                            Ast::Label(l) => {
                                let pos = pos_setter(l.to_owned());
                                bytes.append(&mut pos.to_le_bytes().to_vec());
                                ops.push(MoveOps::Number);
                            },
                            Ast::Address(a) => {
                                bytes.append(&mut a.to_le_bytes().to_vec());
                                ops.push(MoveOps::Address);
                            },
                            _ => return Err("Expected number, register or label".to_owned()),
                        }
                    }

                    match ops.as_slice() {
                        //move opcode
                        [MoveOps::Number, MoveOps::Number, MoveOps::Address, MoveOps::Register] => {
                            let mut temp = vec![128,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        //deref opcode
                        [MoveOps::Number, MoveOps::Number, MoveOps::Register, MoveOps::Address] => {
                            let mut temp = vec![127,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        //derefreg
                        [MoveOps::Number, MoveOps::Number, MoveOps::Register, MoveOps::Register, MoveOps::Number] => {
                            let mut temp = vec![126,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        _ => return Err("Invalid arguments for movecstack".to_owned()),
                    }
                },
                "movecstackf" => {
                    let mut bytes: Vec<u8> = Vec::new();
                    let mut ops: Vec<MoveOps> = Vec::new();
                    for arg in args {
                        match arg {
                            Ast::Number(n) => {
                                bytes.append(&mut n.to_bytes());
                                ops.push(MoveOps::Number);
                            },
                            Ast::Register(r) => {
                                bytes.push(*r);
                                ops.push(MoveOps::Register);
                            },
                            Ast::Label(l) => {
                                let pos = pos_setter(l.to_owned());
                                bytes.append(&mut pos.to_le_bytes().to_vec());
                                ops.push(MoveOps::Number);
                            },
                            Ast::Address(a) => {
                                bytes.append(&mut a.to_le_bytes().to_vec());
                                ops.push(MoveOps::Address);
                            },
                            _ => return Err("Expected number, register or label".to_owned()),
                        }
                    }

                    match ops.as_slice() {
                        //move opcode
                        [MoveOps::Number, MoveOps::Number, MoveOps::Address, MoveOps::Register] => {
                            let mut temp = vec![131,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        //deref opcode
                        [MoveOps::Number, MoveOps::Number, MoveOps::Register, MoveOps::Address] => {
                            let mut temp = vec![130,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        //derefreg
                        [MoveOps::Number, MoveOps::Number, MoveOps::Register, MoveOps::Register, MoveOps::Number] => {
                            let mut temp = vec![129,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        _ => return Err("Invalid arguments for movecstackf".to_owned()),
                    }
                },
                "movecorestacka" => {
                    let mut bytes: Vec<u8> = Vec::new();
                    let mut ops: Vec<MoveOps> = Vec::new();
                    for arg in args {
                        match arg {
                            Ast::Number(n) => {
                                bytes.append(&mut n.to_bytes());
                                ops.push(MoveOps::Number);
                            },
                            Ast::Register(r) => {
                                bytes.push(*r);
                                ops.push(MoveOps::Register);
                            },
                            Ast::Label(l) => {
                                let pos = pos_setter(l.to_owned());
                                bytes.append(&mut pos.to_le_bytes().to_vec());
                                ops.push(MoveOps::Number);
                            },
                            Ast::Address(a) => {
                                bytes.append(&mut a.to_le_bytes().to_vec());
                                ops.push(MoveOps::Address);
                            },
                            _ => return Err("Expected number, register or label".to_owned()),
                        }
                    }

                    match ops.as_slice() {
                        //move opcode
                        [MoveOps::Number, MoveOps::Number, MoveOps::Address, MoveOps::Register] => {
                            let mut temp = vec![134,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        //deref opcode
                        [MoveOps::Number, MoveOps::Number, MoveOps::Register, MoveOps::Address] => {
                            let mut temp = vec![133,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        //derefreg
                        [MoveOps::Number, MoveOps::Number, MoveOps::Register, MoveOps::Register, MoveOps::Number] => {
                            let mut temp = vec![132,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        _ => return Err("Invalid arguments for movecstacka".to_owned()),
                    }
                },
                "malloc" => {
                    let mut bytes: Vec<u8> = Vec::new();
                    let mut ops: Vec<MoveOps> = Vec::new();
                    for arg in args {
                        match arg {
                            Ast::Register(r) => {
                                bytes.push(*r);
                                ops.push(MoveOps::Register);
                            },
                            _ => return Err("Expected only registers".to_owned()),
                        }
                    }

                    match ops.as_slice() {
                        [MoveOps::Register, MoveOps::Register] => {
                            let mut temp = vec![135,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        _ => return Err("Invalid arguments for malloc".to_owned()),
                    }
                },
                "free" => {
                    let mut bytes: Vec<u8> = Vec::new();
                    let mut ops: Vec<MoveOps> = Vec::new();
                    for arg in args {
                        match arg {
                            Ast::Register(r) => {
                                bytes.push(*r);
                                ops.push(MoveOps::Register);
                            },
                            _ => return Err("Expected only registers".to_owned()),
                        }
                    }

                    match ops.as_slice() {
                        [MoveOps::Register] => {
                            let mut temp = vec![136,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        _ => return Err("Invalid arguments for free".to_owned()),
                    }
                },
                "readbyte" => {
                    let mut bytes: Vec<u8> = Vec::new();
                    let mut ops: Vec<MoveOps> = Vec::new();
                    for arg in args {
                        match arg {
                            Ast::Register(r) => {
                                bytes.push(*r);
                                ops.push(MoveOps::Register);
                            },
                            _ => return Err("Expected only registers".to_owned()),
                        }
                    }

                    match ops.as_slice() {
                        [MoveOps::Register, MoveOps::Register] => {
                            let mut temp = vec![137,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        _ => return Err("Invalid arguments for readbyte".to_owned()),
                    }
                },
                "read" => {
                    let mut bytes: Vec<u8> = Vec::new();
                    let mut ops: Vec<MoveOps> = Vec::new();
                    for arg in args {
                        match arg {
                            Ast::Register(r) => {
                                bytes.push(*r);
                                ops.push(MoveOps::Register);
                            },
                            _ => return Err("Expected only registers".to_owned()),
                        }
                    }

                    match ops.as_slice() {
                        [MoveOps::Register, MoveOps::Register, MoveOps::Register] => {
                            let mut temp = vec![138,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        _ => return Err("Invalid arguments for read".to_owned()),
                    }
                },
                "writebyte" => {
                    let mut bytes: Vec<u8> = Vec::new();
                    let mut ops: Vec<MoveOps> = Vec::new();
                    for arg in args {
                        match arg {
                            Ast::Register(r) => {
                                bytes.push(*r);
                                ops.push(MoveOps::Register);
                            },
                            _ => return Err("Expected only registers".to_owned()),
                        }
                    }

                    match ops.as_slice() {
                        [MoveOps::Register, MoveOps::Register] => {
                            let mut temp = vec![139,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        _ => return Err("Invalid arguments for writebyte".to_owned()),
                    }
                },
                "write" => {
                    let mut bytes: Vec<u8> = Vec::new();
                    let mut ops: Vec<MoveOps> = Vec::new();
                    for arg in args {
                        match arg {
                            Ast::Register(r) => {
                                bytes.push(*r);
                                ops.push(MoveOps::Register);
                            },
                            _ => return Err("Expected only registers".to_owned()),
                        }
                    }

                    match ops.as_slice() {
                        [MoveOps::Register, MoveOps::Register, MoveOps::Register] => {
                            let mut temp = vec![140,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        _ => return Err("Invalid arguments for write".to_owned()),
                    }
                },
                "open" => {
                    let mut bytes: Vec<u8> = Vec::new();
                    let mut ops: Vec<MoveOps> = Vec::new();
                    for arg in args {
                        match arg {
                            Ast::Register(r) => {
                                bytes.push(*r);
                                ops.push(MoveOps::Register);
                            },
                            _ => return Err("Expected only registers".to_owned()),
                        }
                    }

                    match ops.as_slice() {
                        [MoveOps::Register, MoveOps::Register] => {
                            let mut temp = vec![141,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        _ => return Err("Invalid arguments for open".to_owned()),
                    }
                },
                "close" => {
                    let mut bytes: Vec<u8> = Vec::new();
                    let mut ops: Vec<MoveOps> = Vec::new();
                    for arg in args {
                        match arg {
                            Ast::Register(r) => {
                                bytes.push(*r);
                                ops.push(MoveOps::Register);
                            },
                            _ => return Err("Expected only registers".to_owned()),
                        }
                    }

                    match ops.as_slice() {
                        [MoveOps::Register] => {
                            let mut temp = vec![142,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        _ => return Err("Invalid arguments for close".to_owned()),
                    }
                },
                "flush" => {
                    let mut bytes: Vec<u8> = Vec::new();
                    let mut ops: Vec<MoveOps> = Vec::new();
                    for arg in args {
                        match arg {
                            Ast::Register(r) => {
                                bytes.push(*r);
                                ops.push(MoveOps::Register);
                            },
                            _ => return Err("Expected only registers".to_owned()),
                        }
                    }

                    match ops.as_slice() {
                        [MoveOps::Register] => {
                            let mut temp = vec![143,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        _ => return Err("Invalid arguments for flush".to_owned()),
                    }
                },
                "threadspawn" => {
                    let mut bytes: Vec<u8> = Vec::new();
                    let mut ops: Vec<MoveOps> = Vec::new();
                    for arg in args {
                        match arg {
                            Ast::Register(r) => {
                                bytes.push(*r);
                                ops.push(MoveOps::Register);
                            },
                            _ => return Err("Expected only registers".to_owned()),
                        }
                    }

                    match ops.as_slice() {
                        [MoveOps::Register] => {
                            let mut temp = vec![144,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        _ => return Err("Invalid arguments for threadspawn".to_owned()),
                    }
                },
                "remainder" => {
                    let mut bytes: Vec<u8> = Vec::new();
                    let mut ops: Vec<MoveOps> = Vec::new();
                    for arg in args {
                        match arg {
                            Ast::Number(n) => {
                                bytes.append(&mut n.to_bytes());
                                ops.push(MoveOps::Number);
                            },
                            Ast::Register(r) => {
                                bytes.push(*r);
                                ops.push(MoveOps::Register);
                            },
                            _ => return Err("Expected a size and a register".to_owned()),
                        }
                    }

                    match ops.as_slice() {
                        [MoveOps::Number, MoveOps::Register] => {
                            let mut temp = vec![145,0];
                            temp.append(&mut bytes);
                            return Ok(temp);
                        },
                        _ => return Err("Invalid arguments for threadspawn".to_owned()),
                    }
                },
                "clear" => return Ok(vec![146,0]),
                "addfi" => parse_arithmetic(vec![147,0], args),
                "subfi" => parse_arithmetic(vec![148,0], args),
                "mulfi" => parse_arithmetic(vec![149,0], args),
                "divfi" => parse_arithmetic(vec![150,0], args),
                "addif" => parse_arithmetic(vec![151,0], args),
                "subif" => parse_arithmetic(vec![152,0], args),
                "mulif" => parse_arithmetic(vec![153,0], args),
                "divif" => parse_arithmetic(vec![154,0], args),
                "adduf" => parse_arithmetic(vec![155,0], args),
                "subuf" => parse_arithmetic(vec![156,0], args),
                "muluf" => parse_arithmetic(vec![157,0], args),
                "divuf" => parse_arithmetic(vec![158,0], args),
                instr => return Err(format!("Invalid instruction: {}", instr)),
                
                    
                

                
                
                
            }
        },
        Ast::MemorySet(qualifier, value) => {
            match qualifier.as_str() {
                "string" => {
                    match value.as_ref() {
                        Ast::String(s) => {
                            let bytes = s.as_bytes().to_vec();
                            Ok(bytes)
                        },
                        _ => Err("Expected string".to_owned()),
                    }
                },
                "u8" => {
                    match value.as_ref() {
                        Ast::Number(n) => {
                            match n {
                                Number::U8(n) => {
                                    let bytes = vec![*n];
                                    Ok(bytes)
                                },
                                Number::I8(n) => {
                                    let bytes = vec![n.to_le_bytes()[0]];
                                    Ok(bytes)
                                },
                                _ => Err("Expected number".to_owned()),
                            }
                        },
                        _ => Err("Expected number".to_owned()),
                    }
                },
                "u16" => {
                    match value.as_ref() {
                        Ast::Number(n) => {
                            match n {
                                Number::U16(n) => {
                                    let bytes = n.to_le_bytes().to_vec();
                                    Ok(bytes)
                                },
                                Number::I16(n) => {
                                    let bytes = n.to_le_bytes().to_vec();
                                    Ok(bytes)
                                },
                                _ => Err("Expected number".to_owned()),
                            }
                        },
                        _ => Err("Expected number".to_owned()),
                    }
                },
                "u32" => {
                    match value.as_ref() {
                        Ast::Number(n) => {
                            match n {
                                Number::U32(n) => {
                                    let bytes = n.to_le_bytes().to_vec();
                                    Ok(bytes)
                                },
                                Number::I32(n) => {
                                    let bytes = n.to_le_bytes().to_vec();
                                    Ok(bytes)
                                },
                                _ => Err("Expected number".to_owned()),
                            }
                        },
                        _ => Err("Expected number".to_owned()),
                    }
                },
                "u64" => {
                    match value.as_ref() {
                        Ast::Number(n) => {
                            match n {
                                Number::U64(n) => {
                                    let bytes = n.to_le_bytes().to_vec();
                                    Ok(bytes)
                                },
                                Number::I64(n) => {
                                    let bytes = n.to_le_bytes().to_vec();
                                    Ok(bytes)
                                },
                                _ => Err("Expected number".to_owned()),
                            }
                        },
                        _ => Err("Expected number".to_owned()),
                    }
                },
                "u128" => {
                    match value.as_ref() {
                        Ast::Number(n) => {
                            match n {
                                Number::U128(n) => {
                                    let bytes = n.to_le_bytes().to_vec();
                                    Ok(bytes)
                                },
                                Number::I128(n) => {
                                    let bytes = n.to_le_bytes().to_vec();
                                    Ok(bytes)
                                },
                                _ => Err("Expected number".to_owned()),
                            }
                        },
                        _ => Err("Expected number".to_owned()),
                    }
                },
                _ => Err("Expected qualifier".to_owned()),
            }
        }
        _ => Err("Expected instruction".to_owned()),
    }

}

pub fn parse_file(input: &str) -> Result<(Vec<u8>,usize), String> {

    let parser = file_parser();

    let result = parser.parse(input);

    let mut result = match result {
        Ok(ast) => ast,
        Err(_) => return Err("Error parsing file".to_owned()),
    };


    let mut label_positions = HashMap::new();
    let mut bytes = Vec::new();


    match result {
        Ast::File(ref mut ast) => {
            for i in ast.iter_mut() {
                match i {
                    Ast::Labelled(label, instructions) => {
                        match label.as_ref() {
                            Ast::Label(name) => {
                                label_positions.insert(name.to_owned(), bytes.len());

                                for instruction in instructions.iter_mut() {
                                    let mut instruction_bytes = parse_instruction(instruction, |label| {
                                        label_positions.get(&label).unwrap().to_owned() as u64
                                    })?;
                                    bytes.append(&mut instruction_bytes);
                                }
                            },
                            _ => panic!("Expected label"),
                        }
                        
                    },
                    _ => (),
                }
            }
        },
        _ => (),
    }
    
    let main_pos = label_positions.get("main");

    match main_pos {
        Some(pos) => {
            Ok((bytes, *pos))
        },
        None => return Err("No main function".to_owned()),
    }
}





#[cfg(test)]
mod tests {
    use super::*;
    use crate::virtual_machine::Machine;

    #[test]
    fn test_hello_world_assembly() {
        let input = "LC{
 .string \"Hello, world!\"}
main{
move 64, $0, 1
move 64, $1, LC
move 64, $2, 13
write $0, $1, $2
flush $0
ret}";
        let (bytes, main_pos) = parse_file(input).unwrap();

        let mut vm = Machine::new_with_cores(1);

        vm.add_program(bytes);

        vm.run(main_pos);
        
        
    }


}

    
