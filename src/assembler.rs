use chumsky::prelude::*;

pub enum Number {
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


pub enum Ast {
    Instruction(String, Vec<Ast>),
    Register(u8),
    Number(Number),
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
    let memory = just('.')
        .ignore_then(choice((
            just("u8").to("u8"),
            just("u16").to("u16"),
            just("u32").to("u32"),
            just("u64").to("u64"),
            just("u128").to("u128"),
            just("i8").to("i8"),
            just("i16").to("i16"),
            just("i32").to("i32"),
            just("i64").to("i64"),
            just("i128").to("i128"),
            just("f32").to("f32"),
            just("f64").to("f64"),
            just("string").to("string"),
        )).padded())
        .then(choice((number_parser(), string_parser())).padded())
        .map(|(t, v)| Ast::MemorySet(t.to_owned(), Box::new(v)));

    memory

}

fn op_parser() -> impl Parser<char, Ast, Error = Simple<char>> {
    let instruction = one_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ").repeated().at_least(3).padded()
        .then(choice((register_parser(), label_parser(), number_parser())).padded()
            .separated_by(just(",").padded()))
        .map(|(chars, args)| Ast::Instruction(chars.into_iter().collect(), args));


    instruction
}

fn register_parser() -> impl Parser<char, Ast, Error = Simple<char>> {
    let register = just("$")
        .ignore_then(raw_number_parser())
        .map(|n| match n {
            Number::U8(i) => Ast::Register(i),
            Number::U16(i) => Ast::Register(i as u8),
            Number::U32(i) => Ast::Register(i as u8),
            Number::U64(i) => Ast::Register(i as u8),
            Number::U128(i) => Ast::Register(i as u8),
            _ => panic!("Invalid register"),
        });

    register
}

fn label_parser() -> impl Parser<char, Ast, Error = Simple<char>> {
    let label = none_of(" \r\n\t").repeated()
        .map(|chars| Ast::Label(chars.into_iter().collect()));

    label
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
        .then_ignore(end())
        .map(|s| Ast::String(s.iter().collect()));
    

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
            _ => Number::I64(i.parse::<i64>().unwrap()),
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
        

    let number = choice((float, integer));

        
    number
}

fn number_parser() -> impl Parser<char, Ast, Error = Simple<char>> {
    raw_number_parser().map(|n| Ast::Number(n))
}



fn labelled_parser() -> impl Parser<char, Ast, Error = Simple<char>> {
    let parser = label_parser()
        .then_ignore(just(":").padded())
        .then(instruction_parser().repeated().padded())
        .map(|(l, i)| Ast::Labelled(Box::new(l), i));

    parser
}

fn file_parser() -> impl Parser<char, Ast, Error = Simple<char>> {
    let parser = labelled_parser().repeated().padded()
        .map(|i| Ast::File(i));

    parser
}
