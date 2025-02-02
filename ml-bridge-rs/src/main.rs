use base64::{engine::general_purpose, Engine as _};
use eerie::runtime::api::Session;
use serde::{Deserialize, Serialize};
use std::fs;

use eerie::runtime::api;
use eerie::runtime::hal;
use eerie::runtime::vm::*;
use eerie::runtime::*;

#[derive(Debug, Serialize, Deserialize)]
pub struct ExportData {
    pub name: String,
    pub input_format: TreeFormat,
    pub output_format: TreeFormat,
    pub sample_input_flat: Vec<serde_json::Value>,
    pub sample_output_flat: Vec<serde_json::Value>,
    pub compiled_module: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TreeFormat {
    pub tree: serde_json::Value,
    pub leaves: Vec<LeafInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LeafInfo {
    pub shape: Vec<u64>,
    pub dtype: String,
}

// fn leaf_info_to_type(leaf_info: &LeafInfo) -> Type {
//     // Map the dtype string (e.g. "float32") to a Rust numeric type.
//     let rust_primitive = match leaf_info.dtype.as_str() {
//         "float32" => "f32",
//         "float64" => "f64",
//         "int32" => "i32",
//         "int64" => "i64",
//         "bool" => "bool",
//         _ => "f32", // fallback or error handling
//     };

//     let mut shape_str = rust_primitive.to_string(); // Start with the primitive type (scalar case)
//     for dim in leaf_info.shape.iter().rev() {
//         // Add the dimensions to the type string
//         shape_str = format!("[{}; {}]", shape_str, dim);
//     }

//     // Return the Rust type string
//     Type::new(shape_str)
// }

pub struct RuntimeContext<'a, const IN_STORAGE_SIZE: usize, const OUT_STORAGE_SIZE: usize> {
    pub session: &'a Session<'a>,
    pub instance: &'a api::Instance,
    pub input_storage: [u8; IN_STORAGE_SIZE],
    pub output_storage: [u8; OUT_STORAGE_SIZE],
}

fn invoke<
    T,
    const NUM_INPUTS: usize,
    const NUM_OUTPUTS: usize,
    const INPUT_SIZE: usize,
    const OUTPUT_SIZE: usize,
    const IN_STORAGE_SIZE: usize,
    const OUT_STORAGE_SIZE: usize,
>(
    ctx: &mut RuntimeContext<IN_STORAGE_SIZE, OUT_STORAGE_SIZE>,
    function: &vm::Function,
    inputs: [&[T; INPUT_SIZE]; NUM_INPUTS],
    outputs: [&mut [T; OUTPUT_SIZE]; NUM_OUTPUTS],
) where
    T: hal::ToElementType + Copy,
{
    let input_span = base::ByteSpan::from(&mut ctx.input_storage[..]);
    let output_span = base::ByteSpan::from(&mut ctx.output_storage[..]);

    let input_list =
        StaticList::<Ref<hal::BufferView<T>>>::new(input_span, NUM_INPUTS, &ctx.instance).unwrap();
    let output_list =
        StaticList::<Ref<hal::BufferView<T>>>::new(output_span, NUM_OUTPUTS, &ctx.instance)
            .unwrap();

    for input_idx in 0..NUM_INPUTS {
        let input_buffer = hal::BufferView::<T>::new(
            ctx.session,
            &[INPUT_SIZE],
            hal::EncodingType::DenseRowMajor,
            inputs[input_idx],
        )
        .unwrap();
        let input_buffer_ref = input_buffer.to_ref(&ctx.instance).unwrap();
        input_list.push_ref(&input_buffer_ref).unwrap();
    }

    function.invoke(&input_list, &output_list).unwrap();

    for output_idx in 0..NUM_OUTPUTS {
        let output_buffer_ref = output_list.get_ref(output_idx).unwrap();
        let output_buffer: hal::BufferView<T> = output_buffer_ref.to_buffer_view(ctx.session);
        let output_mapping = hal::BufferMapping::new(output_buffer).unwrap();
        let out_vals = output_mapping.data().get(..).unwrap();
        outputs[output_idx].copy_from_slice(out_vals);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file_str = fs::read_to_string("../ml-bridge-py/export_data.json")?;
    let data: ExportData = serde_json::from_str(&file_str)?;
    // println!("{:#?}", data);

    let vmfb = general_purpose::STANDARD
        .decode(data.compiled_module)
        .unwrap();

    let instance = api::Instance::new(
        &api::InstanceOptions::new(&mut hal::DriverRegistry::new()).use_all_available_drivers(),
    )
    .unwrap();
    let device = instance
        .try_create_default_device("local-task")
        .expect("Failed to create device");
    let session =
        api::Session::create_with_device(&instance, &api::SessionOptions::default(), &device)
            .unwrap();
    unsafe { session.append_module_from_memory(vmfb.as_slice()) }.unwrap();

    let function = session.lookup_function("jit_test_fn.main").unwrap();

    let inp1: [f32; 3] = [1.0, 2.0, 3.0];
    let inp2: [f32; 3] = [4.0, 5.0, 6.0];
    let mut out: [f32; 3] = [0.0; 3];

    let inputs = [&inp1, &inp2];
    let outputs = [&mut out];

    {
        let mut ctx = RuntimeContext::<16384, 16384> {
            session: &session,
            instance: &instance,
            input_storage: [0u8; 16384],
            output_storage: [0u8; 16384],
        };
        let start = std::time::Instant::now();
        invoke(&mut ctx, &function, inputs, outputs);
        let duration = start.elapsed();
        println!("Execution took: {:?}", duration);
    }
    println!("out:\t{:?}", out);

    Ok(())
}
