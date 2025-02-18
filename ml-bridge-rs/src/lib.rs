use proc_macro::TokenStream;
use quote::{format_ident, quote};
use serde_json::Value;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, Ident, LitStr, Token,
};

mod helpers;
use crate::helpers::{parse_leaf, value_to_ref_tuple_tokens, value_to_tuple_tokens};

struct LoadExportedFnArgs {
    name: Ident,
    _comma: Token![,],
    path: LitStr,
}

impl Parse for LoadExportedFnArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(LoadExportedFnArgs {
            name: input.parse()?,
            _comma: input.parse()?,
            path: input.parse()?,
        })
    }
}

#[proc_macro]
pub fn load_exported_fn(input: TokenStream) -> TokenStream {
    // Parse macro input: an identifier for the module name and a string literal for the JSON file path.
    let args = parse_macro_input!(input as LoadExportedFnArgs);
    let module_name = args.name;
    let file_path = args.path.value();

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let full_path = std::path::PathBuf::from(manifest_dir).join(&file_path);
    let full_path_str = full_path.to_str().unwrap();

    let json_str = std::fs::read_to_string(&full_path).expect("Failed to read JSON file");

    // Parse the JSON.
    let json_value: Value = serde_json::from_str(&json_str).expect("Failed to parse JSON input");

    let sample_input_tokens = value_to_ref_tuple_tokens(
        json_value
            .get("sample_input_flat")
            .expect("Expected 'sample_input_flat' in JSON"),
    );
    let sample_output_tokens = value_to_tuple_tokens(
        json_value
            .get("sample_output_flat")
            .expect("Expected 'sample_output_flat' in JSON"),
    );

    // Process input leaves.
    let input_leaves = json_value
        .get("input_format")
        .and_then(|v| v.get("leaves"))
        .and_then(|v| v.as_array())
        .expect("Could not find 'input_format.leaves' as an array");
    let input_specs: Vec<(syn::Type, Vec<syn::LitInt>, syn::Type)> =
        input_leaves.iter().map(parse_leaf).collect();

    // Process output leaves.
    let output_leaves = json_value
        .get("output_format")
        .and_then(|v| v.get("leaves"))
        .and_then(|v| v.as_array())
        .expect("Could not find 'output_format.leaves' as an array");
    let output_specs: Vec<(syn::Type, Vec<syn::LitInt>, syn::Type)> =
        output_leaves.iter().map(parse_leaf).collect();

    let num_inputs = input_specs.len();
    let num_outputs = output_specs.len();

    // Build parameter types for the function: use the nested type (param_ty).
    let input_types = input_specs.iter().map(|(param_ty, _dims, _base)| {
        quote! { & #param_ty }
    });
    let output_types = output_specs.iter().map(|(param_ty, _dims, _base)| {
        quote! { #param_ty }
    });

    // Generate identifiers for each input (i0, i1, …)
    let input_idents: Vec<_> = (0..num_inputs).map(|i| format_ident!("i{}", i)).collect();

    // For each input, generate code to create a BufferView with the proper base type.
    let input_buffer_code =
        input_specs
            .iter()
            .enumerate()
            .map(|(i, (_param_ty, dims, base_ty))| {
                // Create a literal slice for the shape.
                let shape_tokens = quote! { &[ #( #dims as usize ),* ] };
                let ident = &input_idents[i];
                quote! {
                    {
                        // Use the base type for constructing the BufferView.
                        let input_buffer = eerie::runtime::hal::BufferView::<#base_ty>::new(
                            session,
                            #shape_tokens,
                            eerie::runtime::hal::EncodingType::DenseRowMajor,
                            flatten_array(#ident)
                        ).unwrap();
                        let input_buffer_ref = input_buffer.to_ref(instance).unwrap();
                        input_list.push_ref(&input_buffer_ref).unwrap();
                    }
                }
            });

    // For each output, generate code to extract its value.
    let output_buffer_code = output_specs.iter().enumerate().map(|(i, (_param_ty, dims, base_ty))| {
        let out_ident = format_ident!("out{}", i);
        if dims.is_empty() {
            quote! {
                let #out_ident = {
                    let buf_ref = output_list.get_ref(#i).unwrap();
                    let buf: eerie::runtime::hal::BufferView<#base_ty> = buf_ref.to_buffer_view(session);
                    let mapping = eerie::runtime::hal::BufferMapping::new(buf).unwrap();
                    mapping.data()[0]
                };
            }
        } else {
            quote! {
                let #out_ident = {
                    let buf_ref = output_list.get_ref(#i).unwrap();
                    let buf: eerie::runtime::hal::BufferView<#base_ty> = buf_ref.to_buffer_view(session);
                    let mapping = eerie::runtime::hal::BufferMapping::new(buf).unwrap();
                    mapping.data().try_into().unwrap()
                };
            }
        }
    }).collect::<Vec<_>>();

    // Build the tuple to return.
    let output_tuple = {
        let idents = (0..num_outputs).map(|i| {
            let ident = format_ident!("out{}", i);
            quote! { #ident }
        });
        quote! { ( #( #idents ),* ) }
    };

    // Get the compiled module string.
    let compiled_module = json_value
        .get("compiled_module")
        .and_then(|v| v.as_str())
        .expect("Expected 'compiled_module' to be a string");

    // Get the function name.
    let fun_name_str = json_value
        .get("name")
        .and_then(|v| v.as_str())
        .expect("Expected 'name' to be a string");

    let jit_fun_str = format!("jit_{}.main", fun_name_str);

    // Generate the full function.
    let expanded = quote! {
        mod #module_name {
            use base64::{engine::general_purpose, Engine as _};
            use eerie::runtime::vm::*;

            // Dummy const that forces the JSON file to be tracked for rebuilds.
            const _FOR_REBUILD_TRACKING: &str = include_str!(#full_path_str);

            fn flatten_array<T, U>(nested: &U) -> &[T] {
                unsafe {
                    let ptr = nested as *const U as *const T;
                    let len = std::mem::size_of_val(nested) / std::mem::size_of::<T>();
                    std::slice::from_raw_parts(ptr, len)
                }
            }

            pub fn register(
                instance: &eerie::runtime::api::Instance,
                session: &eerie::runtime::api::Session,
            ) {
                let module_bytes = base64::engine::general_purpose::STANDARD.decode(#compiled_module)
                    .expect("Failed to decode the compiled module");
                unsafe {
                    session.append_module_from_memory(module_bytes.as_slice())
                }.unwrap();
            }

            pub fn call(
                instance: &eerie::runtime::api::Instance,
                session: &eerie::runtime::api::Session,
                inputs: ( #( #input_types ),* )
            ) -> ( #( #output_types ),* ) {
                let function = session.lookup_function(#jit_fun_str)
                    .expect("Function lookup failed");

                let ( #( #input_idents ),* ) = inputs;

                let mut input_storage = [0u8; 16384];
                let mut output_storage = [0u8; 16384];
                let input_span = eerie::runtime::base::ByteSpan::from(&mut input_storage[..]);
                let output_span = eerie::runtime::base::ByteSpan::from(&mut output_storage[..]);

                // Create the input StaticList.
                let mut input_list = eerie::runtime::vm::StaticList::<eerie::runtime::vm::Ref<eerie::runtime::hal::BufferView<f32>>>::new(
                    input_span, #num_inputs, instance
                ).unwrap();
                // Create the output StaticList.
                let mut output_list = eerie::runtime::vm::StaticList::<eerie::runtime::vm::Ref<eerie::runtime::hal::BufferView<f32>>>::new(
                    output_span, #num_outputs, instance
                ).unwrap();

                {
                    #( #input_buffer_code )*
                }

                function.invoke(&input_list, &output_list)
                    .expect("Function invocation failed");

                #( #output_buffer_code )*

                #output_tuple
            }

            #[cfg(test)]
            mod tests {
                use super::*;
                use eerie::runtime::api;
                use eerie::runtime::hal;

                #[test]
                fn call_generated_fn() {
                    // Create an instance and session.
                    let instance = api::Instance::new(
                        &api::InstanceOptions::new(&mut hal::DriverRegistry::new())
                            .use_all_available_drivers(),
                    ).expect("Failed to create instance");
                    let device = instance
                        .try_create_default_device("local-task")
                        .expect("Failed to create device");
                    let session = api::Session::create_with_device(
                        &instance,
                        &api::SessionOptions::default(),
                        &device
                    ).expect("Failed to create session");

                    // Setup the function.
                    register(&instance, &session);

                    // Build the sample input tuple from the JSON using embedded literals.
                    let sample_input = (#sample_input_tokens);

                    // Call the generated function.
                    let output = call(&instance, &session, sample_input);

                    // The expected output as provided in the JSON.
                    let expected_output = (#sample_output_tokens);

                    // Assertions: perform any appropriate comparisons here.
                    assert_eq!(output, expected_output);
                }
            }
        }
    };

    // Uncomment to print the generated code.
    // println!("{}", expanded);

    TokenStream::from(expanded)
}
