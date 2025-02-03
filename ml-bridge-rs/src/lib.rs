use proc_macro::TokenStream;
use quote::{format_ident, quote};
use serde_json::Value;
use syn::{parse_macro_input, LitStr};

#[proc_macro]
pub fn make_fn_from_json(input: TokenStream) -> TokenStream {
    // Get the JSON literal string.
    let file_path = parse_macro_input!(input as LitStr).value();

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let full_path = std::path::PathBuf::from(manifest_dir).join(&file_path);
    let full_path_str = full_path.to_str().unwrap();

    let json_str = std::fs::read_to_string(&full_path).expect("Failed to read JSON file");

    // Parse the JSON.
    let json_value: Value = serde_json::from_str(&json_str).expect("Failed to parse JSON input");

    // Helper: for each leaf, we now record a triple:
    // (param_ty, dims, base_ty)
    // where param_ty is the nested array type (e.g. [f32; 3]),
    // dims is a vector of literal dimensions,
    // and base_ty is the underlying primitive type (e.g. f32).
    fn parse_leaf(leaf: &Value) -> (syn::Type, Vec<syn::LitInt>, syn::Type) {
        // Get the dimensions.
        let dims: Vec<syn::LitInt> = leaf
            .get("shape")
            .and_then(|v| v.as_array())
            .expect("Expected 'shape' to be an array")
            .iter()
            .map(|dim_val| {
                let dim = dim_val
                    .as_u64()
                    .expect("Each shape dimension must be a u64");
                syn::LitInt::new(&dim.to_string(), proc_macro2::Span::call_site())
            })
            .collect();

        let dtype = leaf
            .get("dtype")
            .and_then(|v| v.as_str())
            .expect("Expected 'dtype' to be a string");

        // base type (primitive) used for BufferView
        let base_ty: syn::Type = syn::parse_str(match dtype {
            "float32" => "f32",
            "float64" => "f64",
            "int32" => "i32",
            "int64" => "i64",
            "bool" => "bool",
            _ => "f32",
        })
        .expect("Failed to parse primitive type");

        // param_ty is the nested type (e.g. [f32; 3]) built by wrapping the base type with each array dimension.
        let mut type_str = match dtype {
            "float32" => "f32".to_string(),
            "float64" => "f64".to_string(),
            "int32" => "i32".to_string(),
            "int64" => "i64".to_string(),
            "bool" => "bool".to_string(),
            _ => "f32".to_string(),
        };
        for dim in dims.iter().rev() {
            type_str = format!("[{}; {}]", type_str, dim.to_string());
        }
        let param_ty: syn::Type = syn::parse_str(&type_str)
            .unwrap_or_else(|_| panic!("Failed to parse nested type: {}", type_str));

        (param_ty, dims, base_ty)
    }

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
    let output_buffer_code = output_specs.iter().enumerate().map(|(i, (_param_ty, _dims, base_ty))| {
        let out_ident = format_ident!("out{}", i);
        quote! {
            let #out_ident = {
                let buf_ref = output_list.get_ref(#i).unwrap();
                let buf: eerie::runtime::hal::BufferView<#base_ty> = buf_ref.to_buffer_view(session);
                let mapping = eerie::runtime::hal::BufferMapping::new(buf).unwrap();
                mapping.data().try_into().unwrap()
            };
        }
    });

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
    let fun_ident = format_ident!("{}", fun_name_str);

    let jit_fun_str = format!("jit_{}.main", fun_name_str);

    // Generate the full function.
    let expanded = quote! {
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

        #[allow(dead_code)]
        pub fn #fun_ident(
            instance: &eerie::runtime::api::Instance,
            session: &eerie::runtime::api::Session,
            inputs: ( #( #input_types ),* )
        ) -> ( #( #output_types ),* ) {
            let module_bytes = base64::engine::general_purpose::STANDARD.decode(#compiled_module)
                .expect("Failed to decode the compiled module");
            unsafe {
                session.append_module_from_memory(module_bytes.as_slice())
            }.unwrap();

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
    };

    // Uncomment to print the generated code.
    println!("{}", expanded);

    TokenStream::from(expanded)
}
