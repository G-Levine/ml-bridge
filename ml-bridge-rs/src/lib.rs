use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, LitStr};

#[proc_macro]
pub fn make_fn_from_json(input: TokenStream) -> TokenStream {
    // Get the JSON literal string.
    let json_literal = parse_macro_input!(input as LitStr);
    let json_str = json_literal.value();

    // Parse the JSON.
    let json_value: serde_json::Value =
        serde_json::from_str(&json_str).expect("Failed to parse JSON input");

    // Process the input leaves.
    let input_leaves = json_value
        .get("input_format")
        .and_then(|v| v.get("leaves"))
        .and_then(|v| v.as_array())
        .expect("Could not find 'input_format.leaves' as an array");

    let mut input_types = Vec::new();
    for leaf in input_leaves.iter() {
        // Read the shape dimensions.
        let dims = leaf
            .get("shape")
            .and_then(|v| v.as_array())
            .expect("Expected 'shape' to be an array")
            .iter()
            .map(|v| v.as_u64().expect("Each shape dimension must be a u64"))
            .collect::<Vec<u64>>();

        // Get the dtype string.
        let dtype = leaf
            .get("dtype")
            .and_then(|v| v.as_str())
            .expect("Expected 'dtype' to be a string");

        // Map the dtype to a Rust primitive.
        let rust_primitive = match dtype {
            "float32" => "f32",
            "float64" => "f64",
            "int32" => "i32",
            "int64" => "i64",
            "bool" => "bool",
            _ => "f32", // default fallback
        };

        // Create a type string by wrapping the primitive in an array for each dimension.
        let mut type_str = rust_primitive.to_string();
        for dim in dims.iter().rev() {
            type_str = format!("[{}; {}]", type_str, dim);
        }
        // Parse the string into a type.
        let ty: syn::Type = syn::parse_str(&type_str)
            .unwrap_or_else(|_| panic!("Failed to parse type: {}", type_str));
        input_types.push(ty);
    }

    // Process the output leaves.
    let output_leaves = json_value
        .get("output_format")
        .and_then(|v| v.get("leaves"))
        .and_then(|v| v.as_array())
        .expect("Could not find 'output_format.leaves' as an array");

    let mut output_types = Vec::new();
    for leaf in output_leaves.iter() {
        let dims = leaf
            .get("shape")
            .and_then(|v| v.as_array())
            .expect("Expected 'shape' to be an array")
            .iter()
            .map(|v| v.as_u64().expect("Each shape dimension must be a u64"))
            .collect::<Vec<u64>>();

        let dtype = leaf
            .get("dtype")
            .and_then(|v| v.as_str())
            .expect("Expected 'dtype' to be a string");

        let rust_primitive = match dtype {
            "float32" => "f32",
            "float64" => "f64",
            "int32" => "i32",
            "int64" => "i64",
            "bool" => "bool",
            _ => "f32",
        };

        let mut type_str = rust_primitive.to_string();
        for dim in dims.iter().rev() {
            type_str = format!("[{}; {}]", type_str, dim);
        }
        let ty: syn::Type = syn::parse_str(&type_str)
            .unwrap_or_else(|_| panic!("Failed to parse type: {}", type_str));
        output_types.push(ty);
    }

    // Create tuple types for the inputs (as immutable references) and outputs (as mutable references).
    let input_refs = input_types.iter().map(|ty| quote! { & #ty });
    let output_refs = output_types.iter().map(|ty| quote! { &mut #ty });

    // Build the full tuple types.
    let input_tuple_type = quote! { ( #(#input_refs),* ) };
    let output_tuple_type = quote! { ( #(#output_refs),* ) };

    // Get the function name.
    let name_str = json_value
        .get("name")
        .and_then(|v| v.as_str())
        .expect("Expected 'name' to be a string");

    let name_ident = syn::Ident::new(name_str, proc_macro2::Span::call_site());

    // Generate the function. It now accepts:
    //   - a first parameter "inputs" of type ( &T, &U, … )
    //   - a second parameter "outputs" of type ( &mut U1, &mut U2, … )
    // Note: The implementation is left as unimplemented!() for now.
    let expanded = quote! {
        #[allow(dead_code)]
        pub fn #name_ident(inputs: #input_tuple_type, mut outputs: #output_tuple_type) {
            // This is a placeholder.
            // unimplemented!()
        }
    };

    TokenStream::from(expanded)
}
