use quote::quote;
use serde_json::Value;

// Helper: Recursively convert a serde_json::Value into a token stream representing a Rust literal.
pub fn value_to_tokens(v: &Value) -> proc_macro2::TokenStream {
    match v {
        Value::Null => quote! { () },
        Value::Bool(b) => quote! { #b },
        Value::Number(num) => {
            // Always produce a float literal.
            let s = num.to_string();
            // If there's no decimal point, add one.
            let with_decimal = if s.contains('.') {
                s
            } else {
                format!("{}.0", s)
            };
            let lit = syn::LitFloat::new(&with_decimal, proc_macro2::Span::call_site());
            quote! { #lit }
        }
        Value::String(s) => {
            let lit = syn::LitStr::new(s, proc_macro2::Span::call_site());
            quote! { #lit }
        }
        Value::Array(arr) => {
            let elems: Vec<_> = arr.iter().map(value_to_tokens).collect();
            quote! { [ #( #elems ),* ] }
        }
        Value::Object(_) => {
            panic!("Cannot convert object to literal")
        }
    }
}

pub fn value_to_tuple_tokens(v: &Value) -> proc_macro2::TokenStream {
    if let Value::Array(items) = v {
        let elems: Vec<_> = items.iter().map(value_to_tokens).collect();
        quote! { ( #( #elems ),* ) }
    } else {
        value_to_tokens(v)
    }
}

pub fn value_to_ref_tuple_tokens(v: &Value) -> proc_macro2::TokenStream {
    if let Value::Array(items) = v {
        // For each item in the top-level array, get its token stream and prefix it with &
        let elems: Vec<_> = items
            .iter()
            .map(|item| {
                let tokens = value_to_tokens(item);
                quote! { & #tokens }
            })
            .collect();
        quote! { ( #( #elems ),* ) }
    } else {
        value_to_tokens(v)
    }
}

// Helper: for each leaf, we now record a triple:
// (param_ty, dims, base_ty)
// where param_ty is the nested array type (e.g. [f32; 3]),
// dims is a vector of literal dimensions,
// and base_ty is the underlying primitive type (e.g. f32).
pub fn parse_leaf(leaf: &Value) -> (syn::Type, Vec<syn::LitInt>, syn::Type) {
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
