# ml-bridge

ml-bridge is a monorepo that exports JAX computations for use in Rust, leveraging [IREE](https://github.com/iree-org/iree) for ahead-of-time compilation and runtime execution. It is ideal for embedded or real-time deployments, such as running neural network control policies on robots.

## Components
- **ml-bridge-py**: A Python package that transforms JAX computations into compiled MLIR modules with IREE, saving the bytecode, metadata, and sample inputs/outputs to a JSON file.
- **ml-bridge-rs**: A Rust crate featuring a procedural macro (`load_exported_fn!`) that reads these JSON exports to generate Rust functions at compile-time. It also autogenerates unit tests to verify that the outputs when the function is called in Rust are the same as the original JAX outputs.

## Basic Usage

### Python

1. **Install the package:**
   - If you are installing directly from the repository, you can use:
     ```
     pip install git+https://github.com/yourname/ml-bridge.git#subdirectory=ml-bridge/ml-bridge-py
     ```
   - If installing locally:
     ```
     cd ml-bridge/ml-bridge-py
     uv pip install -e .
     ```

2. **Exporting a function:**
   - Use the provided functions in `ml_bridge/export.py` to export a JAX function, compile it via IREE, and save export data to a JSON file.
   - Example:
     ```python
     import jax.numpy as jnp
     import ml_bridge.export as mlexport

     def my_fn(x):
         return x + 1

     sample_input = (jnp.array([1,2,3], dtype=jnp.float32),)
     mlexport.compile_and_save(my_fn, sample_input, "my_fn.json")
     ```

3. **Running Tests:**
   - From the `ml-bridge/ml-bridge-py` directory, run:
     ```
     uv run pytest
     ```

### Rust

1. **Install the Crate:**
   - Add it as a dependency from Git:
     ```toml
     [dependencies]
     ml-bridge-rs = { git = "https://github.com/G-Levine/ml-bridge.git", subdirectory = "ml-bridge-rs" }
     ```
   - Alternatively, build and run the example in the `ml-bridge/ml-bridge-rs` directory:
     ```
     cd ml-bridge/ml-bridge-rs
     cargo run
     ```
2. **Using the Procedural Macro:**
   - The procedural macro `load_exported_fn!` reads a JSON export file and generates a function wrapper that sets up and calls the compiled module. For example, in `main.rs`:
     ```rust
     use ml_bridge_rs::load_exported_fn;

     load_exported_fn!("../ml-bridge-py/test_outputs/my_fn.json");
     ```
   - Once generated, you can invoke the function with input data to execute the compiled module.
3. **Running the Demo Example:**
   - There's an included demo in the `examples/` folder:
     ```
     cargo run --example demo
     ```
   - This demonstrates how to load your compiled modules and run them end-to-end using IREE.
