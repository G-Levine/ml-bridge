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
     pip install git+https://github.com/G-Levine/ml-bridge.git#subdirectory=ml-bridge-py
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

1. **Installing and Running the Crate:**

- Add the crate as a dependency directly from Git:

  ```toml
  [dependencies]
  ml-bridge-rs = { git = "https://github.com/G-Levine/ml-bridge.git", subdirectory = "ml-bridge-rs" }
  ```

2. **Using the load_exported_fn! Macro:**

- The macro accepts two arguments: an identifier to be used as the module name and a string literal for the JSON export file path. The generated module will provide two public functions:

  - `register(instance, session)`: Registers the compiled module with the IREE session.
  - `call(instance, session, inputs)`: Invokes the exported function with the provided inputs and returns the outputs.

- **Example Usage:**
  In your Rust code, add an invocation of the macro as follows:

  ```rust
  use ml_bridge_rs::load_exported_fn;
  use eerie::runtime::api;
  use eerie::runtime::hal;

  // Generate a module named `my_exported_fn` using the JSON file at the given path.
  load_exported_fn!(my_exported_fn, "../ml-bridge-py/test_outputs/my_fn.json");

  fn main() -> Result<(), Box<dyn std::error::Error>> {
      // Obtain an IREE instance and session using the bindings provided by the EERIE crate.
      let instance = api::Instance::new(
          &api::InstanceOptions::new(&mut hal::DriverRegistry::new()).use_all_available_drivers(),
      )?;
      let device = instance.try_create_default_device("local-task")?;
      let session = api::Session::create_with_device(&instance, &api::SessionOptions::default(), &device)?;

      // Register the compiled module (this is a one-time operation).
      my_exported_fn::register(&instance, &session);

      // Build input tuple with correct types, e.g.:
      let inp1 = [1f32; 128];
      let inp2 = [[1f32; 128]; 128];
      let inp3 = [1f32; 128];

      let output = my_exported_fn::call(&instance, &session, (&inp1, &inp2, &inp3));
      println!("Output: {:?}", output);
      Ok(())
  }
  ```

- This approach makes it explicit: you choose the module name for the generated code, and you can directly refer to it via `my_exported_fn::register` and `my_exported_fn::call` in your code.

3. **Running the Demo Example:**

- There's an included demo in the `examples/` folder which shows how to load your compiled modules and run them end-to-end using IREE.
  ```
  cargo run --example demo     # to benchmark the execution speed of a simple neural network
  cargo test --example demo    # to run the autogenerated unit tests
  ```
