import jax
import jax.numpy as jnp
import ml_bridge.export as mlexport
import numpy as np


def test_simple_function():
    # Define a simple function to test.
    def my_fn(x, W, b):
        return jax.nn.relu(jnp.dot(x, W) + b)

    # Create sample inputs.
    x = jnp.ones((128,), dtype=jnp.float32)
    W = jnp.ones((128, 128), dtype=jnp.float32)
    b = jnp.ones((128,), dtype=jnp.float32)

    positional_input = (x, W, b)

    # Obtain the exported function and compiled module.
    exported, compiled = mlexport.export_and_compile(my_fn, positional_input)

    # Run the function normally.
    expected = my_fn(*positional_input)
    # Run via IREE.
    iree_output = mlexport.run_with_iree(exported, compiled, positional_input)
    # Depending on your test framework you can use assertions:
    np.testing.assert_allclose(iree_output, expected)

    mlexport.compile_and_save(my_fn, positional_input, "test_outputs/my_fn.json")


def test_another_function():
    # Test with another function
    def another_fn(a, b):
        return a + b

    a = jnp.ones((32,), dtype=jnp.float32)
    b = jnp.full((32,), 2.0, dtype=jnp.float32)
    positional_input = (a, b)

    exported, compiled = mlexport.export_and_compile(another_fn, positional_input)
    expected = another_fn(*positional_input)
    iree_output = mlexport.run_with_iree(exported, compiled, positional_input)
    np.testing.assert_allclose(iree_output, expected)

    mlexport.compile_and_save(
        another_fn, positional_input, "test_outputs/another_fn.json"
    )


def test_function_with_external_variable():
    # Define an external array variable outside the scope of the function being compiled.
    external_array = jnp.full((128,), 3.0, dtype=jnp.float32)

    # Define a function that uses the external variable.
    def fn_with_external(x):
        return x + external_array

    # Create sample input.
    x = jnp.ones((128,), dtype=jnp.float32)
    positional_input = (x,)

    # Obtain the exported function and compiled module.
    exported, compiled = mlexport.export_and_compile(fn_with_external, positional_input)

    # Run the function normally.
    expected = fn_with_external(x)
    # Run via IREE.
    iree_output = mlexport.run_with_iree(exported, compiled, positional_input)
    np.testing.assert_allclose(iree_output, expected)

    mlexport.compile_and_save(
        fn_with_external, positional_input, "test_outputs/fn_with_external.json"
    )


def test_function_with_scalars():
    # Define a function that uses scalar inputs.
    def fn_with_scalars(a, b):
        return a + b

    a = jnp.array(3.0, dtype=jnp.float32)
    b = jnp.array(4.0, dtype=jnp.float32)
    positional_input = (a, b)

    exported, compiled = mlexport.export_and_compile(fn_with_scalars, positional_input)
    expected = fn_with_scalars(*positional_input)
    iree_output = mlexport.run_with_iree(exported, compiled, positional_input)
    np.testing.assert_allclose(iree_output, expected)

    mlexport.compile_and_save(
        fn_with_scalars, positional_input, "test_outputs/fn_with_scalars.json"
    )
