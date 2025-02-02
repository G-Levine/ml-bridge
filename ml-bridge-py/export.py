import base64
import dataclasses
import json

import iree.compiler as ireec
import iree.runtime as ireert
import jax
import jax.numpy as jnp
import numpy as np
from jax import export, tree_util

# @jax.tree_util.register_dataclass
# @dataclass
# class Config:
#     scale: jnp.ndarray
#     nested: tuple
#     multiplier: jnp.ndarray


# # def test_fn(x, config):
# #     # x: a JAX array (required positional)
# #     # config: an instance of Config dataclass containing several types of values

# #     # Unpack from config which has a nested structure.
# #     scale = config.scale  # a scalar
# #     nested_tuple = config.nested[0]  # a tuple containing two elements
# #     extra = config.nested[1]  # a dict with more nested fields

# #     # Create a result that applies operations combining these types.
# #     result_array = jnp.sin(x) * scale + nested_tuple[0]  # combine array with a number
# #     result_scalar = config.multiplier * extra["bias"]  # a scalar result

# #     # Return a mixed pytree: a tuple with dictionary and list values.
# #     return {"result_array": result_array, "result_info": {"added": nested_tuple[1]}}, [
# #         result_scalar,
# #         scale,
# #     ]


# # positional_input = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
# # keyword_input = Config(
# #     scale=jnp.array(2.0),  # scalar multiplier
# #     nested=(
# #         (jnp.array(5.0), jnp.array(10.0)),  # tuple inside the Config
# #         {
# #             "bias": jnp.array(3.0),
# #             "offset": [jnp.array(0.5), jnp.array(1.5)],
# #         },  # dict inside tuple (and a list inside that dict)
# #     ),
# #     multiplier=jnp.array(4.0),
# # )

# # exported = export.export(jax.jit(test_fn))(positional_input, keyword_input)

positional_input = (
    jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
    jnp.array([4.0, 5.0, 6.0], dtype=jnp.float32),
)


def test_fn(x, y):
    return jnp.sin(x) + y**2


exported = export.export(jax.jit(test_fn))(*positional_input)

compiled_flatbuffer = ireec.compile_str(
    exported.mlir_module_serialized, target_backends=["vmvx"]
)

# Execute the exported function using IREE.
config = ireert.Config("local-task")
ctx = ireert.SystemContext(config=config)
vm_module = ireert.VmModule.copy_buffer(ctx.instance, compiled_flatbuffer)
ctx.add_vm_module(vm_module)

print("INVOKE main")

sample_input = positional_input


def wrapped_jit_test_fn(pytree_input):
    # Flatten the input pytree using JAX tree utilities.
    # (We assume the exported.in_tree structure matches tree_structure(pytree_input).)
    flat_in, _ = tree_util.tree_flatten(pytree_input)

    # (Optional) If you want to ensure that only the kept variables are used,
    # you could filter flat_in with exported.module_kept_var_idx. In this example,
    # all flattened variables are kept (as shown by (0,1,2)):
    kept_inputs = [flat_in[idx] for idx in exported.module_kept_var_idx]

    # Call the underlying IREE module's "main" entry point.
    # Notice that the module expects separate arguments.
    flat_out = ctx.modules.jit_test_fn["main"](*kept_inputs)

    # Ensure that if the result is a single value (or tuple), it is in tuple form.
    # It should have the same length as exported.out_avals.
    if not isinstance(flat_out, (tuple, list)):
        flat_out = (flat_out,)

    # Transfer the output back to the host.
    flat_out = [output.to_host() for output in flat_out]

    # Reassemble the output flat tuple back into the original structured pytree.
    result = tree_util.tree_unflatten(exported.out_tree, flat_out)
    return result


iree_output = wrapped_jit_test_fn(sample_input)
sample_output = test_fn(*sample_input)
print("Python execution result:", sample_output)
print("IREE execution result:", iree_output)


compiled_flatbuffer_base64 = base64.b64encode(compiled_flatbuffer).decode("ascii")


def pytree_format(treedef, avals):
    """
    Function to convert a PyTreeDef to a structured representation.
    Inputs:
    - treedef: PyTreeDef
    - avals: list of ShapedArray
    Outputs:
    - idx_tree: PyTree with the same structure as treedef, but with leaves
        replaced by their index in avals
    - leaf_info: list of dictionaries of form {"shape": shape, "dtype": dtype},
        where shape is a tuple of integers and dtype is a string
    """
    idx_tree = tree_util.tree_unflatten(treedef, list(range(len(avals))))
    leaf_info = [{"shape": aval.shape, "dtype": str(aval.dtype)} for aval in avals]
    return {"tree": idx_tree, "leaves": leaf_info}


input_format = pytree_format(exported.in_tree, exported.in_avals)
output_format = pytree_format(exported.out_tree, exported.out_avals)

export_data = {
    "name": str(exported.fun_name),
    "input_format": input_format,
    "output_format": output_format,
    "sample_input_flat": tree_util.tree_flatten(sample_input)[0],
    "sample_output_flat": tree_util.tree_flatten(sample_output)[0],
    "compiled_module": compiled_flatbuffer_base64,
}


def custom_default(o):
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    if isinstance(o, (np.ndarray, jnp.ndarray)):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


with open("export_data.json", "w") as f:
    json.dump(export_data, f, indent=2, default=custom_default)
