import base64
import dataclasses
import json

import iree.compiler as ireec
import iree.runtime as ireert
import jax
import jax.numpy as jnp
import numpy as np
from jax import export, tree_util


def export_and_compile(fn, positional_input, target_backends=["llvm-cpu"]):
    # Create the jitted function and export it.
    exported = export.export(jax.jit(fn))(*positional_input)
    # compile the MLIR module using IREE
    compiled_flatbuffer = ireec.compile_str(
        exported.mlir_module_serialized, target_backends=target_backends
    )
    return exported, compiled_flatbuffer


def create_export_data(exported, compiled_flatbuffer, sample_input, sample_output):
    compiled_flatbuffer_base64 = base64.b64encode(compiled_flatbuffer).decode("ascii")

    def pytree_format(treedef, avals):
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
    return export_data


def dump_export_data(export_data, filename="export_data.json"):
    def custom_default(o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, (np.ndarray, jnp.ndarray)):
            return o.tolist()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(filename, "w") as f:
        json.dump(export_data, f, indent=2, default=custom_default)


def compile_and_save(
    fn, sample_input, filename="export_data.json", target_backends=["llvm-cpu"]
):
    exported, compiled_flatbuffer = export_and_compile(fn, sample_input, target_backends)
    sample_output = fn(*sample_input)
    export_data = create_export_data(
        exported, compiled_flatbuffer, sample_input, sample_output
    )
    dump_export_data(export_data, filename)


def run_with_iree(exported, compiled_flatbuffer, sample_input):
    config = ireert.Config("local-task")
    ctx = ireert.SystemContext(config=config)
    vm_module = ireert.VmModule.copy_buffer(ctx.instance, compiled_flatbuffer)
    ctx.add_vm_module(vm_module)

    # Define the wrapper that matches the exported function interface.
    def wrapped_jit_test_fn(pytree_input):
        flat_in, _ = tree_util.tree_flatten(pytree_input)
        kept_inputs = [flat_in[idx] for idx in exported.module_kept_var_idx]

        flat_out = ctx.modules[f"jit_{exported.fun_name}"]["main"](*kept_inputs)

        if not isinstance(flat_out, (tuple, list)):
            flat_out = (flat_out,)
        flat_out = [output.to_host() for output in flat_out]
        result = tree_util.tree_unflatten(exported.out_tree, flat_out)
        return result

    # Run the wrapped function.
    return wrapped_jit_test_fn(sample_input)
