import torch
import functools
import tensorflow as tf
import tf2onnx
from jax.experimental import jax2tf
import jax.numpy as jnp
import onnx
import jax
import time
import json
import os, sys
import gc

import tools.bab_tools.vnnlib_utils as vnnlib_utils
import tools.bab_tools.bab_runner as ovalbab_runner
from tools.bab_tools.model_utils import one_vs_all_from_model as ovalbab_1vsall_constructor


# Disable OVAL BaB's verbose printing.
class do_not_print:
    # Taken from https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# Note: adapted from
# https://colab.research.google.com/gist/tomhennigan/5e3d316ec6ebd81500a660155770e34a/example-of-jax-haiku-onnx.ipynb
def haiku_to_onnx_file(net_params, net_fn, in_shape, filename, rng_key):

    # Have tensorflow run on cpu
    tf.config.set_visible_devices([], 'GPU')

    # input_size, input_size, input_channel
    x = jnp.ones((1, *in_shape))
    forward = functools.partial(net_fn, net_params, rng_key)

    # NOTE: We need enable_xla=False because tf2onnx is not aware of some XLA specific TF ops (e.g. DotGeneral).
    forward_tf = tf.function(tf.autograph.experimental.do_not_convert(jax2tf.convert(forward, enable_xla=False)))

    input_signature = (tf.TensorSpec(x.shape, x.dtype),)
    model_proto, external_tensor_storage = tf2onnx.convert.from_function(forward_tf, input_signature)

    # Load into ONNX, then dump into file if the file doesn't exist already
    onnx_model = onnx.load_model_from_string(model_proto.SerializeToString())
    onnx.checker.check_model(onnx_model)
    if not os.path.isfile(filename):
        onnx.save(onnx_model, filename)
    else:
        print("Onnx file already exists: the PyTorch model will be created from the existing file.")
    return onnx_model


def get_pytorch_network(net_params, net_fn, in_shape, filename, rng=None):

    # Create onnx model from jax and save it into file if it doesn't already exist
    key1, key2 = jax.random.split(rng if rng is not None else jax.random.PRNGKey(0))
    onnx_model = haiku_to_onnx_file(net_params, net_fn, in_shape, filename, key1)

    # Assert jax and onnx comply
    torch_in = torch.rand((1, *in_shape))
    jax_in = jnp.array(torch_in.numpy())
    onnx_out = list(vnnlib_utils.predict_with_onnxruntime(onnx_model, torch_in.numpy()).values())[0]
    jax_out = net_fn(net_params, key2, jax_in)
    assert jnp.abs(onnx_out - jax_out).max() < 1e-4

    # Convert the ONNX into PyTorch
    assert filename.endswith('.onnx')
    model, in_shape, out_shape, dtype, model_correctness = vnnlib_utils.onnx_to_pytorch(filename)
    assert model_correctness

    # Assert that the model specification is currently supported.
    supported = vnnlib_utils.is_supported_model(model)
    assert supported

    layers = list(model.children())
    for clayer in layers:
        if isinstance(clayer, torch.nn.Linear):
            clayer.bias.data = clayer.bias.data.squeeze(0)

    # Assert jax and torch comply
    torch_out = torch.nn.Sequential(*layers)(torch_in).detach().numpy()
    assert jnp.abs((torch_out - jax_out).max() < 1e-4)
    return layers


def create_1_vs_all_verification_problem(model, y, input_bounds, max_solver_batch):
    with do_not_print():
        verif_layers = ovalbab_1vsall_constructor(
            torch.nn.Sequential(*model), y, domain=input_bounds, max_solver_batch=max_solver_batch, use_ib=True)
    return verif_layers


def run_oval_bab(verif_layers, input_bounds, ovalbab_json_config, timeout=20, results_table=None, test_idx=None,
                 json_name=None, record_name=None):
    # Run OVAL-BaB with the configuration specified in ovalbab_json_config
    return_dict = dict()
    start_time = time.time()

    with open(ovalbab_json_config) as json_file:
        json_params = json.load(json_file)
    # with do_not_print():
    ovalbab_runner.bab_from_json(
        json_params, verif_layers, input_bounds, return_dict, None, instance_timeout=timeout, start_time=start_time)
    del json_params

    bab_out, bab_nb_states = ovalbab_runner.bab_output_from_return_dict(return_dict)
    bab_time = time.time() - start_time

    # Store BaB results in a table for later analysis.
    if results_table is not None:
        results_table.loc[test_idx]["prop"] = test_idx
        results_table.loc[test_idx][f"BSAT_{json_name}"] = bab_out
        results_table.loc[test_idx][f"BBran_{json_name}"] = bab_nb_states
        results_table.loc[test_idx][f"BTime_{json_name}"] = bab_time
        results_table.to_pickle(record_name)

    torch.cuda.empty_cache()
    gc.collect()

    return bab_out == "False", bab_out != "True"


