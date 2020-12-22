# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Activation functions.
"""

# pylint: disable=unused-import
# re-export activation functions from jax.nn
from jax.nn import celu
from jax.nn import elu
from jax.nn import gelu
from jax.nn import glu
from jax.nn import leaky_relu
from jax.nn import log_sigmoid
from jax.nn import log_softmax
from jax.nn import normalize
from jax.nn import relu
from jax.nn import sigmoid
from jax.nn import soft_sign
from jax.nn import softmax
from jax.nn import softplus
from jax.nn import swish
from jax.nn import silu
from jax.nn import selu
from jax.nn import hard_tanh
from jax.nn import relu6
from jax.nn import hard_sigmoid
from jax.nn import hard_swish

from jax.numpy import tanh
import jax.numpy as jnp
from jax import vmap, pmap, custom_jvp
from jax import random
import jax
import numpy as np
import jax.lax as lax


@jax.partial(jax.jit, static_argnums=(1))
def _make_ix_like_jax(input, dim=-1):
    d = input.shape[dim]
    rho = jnp.arange(1, d + 1, dtype=input.dtype)
    new_shape = [rho.shape[0]] + [1] * (len(input.shape) - 1)
    return rho.reshape(new_shape).swapaxes(0, dim)


@jax.partial(jax.jit, static_argnums=(1))
def _threshold_and_support_jax(input, dim=-1):
    Xsrt = jnp.flip(jnp.sort(input, axis=dim), axis=dim)

    rho = _make_ix_like_jax(input, dim)
    mean = Xsrt.cumsum(axis=dim) / rho
    mean_sq = (Xsrt ** 2).cumsum(axis=dim) / rho
    ss = rho * (mean_sq - mean ** 2)
    delta = (1 - ss) / rho

    delta_nz = jnp.clip(delta, a_min=0)
    tau = mean - jnp.sqrt(delta_nz)
    support_size = jnp.expand_dims((tau <= Xsrt).sum(axis=dim), axis=dim)
    tau_star = jnp.take_along_axis(tau, indices=(support_size - 1), axis=dim)
    return tau_star, support_size


@jax.partial(custom_jvp, nondiff_argnums=(1,))
def entmax15(input, dim=-1):
    max_val = input.max(axis=dim, keepdims=True)
    input_n = input - max_val
    input_n = input_n / 2
    tau_star, _ = _threshold_and_support_jax(input_n, dim)
    output = jnp.clip((input_n - tau_star), a_min=0) ** 2
    return output


@entmax15.defjvp
def entmax15_jvp(dim, primals, tangents):
    input = primals[0]
    Y = entmax15(input, dim)
    gppr = jnp.sqrt(Y)
    grad_output = tangents[0]
    dX = grad_output * gppr
    q = dX.sum(axis=dim) / gppr.sum(axis=dim)
    q = jnp.expand_dims(q, axis=dim)
    dX -= q * gppr
    return Y, dX


def _absolute_dims(rank, dims):
    return tuple([rank + dim if dim < 0 else dim for dim in dims])


@jax.partial(custom_jvp, nondiff_argnums=(1, 2, 3,))
def entmax(input, axis=-1, alpha=1.5, T=10):
    axis = axis if isinstance(axis, tuple) else (axis,)
    axis = _absolute_dims(input.ndim, axis)
    axis = axis[0]
    original_shape = input.shape
    reduce_length = input.shape[axis]
    input = jnp.swapaxes(input, -1, axis)
    former_dim = input.shape
    input = input.reshape(input.size / reduce_length, reduce_length)

    def map_row(z_input, alpha, T):
        z = (alpha - 1) * z_input

        def p_tau(z, tau, alpha=1.5):
            return jnp.clip((alpha - 1) * z - tau, a_min=0) ** (1 / (alpha - 1))

        def get_tau(tau, tau_max, tau_min, z_value):
            return lax.cond(z_value < 1,
                            lambda _: (tau, tau_min),
                            lambda _: (tau_max, tau),
                            operand=None
                            )

        @jax.jit
        def body(kwargs, x):
            tau_min = kwargs['tau_min']
            tau_max = kwargs['tau_max']
            z = kwargs['z']
            alpha = kwargs['alpha']

            tau = (tau_min + tau_max) / 2
            z_value = p_tau(z, tau, alpha).sum()
            taus = get_tau(tau, tau_max, tau_min, z_value)
            tau_max, tau_min = taus[0], taus[1]
            return {'tau_min': tau_min, 'tau_max': tau_max, 'z': z, 'alpha': alpha}, None

        tau_min, tau_max = jnp.min(z) - 1, jnp.max(z) - z.shape[0] ** (1 - alpha)
        # result = lax.fori_loop(0, T, body, {'tau_min':tau_min, 'tau_max':tau_max, 'z':z, 'alpha':alpha})
        result, _ = lax.scan(body, {'tau_min': tau_min, 'tau_max': tau_max, 'z': z, 'alpha': alpha}, xs=None, length=T)
        tau = (result['tau_max'] + result['tau_min']) / 2
        result = p_tau(z, tau, alpha)
        return result / result.sum()

    result = vmap(jax.partial(map_row, alpha=alpha, T=T), 0)(input)
    return jnp.swapaxes(result.reshape(former_dim), -1, axis)


@entmax.defjvp
def entmax_jvp(axis, alpha, T, primals, tangents):
    input = primals[0]
    Y = entmax(input, axis, alpha, T)
    gppr = Y ** (2 - alpha)
    grad_output = tangents[0]
    dX = grad_output * gppr
    q = dX.sum(axis=axis) / gppr.sum(axis=axis)
    q = jnp.expand_dims(q, axis=axis)
    dX -= q * gppr
    return Y, dX
    # pylint: enable=unused-import
