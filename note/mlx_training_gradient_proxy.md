# MLX training gradient proxy note

## Context

PyTorch and MLX expose gradients differently.

In PyTorch, intermediate tensors can use `retain_grad()` so later training logic
can read gradients from non-leaf values such as projected 2D means.

In MLX, gradients are produced through:

```python
mx.value_and_grad(fn, argnums=...)
```

Only the function arguments selected by `argnums` receive gradients. MLX does not
provide the same `retain_grad()` workflow for arbitrary intermediate tensors.

## FastGS MLX pattern

FastGS PyTorch uses `viewspace_points` as a dummy trainable parameter to carry
the gradient information needed by after-training logic.

The same pattern is required in MLX when a training strategy needs gradients for
intermediate values such as `means2d` from the `means3d` projection path.

Typical uses:

```text
screen-space means2d gradient
absolute screen-space gradient
statistics used by clone / densify / split / pruning
after-training process state updates
```

## gsplat_core migration rule

Do not depend on PyTorch-style intermediate `retain_grad()` semantics.

If gsplat training code needs gradient information from an intermediate forward
value, expose an explicit dummy trainable parameter, such as:

```text
viewspace_points
means2d_grad_proxy
```

The exact name should follow the surrounding API. When the behavior mirrors
FastGS, prefer `viewspace_points`.

## MLX argnums requirement

The dummy gradient proxy must be a visible argument to the loss function, and it
must be included in the `argnums` passed to `mx.value_and_grad(...)`.

Example shape:

```python
def loss_fn(params, viewspace_points, batch):
    outputs = gsplat_forward(
        params=params,
        viewspace_points=viewspace_points,
        batch=batch,
    )
    return outputs["loss"]

loss, grads = mx.value_and_grad(loss_fn, argnums=(0, 1))(
    params,
    viewspace_points,
    batch,
)
```

If parameters are passed as a single dictionary, the proxy must be included in
the differentiated argument structure, or it must be a separate argument listed
in `argnums`.

## API implications

Forward-only low-level ops do not need the proxy unless they are part of a
training/backward path.

When implementing training-aware wrappers or backward-capable primitives:

```text
1. Check whether gsplat PyTorch reads means2d.grad, absgrad, or related stats.
2. Add an explicit proxy input when intermediate gradients are required.
3. Ensure the Python loss function exposes the proxy as an argnums-selected argument.
4. Return or store after-training statistics from the proxy gradient, not from an
   assumed retained intermediate tensor.
```

## Related gsplat areas to inspect

```text
gsplat/rendering.py
gsplat/strategy/default.py
gsplat/strategy/mcmc.py
gsplat/strategy/ops.py
gsplat/cuda/_wrapper.py::rasterize_to_pixels(absgrad=...)
```

These files may use screen-space gradient or absolute gradient signals for
density control. When porting that behavior, follow the dummy trainable proxy
rule above.

