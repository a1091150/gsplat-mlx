#!/usr/bin/env python3

import mlx.core as mx
from gsplat_core import dummy_add, dummy_array_add


def main() -> None:
    result = dummy_add(20, 22)
    a = mx.array([1, 2, 3], dtype=mx.float32)
    b = mx.array([4, 5, 6], dtype=mx.float32)

    array_result = dummy_array_add(
        {
            "a": a,
            "b": b,
        }
    )
    mx.eval(array_result)

    print(f"dummy_add(20, 22) = {result}")
    print(f"dummy_array_add([1, 2, 3], [4, 5, 6]) = {array_result}")
    assert result == 42
    assert array_result.tolist() == [5.0, 7.0, 9.0]


if __name__ == "__main__":
    main()
