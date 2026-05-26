#!/usr/bin/env python3

from gsplat_core import dummy_add


def main() -> None:
    result = dummy_add(20, 22)
    print(f"dummy_add(20, 22) = {result}")
    assert result == 42


if __name__ == "__main__":
    main()
