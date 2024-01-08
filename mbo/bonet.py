#!/usr/bin/env python3
import runpy
import sys

sys.path.append("bonet")


def main():
    runpy.run_path(path_name="bonet/scripts/train_desbench.py")


if __name__ == "__main__":
    main()
