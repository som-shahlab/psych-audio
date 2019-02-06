"""Docstring"""
import argparse


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-hi')
    args = parser.parse_args()
    main(args)
