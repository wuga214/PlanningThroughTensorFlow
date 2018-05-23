import argparse


def check_int_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
         raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


def check_float_positive(value):
    ivalue = float(value)
    if ivalue < 0:
         raise argparse.ArgumentTypeError("%s is an invalid positive float value" % value)
    return ivalue