import argparse
import numpy as np

from net.cell import TrainedCell
from net.optimization import ActionOptimizer
from hard.domains import HVAC
from hard.specification import hvac_settings, hvac3_instance
from utils.argument import check_int_positive, check_float_positive


def main(args):

    hvac = HVAC(args.batch, hvac3_instance, hvac_settings)

    optimizer = ActionOptimizer(num_step=args.horizon,
                                num_act=args.action,
                                batch_size=args.batch,
                                domain_settings=hvac,
                                num_state_units=args.state,
                                num_reward_units=7,
                                num_hidden_units=args.neuron,
                                num_hidden_layers=args.layer,
                                dropout=0.1
                                )





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tensorflow Planner")

    parser.add_argument('-w', dest='weight', default="/media/wuga/Data Repository1/JAIR-18/weights/hvac/hvac3/")
    parser.add_argument('-n', dest='neuron', type=check_int_positive, default=32)
    parser.add_argument('-l', dest='layer', type=check_int_positive, default=2)
    parser.add_argument('-d', dest='domain', default='HVAC')
    parser.add_argument('-b', dest='batch', type=check_int_positive, default=100)
    parser.add_argument('-hz', dest='horizon', type=check_int_positive,  default=20)
    parser.add_argument('-a', dest='action', type=check_int_positive,  default=6)
    parser.add_argument('-s', dest='state', type=check_int_positive, default=6)
    parser.add_argument('--prefix', dest='head', default='D')
    args = parser.parse_args()

    main(args)