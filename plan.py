import argparse
import numpy as np

from net.cell import TrainedCell
from net.optimization import ActionOptimizer
from hard.domains import HVAC, NAVI, RESERVOIR
from hard.specification import hvac_settings, reservoir_settings, navi_settings
from hard.instance import hvac6_instance
from utils.argument import check_int_positive, check_float_positive
from utils.io import load_pickle, load_csv

domains = {
    "Navigation": NAVI,
    "Reservoir": RESERVOIR,
    "HVAC": HVAC
}

settings = {
    "Navigation": NAVI,
    "Reservoir": reservoir_settings,
    "HVAC": navi_settings
}


def main(args):

    hvac = HVAC(args.batch, hvac6_instance, hvac_settings)

    pretrained_weights = load_pickle(args.weight, 'weight')
    normalization = load_csv(args.weight, 'normalization')
    normalization[1][normalization[1] == 0] = 1

    optimizer = ActionOptimizer(num_step=args.horizon,
                                num_act=args.action,
                                batch_size=args.batch,
                                domain_settings=hvac,
                                num_state_units=args.state,
                                num_reward_units=7,
                                num_hidden_units=args.neuron,
                                num_hidden_layers=args.layer,
                                dropout=0.1,
                                pretrained=pretrained_weights,
                                normalize=normalization,
                                learning_rate=0.01,
                                action_mean=5,
                                )

    optimizer.Optimize([0, 10], epoch=300)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tensorflow Planner")

    parser.add_argument('-w', dest='weight', default="weights/hvac/hvac6")
    parser.add_argument('-n', dest='neuron', type=check_int_positive, default=32)
    parser.add_argument('-l', dest='layer', type=check_int_positive, default=2)
    parser.add_argument('-d', dest='domain', default='HVAC')
    parser.add_argument('-b', dest='batch', type=check_int_positive, default=10)
    parser.add_argument('-hz', dest='horizon', type=check_int_positive,  default=20)
    parser.add_argument('-a', dest='action', type=check_int_positive,  default=6)
    parser.add_argument('-s', dest='state', type=check_int_positive, default=6)
    parser.add_argument('--prefix', dest='head', default='D')
    args = parser.parse_args()

    main(args)