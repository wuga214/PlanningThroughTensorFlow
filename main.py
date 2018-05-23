import argparse
import numpy as np
from net.residual import DenselyConnectedNetwork
from utils.argument import check_int_positive, check_float_positive
from utils.io import read_data, dump_net_iohead, dump_norm_info, NetTopology
from viz.nav import virtualizing


def main(args):
    pd_data = read_data(args.path+args.data)
    pd_label = read_data(args.path+args.label)
    data = pd_data.as_matrix()
    label = pd_label.as_matrix()
    indecs = np.random.permutation(data.shape[0])
    data = data[indecs]
    label = label[indecs]
    m_data, n_data = data.shape
    m_label, n_label = label.shape

    train_data = data[:int(m_data*0.9)]
    train_label = label[:int(m_data*0.9)]

    test_data = data[int(m_data*0.9):]
    test_label = label[int(m_data*0.9):]

    # Weighted MSE
    mse_weights = (1.0 / np.square(np.max(train_label, axis=0) + 1)) * 10

    dnn = DenselyConnectedNetwork(n_data, args.neuron, n_label, args.layer, 0.1, mse_weights)
    mean_DNN, std_DNN = dnn.train(train_data, train_label, 200, True)
    dnn.showhistory()

    # Dump the I/O info of the network
    dump_net_iohead(pd_data, pd_label, n_data, n_label, args.head, args.layer+1, args.domain, args.type, args.path)

    # Dump normalizing info
    dump_norm_info(pd_data, mean_DNN, std_DNN, args.domain, args.path)

    topo = NetTopology(dnn.getmodel().layers, mean_DNN, std_DNN, args.type)

    topo.net_transform('D', True, args.path, args.domain, True)

    pred_label = dnn.test(test_data, True, mean_DNN, std_DNN)
    print "Complete testing"
    feed_data = test_data[:, args.split:]
    act_tran = test_data[:, :args.split]
    virtualizing(feed_data, act_tran, test_label, pred_label, 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform Learner")

    parser.add_argument('-p', dest='path',  default="/media/wuga/Data Repository1/JAIR-18/domains/nav/")
    parser.add_argument('-x', dest='data', default='Navigation_Large_Data.txt')
    parser.add_argument('-y', dest='label', default='Navigation_Large_Label.txt')
    parser.add_argument('-n', dest='neuron', type=check_int_positive, default=32)
    parser.add_argument('-l', dest='layer', type=check_int_positive, default=2)
    parser.add_argument('-t', dest='type', default='regular')
    parser.add_argument('-d', dest='domain', default='Navigation')
    parser.add_argument('-s', dest='split', type=check_int_positive, default=2)
    parser.add_argument('--prefix', dest='head', default='D')
    args = parser.parse_args()

    main(args)