from networks.classic_network.dcnn import run_dcnn
from networks.classic_network.drsn import run_drsn
from networks.classic_network.lstm import run_lstm
from networks.classic_network.wdcnn import run_wdcnn
from networks.my_nets.network_fn02 import run_mynet02

if __name__ == '__main__':
    # run_wdcnn.run()
    # run_lstm.run()
    # run_drsn.run()

    # run_dcnn.run()
    run_mynet02.run()


    # import torch
    # out = [[0.0924, 0.0977, 0.0884, 0.0873, 0.1135, 0.1258, 0.1072, 0.1084, 0.0950,
    #   0.0843],
    #  [0.0821, 0.0937, 0.0853, 0.0865, 0.1022, 0.1124, 0.1227, 0.1220, 0.0910,
    #   0.1020]]
    # out = torch.tensor(out)
    # _, pred = torch.max(out, dim=1)
    # print(_, pred)
    #
    # pred = torch.argmax(out, dim=1)
    # print( pred)
    pass
