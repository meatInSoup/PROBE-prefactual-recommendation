import argparse

from expt.config import Config


class Expt1(Config):
    __dictpath__ = 'ec.e1'

    all_clfs = ['net0']
    all_datasets = ['synthesis']

    # face_params = {
    #     "mode": "knn",
    #     "fraction": 1.0,
    # }
    
    face_params = {
        "mode": "knn",
        "fraction": 1.0,
        "n_neighbors": 10,
        "weights": 1.0,
    }

    dice_params = {
        "proximity_weight": 0.5,
        "diversity_weight": 1.0,
        "k": 1,
    }

    reup_params = {
        "T": 10,
        "eps": 1e-3,
        "lr": 0.01,
        "lmbda": 1.0,
        "rank": True,
        "knn": True,
        "n": 10,
    }
    
    probe_params = {
        "sessions": 10,
        "iterations": 500,
        "lr": 0.01,
        "n_neighbors": 10,
        "lr_gd": 0.1,
        "rank": True
    }
    

    wachter_params = {
        "lr": 0.01,
        "lmbda": 1.0,
    }

    k = 1

    num_samples = 1000
    max_ins = 20
    num_A = 5
    max_distance = 1.0
    n_neighbors = 0.4
    graph_pre = True


class Expt2(Config):
    __dictpath__ = 'ec.e2'

    all_clfs = ['net0']
    all_datasets = ['synthesis']

    # face_params = {
    #     "mode": "knn",
    #     "fraction": 1.0,
    # }
    
    face_params = {
        "mode": "knn",
        "fraction": 1.0,
        "n_neighbors": 50,
        "weights": 1.0,
    }

    dice_params = {
        "proximity_weight": 0.5,
        "diversity_weight": 1.0,
        "k": 1,
    }

    reup_params = {
        "T": 10,
        "eps": 1e-3,
        "lr": 0.01,
        "lmbda": 1.0,
        "rank": False,
        "knn": True,
        "n": 50,
    }
    
    probe_params = {
        "sessions": 10,
        "iterations": 500,
        "lr": 0.1,
        "n_neighbors": 50,
        "lr_gd": 0.1,
        "rank": False
    }
    
    wachter_params = {
        "lr": 0.01,
        "lmbda": 1.0,
    }

    k = 4

    num_samples = 1000
    max_ins = 20
    num_w = 5
    max_distance = 10.0
    n_neighbors = 0.5
    graph_pre = True


class ExptConfig(Config):
    __dictpath__ = 'ec'

    e1 = Expt1()
    e2 = Expt2()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration')
    parser.add_argument('--dump', default='config.yml', type=str)
    parser.add_argument('--load', default=None, type=str)
    parser.add_argument('--mode', default='merge_cls', type=str)

    args = parser.parse_args()
    if args.load is not None:
        ExptConfig.from_file(args.load)
    ExptConfig.to_file(args.dump, mode=args.mode)
