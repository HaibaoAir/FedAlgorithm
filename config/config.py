import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # Training config
    parser.add_argument("--alg_name", type=str, default="fedavg", help="algorithm name")
    parser.add_argument("--dataset_name", type=str, default="svhn", help="dataset name")
    parser.add_argument("--num_class", type=int, default=10, help="class num")
    parser.add_argument("--init_num_class", type=int, default=7, help="init class num")
    parser.add_argument("--dirichlet", type=float, default=0.7, help="non-iid rate")

    parser.add_argument("--net_name", type=str, default="cnn", help="net name")
    parser.add_argument("--num_epoch", type=int, default=10, help="client epoch")
    parser.add_argument("--batch_size", type=int, default=64, help="client batch_size")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="lr")
    parser.add_argument("--eval_freq", type=int, default=5, help="eval frequency")
    parser.add_argument(
        "--path_prefix",
        type=str,
        default="../logs/fedstream/",
        help="path prefix for saving logs",
    )
    parser.add_argument("--device", type=str, default="cuda", help="gpu id")

    # Pretraining config
    parser.add_argument("--num_client", type=int, default=5, help="client num")
    parser.add_argument("--num_sample", type=int, default=5, help="client num")
    parser.add_argument("--num_round", type=int, default=100, help="round num")
    parser.add_argument("--init_data_lb", type=int, default=700, help="init lower")
    parser.add_argument("--init_data_ub", type=int, default=800, help="init upper")

    parser.add_argument("--delta", type=float, default=1, help="delta")
    parser.add_argument("--psi", type=float, default=1, help="psi")
    parser.add_argument("--alpha", type=float, default=1.0e-3, help="alpha")
    parser.add_argument("--beta", type=float, default=1.0e-7, help="beta")
    parser.add_argument("--sigma", type=float, default=0.75, help="sigma")
    parser.add_argument("--kappa_1", type=float, default=1, help="kappa_1")
    parser.add_argument("--kappa_2", type=float, default=1, help="kappa_2")
    parser.add_argument("--kappa_3", type=float, default=1.0e-2, help="kappa_3")
    parser.add_argument("--kappa_4", type=float, default=1.0e-2, help="kappa_4")
    parser.add_argument("--gamma", type=float, default=1.0e-4, help="gamma")
    parser.add_argument("--new_gamma", type=float, default=5.0e-4, help="new_gamma")

    parser.add_argument("--fix_eps_1", type=float, default=1.0e-2, help="fix_eps_1")
    parser.add_argument(
        "--fix_eps_2", type=float, default=1, help="fix_eps_2"
    )  # round=5(3), round=50(20), round=100(20)
    parser.add_argument("--fix_eps_3", type=float, default=1, help="fix_eps_3")
    parser.add_argument(
        "--fix_max_iter", type=int, default=10000000, help="fix_max_iter"
    )

    parser.add_argument(
        "--reward_lb", type=float, default=55, help="reward lower bound"
    )
    parser.add_argument(
        "--reward_ub", type=float, default=65, help="reward upper bound"
    )
    parser.add_argument("--theta_lb", type=float, default=0.5, help="theta lower bound")
    parser.add_argument(
        "--theta_ub", type=float, default=0.65, help="theta upper bound"
    )
    parser.add_argument("--n_calls", type=int, default=100, help="n_calls")
    parser.add_argument(
        "--base_estimator", type=str, default="GP", help="base_estimator"
    )
    parser.add_argument("--noise", type=str, default="gaussian", help="noise")
    parser.add_argument("--acq_func", type=str, default="EI", help="acq_func")
    parser.add_argument("--xi", type=float, default=0.005, help="xi")
    parser.add_argument("--random_state", type=int, default=42, help="random_state")

    # Others
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx")
    parser.add_argument("--temperature", type=float, default=0.5, help="MOON")
    parser.add_argument("--moon_mu", type=float, default=1, help="MOON")
    parser.add_argument("--feddyn_alpha", type=float, default=0.01, help="feddyn alpha")

    args = parser.parse_args()
    return args
