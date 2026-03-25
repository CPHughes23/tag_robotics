"""
This file stores the hyperparameters for the model.
This is used at train time and evaluation
"""

train_cfg_dict = {
    "seed": 42,
    "device": "cuda:0",
    "num_steps_per_env": 24,
    "max_iterations": 3000,
    "save_interval": 100,
    "experiment_name": "rc_car",
    "empirical_normalization": False,
    "obs_groups": {
        "actor": ["policy"],
        "critic": ["policy"],
    },
    "actor": {
        "class_name": "MLPModel",
        "hidden_dims": [128, 64, 32],
        "activation": "elu",
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
        },
    },
    "critic": {
        "class_name": "MLPModel",
        "hidden_dims": [128, 64, 32],
        "activation": "elu",
    },
    "algorithm": {
        "class_name": "PPO",
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.1,
        "entropy_coef": 0.005,
        "num_learning_epochs": 3,
        "num_mini_batches": 4,
        "learning_rate": 3.0e-4,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.02,
        "max_grad_norm": 0.5,
        "rnd_cfg": None,
        "symmetry_cfg": None,
    },
}