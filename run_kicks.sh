#!/bin/sh

python3 training.py \
        dataset.name=kicks \
        dataset.data_dim=1024 \
        dataset.latent_dim=4 \
        dataset.latent_prior=uniform \
        model.eps_dim=1 \
        model.name=cnn1d \
        model.type=veegan \
        model.variant=0gp \
        model.normalization.gen=batch \
        training.learning_rate=1e-3 \
        training.learning_rate_target=1e-4 \
        training.critic_ratio=2 \
        training.batch_size=32
