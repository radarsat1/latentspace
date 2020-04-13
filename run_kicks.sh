#!/bin/sh

python3 training.py \
        dataset.name=kicks \
        dataset.data_dim=1024 \
        dataset.latent_dim=4 \
        dataset.latent_prior=uniform \
        model.eps_dim=1 \
        model.name=cnn1d \
        model.type=bigan \
        model.variant=0gp \
        model.gp_weight=10 \
        model.normalization.gen=none \
        model.normalization.critic=none \
        model.shape.filters=16 \
        training.learning_rate=1e-3 \
        training.learning_rate_target=1e-3 \
        training.critic_ratio=2 \
        training.epochs=2000 \
        training.batch_size=32
