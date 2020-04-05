#!/bin/sh

python3 training.py \
        dataset.name=kicks \
        dataset.data_dim=64 \
        model.eps_dim=2 \
        model.name=cnn1d \
        model.type=veegan \
        training.batch_size=32
