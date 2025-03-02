# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
Author: Seraphin Zunzer, Fabian Hofmann
Modified: 02.03.25
"""

import tensorflow as tf

from daisy.federated_ids_components import FederatedOnlineClient
from daisy.federated_learning import TFFederatedModel, FederatedIFTM

from daisy.personalized_fl_components.generative.generative_model import GenerativeGAN
from daisy.personalized_fl_components.auto_model_scaler import AutoModelScaler
from daisy.personalized_fl_components.generative.generative_node import (
    pflGenerativeNode,
)
from daisy.personalized_fl_components.distillative.distillative_node import (
    pflDistillativeNode,
)


def load_traditional_fl_conf(
    args,
    t_m,
    err_fn,
    data_handler,
    metrics,
    m_aggr_serv,
    eval_serv,
    aggr_serv,
    input_size,
    label_split,
):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.MeanAbsoluteError()
    id_fn = TFFederatedModel.get_fae(
        input_size=input_size,
        optimizer=optimizer,
        loss=loss,
        batch_size=args.batchSize,
        epochs=1,
    )
    model = FederatedIFTM(identify_fn=id_fn, threshold_m=t_m, error_fn=err_fn)

    # Client
    client = FederatedOnlineClient(
        data_handler=data_handler,
        batch_size=args.batchSize,
        model=model,
        label_split=label_split,
        metrics=metrics,
        m_aggr_server=m_aggr_serv,
        eval_server=eval_serv,
        aggr_server=aggr_serv,
        update_interval_t=args.updateInterval,
        poisoning_mode=args.poisoningMode,
    )
    client.start()
    input("Press Enter to stop client...")
    client.stop()


def load_generative_pfl_conf(
    args,
    t_m,
    err_fn,
    data_handler,
    metrics,
    m_aggr_serv,
    eval_serv,
    aggr_serv,
    input_size,
    label_split,
):
    """
    Demo client using GANs for knowledge transfer between heterogeneous models

    :param args:
    :param t_m:
    :param err_fn:
    :param data_handler:
    :param metrics:
    :param m_aggr_serv:
    :param eval_serv:
    :param aggr_serv:
    :param input_size:
    :param label_split:
    :return:
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.MeanAbsoluteError()
    id_fn = None
    epochs = 1
    ams = AutoModelScaler()

    if args.autoModel:
        id_fn = ams.choose_model(input_size, optimizer, loss, args.batchSize, epochs)
    if not args.autoModel:
        id_fn = ams.get_manual_model(
            args.manualModel, input_size, optimizer, loss, args.batchSize, epochs
        )

    model = FederatedIFTM(identify_fn=id_fn, threshold_m=t_m, error_fn=err_fn)

    generative_gan = GenerativeGAN.create_gan(
        input_size=65,  # Note the difference between GAN input size and detection model input size
        discriminator_optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        generator_optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    )

    # Client
    client = pflGenerativeNode(
        data_handler=data_handler,
        batch_size=args.batchSize,
        model=model,
        label_split=label_split,
        metrics=metrics,
        m_aggr_server=m_aggr_serv,
        eval_server=eval_serv,
        aggr_server=aggr_serv,
        update_interval_t=args.updateInterval,
        generative_model=generative_gan,
        poisoning_mode=args.poisoningMode,
    )
    client.start()
    input("Press Enter to stop client...")
    client.stop()


def load_distillative_pfl_conf(
    args,
    t_m,
    err_fn,
    data_handler,
    metrics,
    m_aggr_serv,
    eval_serv,
    aggr_serv,
    input_size,
    label_split,
):
    """
        Demo client using the knowledge distilation implemention for pFL.

    :param args:
    :param t_m:
    :param err_fn:
    :param data_handler:
    :param metrics:
    :param m_aggr_serv:
    :param eval_serv:
    :param aggr_serv:
    :param input_size:
    :param label_split:
    :return:
    """

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.MeanAbsoluteError()
    epochs = 1
    aMS = AutoModelScaler()

    if args.autoModel:
        print("AUTO MODEL")
        id_fn = aMS.choose_model(input_size, optimizer, loss, args.batchSize, epochs)
    else:
        print("Manual MODEL")
        print(args.manualModel)
        id_fn = aMS.get_manual_model(
            args.manualModel, input_size, optimizer, loss, args.batchSize, epochs
        )

    model = FederatedIFTM(identify_fn=id_fn, threshold_m=t_m, error_fn=err_fn)

    # Client
    client = pflDistillativeNode(
        data_handler=data_handler,
        batch_size=args.batchSize,
        model=model,
        label_split=label_split,
        metrics=metrics,
        m_aggr_server=m_aggr_serv,
        eval_server=eval_serv,
        aggr_server=aggr_serv,
        update_interval_t=args.updateInterval,
        poisoning_mode=args.poisoningMode,
        input_size=65,  # TODO Check if equal input_size
    )
    client.start()
    input("Press Enter to stop client...")
    client.stop()
