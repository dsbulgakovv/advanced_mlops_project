import json
import os
import time

import hydra
import numpy as np
import torch
import torch.nn as nn
from metrics import calculate_metrics, get_score_distributions
from nn_model import FullyConnectedNeuralNetwork
from utils import batch_generator, load_data, make_transformer_image


def compute_loss(model, data_batch, loss_function=nn.CrossEntropyLoss()):
    """Compute the loss using loss_function (nn.CrossEntropyLoss by default)
    for the batch of data and return mean loss value for this batch."""
    # load the data
    img_batch = data_batch["img"]
    label_batch = data_batch["label"]
    # forward pass
    logits = model(img_batch)
    # loss computation
    loss = loss_function(logits, label_batch)

    return loss, model


def train_model(
    device,
    model,
    batch_size,
    n_train,
    train_batch_generator,
    val_batch_generator,
    opt,
    ckpt_name=None,
    n_epochs=10,
):
    """
    Run training: forward/backward pass using train_batch_generator and
    evaluation using val_batch_generator.
    Log performance using loss monitoring and score distribution
    plots for validation set.
    """
    metrics_dict = dict()
    train_loss, val_loss = [], [1]
    val_loss_idx = [0]
    top_val_accuracy = 0

    for epoch in range(n_epochs):
        start_time = time.time()

        # Train phase
        model.train(True)  # enable dropout / batch_norm training behavior
        for (x_batch, y_batch) in train_batch_generator:
            # move data to target device
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            data_batch = {"img": x_batch, "label": y_batch}

            # train on batch: compute loss, calc grads, perform optimizer step and zero the grads
            loss, model = compute_loss(model, data_batch)

            # compute backward pass
            opt.zero_grad()
            loss.backward()
            opt.step()

            # log train loss
            train_loss.append(loss.detach().cpu().numpy())

        # Evaluation phase
        metric_results = test_model(model, val_batch_generator, subset_name="val")
        metric_results = get_score_distributions(metric_results)

        # Logging
        val_loss_value = np.mean(metric_results["loss"])
        val_loss_idx.append(len(train_loss))
        val_loss.append(val_loss_value)

        # print("Epoch {} of {} took {:.3f}s".format(epoch + 1, n_epochs, time.time() - start_time))
        train_loss_value = np.mean(train_loss[(-n_train // batch_size) :])  # noqa: E203
        val_accuracy_value = metric_results["accuracy"]

        metrics_dict["epoch_" + str(epoch)] = {
            "start_time": start_time,
            "metric_results": metric_results,
            "val_loss_value": val_loss_value,
            "train_loss_value": train_loss_value,
        }

        if val_accuracy_value > top_val_accuracy and ckpt_name is not None:
            top_val_accuracy = val_accuracy_value
            # save checkpoint of the best model to disk
            with open(ckpt_name, "wb") as f:
                torch.save(model, f)

    return metrics_dict


@torch.no_grad()
def test_model(
    device, loss_function, model, batch_generator_f, subset_name="test", print_log=True
) -> dict:
    """Evaluate the model using data from batch_generator and metrics defined above."""

    # disable dropout / use averages for batch_norm
    model.train(False)

    # save scores, labels and loss values for performance logging
    score_list = []
    label_list = []
    loss_list = []

    for X_batch, y_batch in batch_generator_f:
        # do the forward pass
        logits = model(X_batch.to(device))
        _, scores = torch.max(logits.cpu().data, 1)
        labels = y_batch.numpy().tolist()

        # compute loss value
        loss = loss_function(logits, y_batch.to(device))

        # save the necessary data
        loss_list.append(loss.detach().cpu().numpy().tolist())
        score_list.extend(scores)
        label_list.extend(labels)

    if print_log:
        print("Results on {} set | ".format(subset_name), end="")

    metric_results = calculate_metrics(score_list, label_list, print_log)
    metric_results["scores"] = score_list
    metric_results["labels"] = label_list
    metric_results["loss"] = loss_list

    return metric_results


@hydra.main(version_base=None, config_path="configs", config_name="cfg")
def main(cfg) -> None:
    transformer = make_transformer_image(
        cfg.img.size_h, cfg.img.size_w, cfg.img.image_mean, cfg.img.image_std
    )
    train_dataset, val_dataset, test_dataset, n_train, n_val, n_test = load_data(
        cfg.data.data_path, transformer
    )
    train_batch_gen = batch_generator(
        train_dataset, cfg.nn.batch_size, cfg.hardware.num_workers
    )
    test_batch_gen = batch_generator(
        test_dataset, cfg.nn.batch_size, cfg.hardware.num_workers
    )
    val_batch_gen = batch_generator(
        val_dataset, cfg.nn.batch_size, cfg.hardware.num_workers
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = FullyConnectedNeuralNetwork(
        cfg.img.size_h, cfg.img.size_w, cfg.img.embedding_size, cfg.img.num_classes
    ).model

    # train on mini-batches
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad()
    ckpt_path = cfg.data.ckpt_path
    os.mkdir(ckpt_path)
    ckpt_name = ckpt_path + "model_base.ckpt"
    model = model.to(device)
    metrics_dict = train_model(
        device=device,
        model=model,
        batch_size=cfg.nn.batch_size,
        n_train=n_train,
        train_batch_generator=train_batch_gen,
        val_batch_generator=val_batch_gen,
        opt=opt,
        ckpt_name=ckpt_name,
        n_epochs=5,
    )

    # Evaluate the best model using test set
    with open(ckpt_name, "rb") as f:
        best_model = torch.load(f)

    torch.save(best_model.state_dict(), cfg.data.ckpt_path + "best_model.pt")

    val_stats = test_model(best_model, val_batch_gen, "val")
    test_stats = test_model(best_model, test_batch_gen, "test")

    os.mkdir(cfg.data.ckpt_path + "metrics")
    with open(
        cfg.data.ckpt_path + "metrics/" + "val_stats.json", "w", encoding="utf-8"
    ) as f:
        json.dump(val_stats, f, ensure_ascii=False, indent=4)

    with open(
        cfg.data.ckpt_path + "metrics/" + "val_stats.json", "w", encoding="utf-8"
    ) as f:
        json.dump(test_stats, f, ensure_ascii=False, indent=4)

    with open(
        cfg.data.ckpt_path + "metrics/" + "metrics_dict.json", "w", encoding="utf-8"
    ) as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=4)

    pass


if __name__ == "__main__":
    main()
