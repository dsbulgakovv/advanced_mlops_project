import hydra
import pandas as pd
import torch
import torch.nn as nn
from nn_model import FullyConnectedNeuralNetwork
from train import test_model
from utils import batch_generator, load_data, make_transformer_image


@hydra.main(version_base=None, config_path="configs", config_name="cfg")
def infer_model(cfg):
    """Infer model on test dataset"""

    transformer = make_transformer_image(
        cfg.img.size_h, cfg.img.size_w, cfg.img.image_mean, cfg.img.image_std
    )
    _, _, test_dataset, _, _, n_test = load_data("../" + cfg.data.data_path, transformer)
    test_batch_gen = batch_generator(
        test_dataset, cfg.nn.batch_size, cfg.hardware.num_workers
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    with open("../" + cfg.data.ckpt_path + cfg.data.best_model_file, "rb") as f:
        model = FullyConnectedNeuralNetwork(
            cfg.img.size_h, cfg.img.size_w, cfg.nn.embedding_size, cfg.data.num_classes
        ).model
        model.load_state_dict(torch.load(f))

    metric_results = test_model(
        device=device,
        loss_function=nn.CrossEntropyLoss(),
        model=model,
        batch_generator_f=test_batch_gen,
        subset_name="test",
        print_log=False,
    )
    files = list(map(lambda x: x[0].split("/")[-1], test_dataset.imgs))
    result_df = pd.DataFrame(
        {
            "filename": files,
            "predicted_label": metric_results.get("labels"),
            "predicted_category": metric_results.get("labels"),
        }
    )
    result_df["predicted_category"].replace({0: "cat", 1: "dog"}, inplace=True)
    result_path = "../" + cfg.data.ckpt_path + cfg.data.inference_file
    result_df.to_csv(result_path, index=False)


if __name__ == "__main__":
    infer_model()
