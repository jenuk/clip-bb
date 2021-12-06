import argparse, os

import torch
import pandas as pd
from omegaconf import OmegaConf

from tqdm.auto import tqdm

import clip

import config2object 


@torch.no_grad()
def clip_bb(model_clip, dataloader, out_data, device):
    for slides, num_slides, starts, tokens, extra in tqdm(dataloader):
        B, max_slides, _, _, _ = slides.shape

        slides = torch.flatten(slides, 0, 1).to(device)
        img_emb = model_clip.encode_image(slides).reshape((B, num_imgs, -1))
        # expect shape (B, max_slides, m)

        tokens = tokens.to(device)
        cap_emb = model_clip.encode_text(tokens).unsqueeze(1)
        # expect shape (B, 1, m)
        # add dimension to make broadcastable

        sims = torch.nn.functional.cosine_similarity(img_emb, cap_emb, dim=2)
        # sims = torch.bmm(img_emb, cap_emb).squeeze(2)
        # expect shape (B, max_slides)

        # mask padded images
        mask = torch.arange(sims.shape[1]).unsqueeze(0) >= num_slides.unsqueeze(1)
        sims[mask] = float("-inf")

        vals, best = torch.max(sims, dim=1)
        vals, best = vals.cpu(), best.cpu()
        # shape (B,)

        start = torch.tensor([starts[k][best[k]] for k in range(B)])
        out_data["start_x"].extend(start[:, 0].tolist())
        out_data["start_y"].extend(start[:, 1].tolist())
        out_data["similarity"].extend(vals.tolist())
        for key in extra:
            out_data[key].extend(extra[key])

    return out_data


def collate_fn(slides, starts, tokens, extra):
    # aggregate images with different number of slides by adding zero-slides
    
    extra = {key: [entry[key] for entry in extra] for key in extra[0]}

    num_slides = torch.tensor([t.shape[0] for t in res["slides_clip"]], dtype=torch.long)
    max_slides = torch.max(num_slides)
    B = num_slides.shape[0]
    img_shape = slides[0].shape[1:] # should always be (3, 224, 224)

    slides_uniform = torch.zeros((B, max_slides, *img_shape), dtype=torch.float)
    for k, imgs in enumerate(slides):
        slides_uniform[k, :num_slides[k]] = imgs

    tokens = torch.stack(tokens)
    starts = torch.stack(starts)

    return slides_uniform, num_slides, starts, tokens, extra


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--gpu",
        help="which gpu to use (-1 for none)",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Number of pixels window is moved with each step",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=30,
        help="batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="number of dataloader workers",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        help="which CLIP model to use",
        default="ViT-B/16",
    )
    parser.add_argument(
        "configs",
        type=str,
        nargs="+",
        help="config files for datasets",
    )

    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_arguments()

    device = "cpu" if opt.gpu == -1 else f"cuda:{opt.gpu}"
    print(f"Using device {device}")
    device = torch.device(device)

    # load clip module
    print("Load CLIP")
    model_clip, _ = clip.load(opt.clip_model, device=device)

    # create dataset
    print("Load dataset")
    datasets = []
    for config_fn in opt.configs:
        config = OmegaConf.load(config_fn)
        dataset = config2object.instantiate_from_config(config["dataset"])
        datasets.append((dataset, config))


    print("Evaluate best clipping")
    for dataset, config in datasets:
        df = {key: [] for key in dataset.extra_keys}

        dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=opt.batch_size,
                num_workers=opt.num_workers,
                collate_fn=collate_fn,
        )
        clip_bb(model_clip, dataloader, csv, device)

        df = pd.DataFrame(df)
        df.to_csv(os.path.join("out", config["csv_name"]))
