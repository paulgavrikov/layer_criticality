import torch
import torch.utils.data
from tqdm import tqdm
import argparse
from utils import seed_everything, str2bool
from models import get_normalized_model
import os
import time
from utils import AverageMeter, accuracy, get_module_by_name, get_imagenet_folder_loader
import torch.nn.functional as F

try:
    import wandb

    HAS_WANDB = True
except:
    HAS_WANDB = False


def istype(a, t):
    return type(a) == t


TYPE_FILTER = lambda nt: type(nt[1]) in [
    torch.nn.Conv2d,
    torch.nn.Linear,
    torch.nn.BatchNorm2d,
]


def eval_loop(
    model,
    dataloader,
    device,
    desc=None,
    max_batches=-1,
    return_batches=False,
    return_logits=False,
):
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    loss_meter = AverageMeter()

    batch_acc = []
    all_logits = []

    if max_batches != -1:
        progress = tqdm(dataloader, desc=desc, total=max_batches)
    else:
        progress = tqdm(dataloader, desc=desc)

    for batch_idx, (x, y) in enumerate(progress):
        bx = x.to(device)
        by = y.to(device)

        with torch.no_grad():
            logits = model(bx)

        loss = F.cross_entropy(logits, by)
        top1, top5 = accuracy(logits, by, topk=(1, 5))

        if return_batches:
            batch_acc.append(top1.item())

        if return_logits:
            all_logits.append(logits.detach().cpu())

        loss_meter.update(loss.item(), bx.size(0))
        top1_meter.update(top1.item(), bx.size(0))
        top5_meter.update(top5.item(), bx.size(0))

        progress.set_postfix(
            {"top1": top1_meter.avg, "top5": top5_meter.avg, "loss": loss_meter.avg}
        )

        # Early stopping
        if max_batches != -1 and batch_idx + 1 >= max_batches:
            progress.update(max_batches)
            break

    progress.close()

    all_logits = torch.cat(all_logits, dim=0) if return_logits else None
    batch_acc = batch_acc if return_batches else None

    return {
        "top1_acc": top1_meter.avg,
        "top5_acc": top5_meter.avg,
        "loss": loss_meter.avg,
        "batch_acc": batch_acc,
        "logits": all_logits,
    }


def _test_individual(name, model, args):
    data_seed = args.data_seed if args.data_seed >= 0 else args.seed
    print(f"Testing {name}, data_seed: {data_seed}, seed: {args.seed}")

    dataloader = get_imagenet_folder_loader(
        path=args.imagenet,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        resize_crop=True,
        seed=data_seed,
    )

    row = eval_loop(
        model,
        dataloader,
        args.device,
        max_batches=args.num_batches,
        return_batches=args.log_batches,
        return_logits=True,
    )

    row["layer"] = name

    return row


def pred_similarity(a, b):
    assert len(a.shape) == 2, "Logits must be (b, p) shaped"
    assert a.shape == b.shape, "Logits must be comparable"

    with torch.no_grad():
        # convert to confidences
        a_probs = torch.nn.functional.softmax(a, dim=1)
        b_probs = torch.nn.functional.softmax(b, dim=1)

        cos_sim = torch.nn.CosineSimilarity(dim=1)(a_probs, b_probs).mean().item()
        l1_dist = torch.norm(a_probs - b_probs, p=1, dim=1).mean().item()
        l2_dist = torch.norm(a_probs - b_probs, p=2, dim=1).mean().item()

        # this is computed on the logits, not probs! Doesnt work well on fp16 ..
        ce_loss = torch.nn.CrossEntropyLoss()(a.float(), b.float()).item()

    return cos_sim, l1_dist, l2_dist, ce_loss


def main(args):

    run = None
    if HAS_WANDB and args.wandb:
        run = wandb.init(project="weight_reset", config=args)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_path, f"{timestamp}_{args.model}.pt")
    print(f"Results will be written to {output_file}")

    model = get_normalized_model(args.model)
    model.eval()

    # info log
    print("Affected modules:")
    for module_name, module in filter(TYPE_FILTER, model.named_modules()):
        print(f"- {module_name} ({module})")
        for i, p in enumerate(module.parameters()):
            print(f"-> Param {i}: {p.shape}")

    # create a backup of the original weights
    state_dict = model.state_dict().copy()
    model.to(args.device)

    rows = []
    original_result = _test_individual("original", model, args)
    original_logits = original_result["logits"].clone()
    if not args.save_logits:
        del original_result["logits"]
    rows.append(original_result)

    for trial in range(args.n_trials):

        # load a random model
        random_model = get_normalized_model(args.model, pretrained=False)

        for module_name, module in filter(TYPE_FILTER, model.named_modules()):

            # set the original weights
            model.load_state_dict(state_dict)

            # reset only the module we test
            with torch.no_grad():
                if args.zero_weights:
                    # zero weights
                    for p in module.parameters():
                        p.data.zero_()
                else:
                    # load random weights from a fresh model
                    random_module_state = get_module_by_name(
                        random_model, module_name
                    ).state_dict()
                    module.load_state_dict(random_module_state)
                    # push back to correct device
                    module.to(args.device)

            result = _test_individual(module_name, model, args)

            # compute prediction distance
            cos_sim, l1_dist, l2_dist, ce_loss = pred_similarity(
                original_logits, result["logits"]
            )
            result["pred_cos_sim"] = cos_sim
            result["pred_l1_err"] = l1_dist
            result["pred_l2_err"] = l2_dist
            result["pred_ce_loss"] = ce_loss

            if not args.save_logits:
                del result["logits"]

            result["trial"] = trial

            # logging and tracking
            if run:
                run.log(result)
            print(result)
            rows.append(result)

            os.makedirs(args.output_path, exist_ok=True)
            torch.save({"args": vars(args), "trials": rows}, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument(
        "--imagenet", type=str, default="/workspace/data/datasets/imagenet"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_seed", type=int, default=-1)
    parser.add_argument("--output_path", type=str, default="measurements/weight_reset/")
    parser.add_argument("--wandb", type=str2bool, default=False)
    parser.add_argument("--log_batches", type=str2bool, default=False)
    parser.add_argument("--zero_weights", type=str2bool, default=False)
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument("--save_logits", type=str2bool, default=True)

    args = parser.parse_args()

    seed_everything(args.seed)

    assert args.n_trials >= 1, "Must do at least 1 trial"

    if args.num_batches > 0 and args.num_workers > args.num_batches:
        print(
            f"Having more workers than batches is pointless, setting num_workers to {args.num_batches}"
        )
        args.num_workers = args.num_batches

    main(args)
