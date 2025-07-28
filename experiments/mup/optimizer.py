from codebase.optim import OptimArgs, build_lr_fn


class MupOptimArgs(OptimArgs):
    scaling_factor: float = None


def build_mup_optimizer(
    model: nn.Module,
    args: OptimArgs,
    n_steps: int,
):
    mup_decay_params = []
    decay_params = []
    nodecay_params = []
    for n, p in model.named_parameters():
        if p.dim() >= 2:
            if (
                n.endswith("wq.weight")
                or n.endswith("wk.weight")
                or n.endswith("wv.weight")
                or n.endswith("wo.weight")
                or n.endswith("w1.weight")
                or n.endswith("w2.weight")
                or n.endswith("w3.weight")
            ):
                print(f"Added {n} to mup_decay_list")
                mup_decay_params.append(p)
            else:
                decay_params.append(p)
        else:
            nodecay_params.append(p)
    optim_groups = [
        {
            "params": mup_decay_params,
            "weight_decay": args.weight_decay,
            "lr_scale": (1 / args.scaling_factor),
        },
        {"params": decay_params, "weight_decay": args.weight_decay, "lr_scale": 1},
        {"params": nodecay_params, "weight_decay": 0.0, "lr_scale": 1},
    ]
    num_mup_decay_params = sum(p.numel() for p in mup_decay_params)
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(
        f"num mup decayed parameter tensors: {len(mup_decay_params)}, with {num_mup_decay_params:,} parameters"
    )
    print(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )

    optimizer = AdamW(
        optim_groups,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.epsilon,
        fused=True,  # Faster optim.step but can throw errors
    )

    # scheduler
    lr_fn = build_lr_fn(args, n_steps)
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_fn
    )  # lr_scheduler.LambdaLR(optimizer, lr_fn)

    return optimizer, scheduler
