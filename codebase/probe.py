import torch
import uuid

class _LogStats(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, name: str):
        uid = str(uuid.uuid4())
        torch.ops.torchprobe.log(x, name, uid)
        ctx.name = name
        ctx.uid = uid
        return x

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        torch.ops.torchprobe.log(grad, f"{ctx.name}.g", ctx.uid)
        return grad, None


_PROBING_ENABLED = False


def log_stats(x: torch.Tensor, name: str) -> torch.Tensor:
    if not _PROBING_ENABLED:
        return x
    return _LogStats.apply(x, name)
