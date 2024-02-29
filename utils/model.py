import torch.nn.functional as F


def act_func_parser(func: str):
    """"""

    match func:
        case "silu":
            return F.silu
        case "gelu":
            return F.gelu
        case "elu":
            return F.elu
        case "leaky_relu":
            return F.leaky_relu
        case "sigmoid":
            return F.sigmoid
        case "soft_sigmoid":
            return F.softsign
        case "tanh":
            return F.tanh
        case "batch_norm":
            return F.batch_norm
        case "linear":
            return F.linear
        case "dropout":
            return F.dropout
        case _:
            print(f"attempted to parse unsupported function {func}, defaulting to ReLU")
            return F.relu
