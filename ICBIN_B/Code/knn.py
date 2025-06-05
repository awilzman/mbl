
import torch


def knn(
    x: torch.Tensor,
    y: torch.Tensor,
    k: int,
    batch_x: torch.Tensor = None,
    batch_y: torch.Tensor = None,
    cosine: bool = False,
    num_workers: int = 1,
    batch_size: int = None,
) -> torch.Tensor:

    if x.numel() == 0 or y.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long, device=x.device)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y
    x, y = x.contiguous(), y.contiguous()

    if batch_size is None:
        batch_size = 1
        if batch_x is not None:
            assert x.size(0) == batch_x.numel()
            batch_size = int(batch_x.max()) + 1
        if batch_y is not None:
            assert y.size(0) == batch_y.numel()
            batch_size = max(batch_size, int(batch_y.max()) + 1)
    assert batch_size > 0

    ptr_x: torch.Tensor = None
    ptr_y: torch.Tensor = None
    if batch_size > 1:
        assert batch_x is not None
        assert batch_y is not None
        arange = torch.arange(batch_size + 1, device=x.device)
        ptr_x = torch.bucketize(arange, batch_x)
        ptr_y = torch.bucketize(arange, batch_y)

    return torch.ops.torch_cluster.knn(x, y, ptr_x, ptr_y, k, cosine,
                                       num_workers)


def knn_graph(
    x: torch.Tensor,
    k: int,
    batch: torch.Tensor = None,
    loop: bool = False,
    flow: str = 'source_to_target',
    cosine: bool = False,
    num_workers: int = 1,
    batch_size: int = None,
) -> torch.Tensor:


    assert flow in ['source_to_target', 'target_to_source']
    edge_index = knn(x, x, k if loop else k + 1, batch, batch, cosine,
                     num_workers, batch_size)

    if flow == 'source_to_target':
        row, col = edge_index[1], edge_index[0]
    else:
        row, col = edge_index[0], edge_index[1]

    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]

    return torch.stack([row, col], dim=0)