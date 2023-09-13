import torch

# Sparsity helper functions
def apply_fake_sparsity(model):
    """
    This function simulates 2:4 sparsity on all linear layers in a model.
    It uses the torch.ao.pruning flow.
    """
    # torch.ao.pruning flow
    sparse_config = []
    for name, mod in model.named_modules():
        if all_linear(mod, name):
            sparse_config.append({"tensor_fqn": f"{name}.weight"})

    sparsifier = WeightNormSparsifier(sparsity_level=1.0,
                                      sparse_block_shape=(1,4),
                                      zeros_per_block=2)
    sparsifier.prepare(model, sparse_config)
    sparsifier.step()
    sparsifier.squash_mask()

class SparseLinear(torch.nn.Linear):
    """
    This class is a replacement for `torch.nn.Linear` to support semi-structured sparsity.
    Assuming the weight is already a 2x4 sparse tensor, this module will be numerically equivalent
    to normal dense matmul.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True
    ) -> None:
        super().__init__(in_features, out_features, bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the sparse linear layer.
        We pass in the weight tensor in compressed form.

        Args:
            X (torch.Tensor): The input tensor to the sparse linear layer.

        Returns:
            torch.Tensor: The output of sparse matmul

        """
        shape = X.shape
        # Because we only support the first element being sparse for our matmul, we use transpose
        # properties to reorder our inputs.

        # F.linear = xW' = (xW')'' = (Wx')' = sparse_mm(W, x')'
        return torch._cslt_sparse_mm(
            self.weight_compressed,  # type: ignore[arg-type]
            X.view(-1, shape[-1]).t(),
            self.bias
        ).t().view(*shape[:-1], -1)


    @classmethod
    def from_dense(cls, mod: torch.nn.Linear):
        """
        Converts a `mod` of class `torch.nn.Linear` to the 2:4 sparse version of it.
        This compresses the weights of mod and stores it for future use.

        Args:
            mod (torch.nn.Linear): The original `torch.nn.Linear` module to convert.

        Returns:
            SparseLinear: The converted sparse linear module.

        """

        # create the new module with a toy size to ensure initialization is fast
        fake_in_features, fake_out_features = 8, 8
        new_mod = cls(
            fake_in_features, fake_out_features, bias=mod.bias is not None)
        new_mod.in_features = mod.in_features
        new_mod.out_features = mod.out_features

        # compress old weight into 2:4 compressed representation
        new_mod.register_buffer('weight_compressed', torch._cslt_compress(mod.weight.contiguous()))
        new_mod.bias = mod.bias
        del new_mod.weight

        device_to_use = next(mod.parameters()).device
        new_mod.to(device_to_use)
        return new_mod

def apply_sparse(model):
    apply_fake_sparsity(model)
    from utils import replace_with_custom_fn_if_matches_filter
    replace_with_custom_fn_if_matches_filter(
        model,
        SparseLinear.from_dense,
        lambda mod, fqn: isinstance(mod, torch.nn.Linear))
