import warnings
from torch import relu
from torch.nn import Identity, Linear, Module, ModuleList


class DenseNet(Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        context=None,
        node_list=None,
        op_activ=Identity(),
        int_activ=relu,
        **kwargs,
    ):

        super().__init__()
        if node_list is None:
            node_list = [10, 10]
        self.ipdim = input_dim
        self.opdim = output_dim
        self.op_activ = op_activ
        self.int_activ = int_activ
        self.context = context

        self.layernorm = kwargs["layernorm"] if "layernorm" in kwargs else False
        self.batchnorm = kwargs["batchnorm"] if "batchnorm" in kwargs else False
        self.islast = kwargs["islast"] if "islast" in kwargs else True
        self.layers = node_list
        if self.islast:
            self.layers += [output_dim]

        if self.layernorm and self.batchnorm:
            self.batchnorm = 1
            self.layernorm = 0
            warnings.warn(
                "Both layernorm and batchnorm were set to True. Turning off layernorm.",
                RuntimeWarning,
            )

        if self.context is not None:
            self.contextual = Linear(context, self.layers[0])

        self.hidden = ModuleList([Linear(self.ipdim, self.layers[0])])

        self.hidden.extend(
            ModuleList(
                [
                    Linear(self.layers[i], self.layers[i + 1])
                    for i in range(len(self.layers) - 1)
                ]
            )
        )

    def forward(self, x, context=None):
        for i, layer in enumerate(self.hidden[:-1]):
            x = layer(x)
            if (context is not None) and (i == 0):
                x += self.contextual(context)
            x = self.int_activ(x)
        x = self.op_activ(self.hidden[-1](x))
        return x