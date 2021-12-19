import torch
import torch.nn as nn


class MyModule(nn.Module):
    def implicitly_compiled_method(self, x):
        return x + 99

    # `forward` is implicitly decorated with `@torch.jit.export`,
    # so adding it here would have no effect
    def forward(self, x):
        return x + 10

    @torch.jit.export
    def another_forward(
        self, x
    ):  # When the compiler sees this call, it will compile # `implicitly_compiled_method`
        return self.implicitly_compiled_method(x)

    def unused_method(self, x):
        return x - 20


# `m` will contain compiled methods:
#     `forward`
#     `another_forward`
#     `implicitly_compiled_method`
# `unused_method` will not be compiled since it was not called from
# any compiled methods and wasn't decorated with `@torch.jit.export`
m = torch.jit.script(MyModule())
