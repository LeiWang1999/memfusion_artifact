import tvm
from tvm import relay, ir
import numpy as np


"""
    Legalize op for rocm (like convert erf into float32).
"""


@relay.transform.function_pass(opt_level=0, required=["InferType"])
class LadderOPLegalization(relay.ExprMutator):
    def __init__(self):
        super().__init__()

    def transform_function(self, func, mod, ctx):
        return self.visit(func)

    def visit_call(self, call):
        if isinstance(call.op, ir.Op) and call.op.name in [
            "erf",
        ]:
            for type in call.type_args:
                if type.dtype != "float16":
                    return super().visit_call(call)
            return relay.cast(relay.erf(relay.cast(self.visit(call.args[0]), "float32")), "float16")

        return super().visit_call(call)
