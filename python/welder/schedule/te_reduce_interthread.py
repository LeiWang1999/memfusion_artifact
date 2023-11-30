import numpy as np
from tvm import te
import tvm

from ..config import Config, Stride
from .te_base import TESchedulerBase

# for debugging.
import os
# get file name and remove the suffix
fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
# create log path
log_path = "progress/" + fname
count = 0

def write_code(code, path, fname):
    global count
    # if path not exist, create it
    fname = str(count) + "." + fname
    count += 1
    if not os.path.exists(path):
        os.makedirs(path)
    # join path and fname
    fname = os.path.join(path, fname)
    with open(fname, "w") as f:
        f.write(code)

class TEReduceInterThreadScheduler(TESchedulerBase):
    def schedule(self) -> te.Schedule:
        sch, config = self.sche, self.config
        for op in self.ops:
            if op is not self.output_op:
                sch[op].compute_inline()
        block = self.config.block
        reduce_thread = self.config.reduce_thread
        thread = self.config.thread
        rstep = self.config.rstep

        assert(self.reduce_op is not None)
        node_shape = self.reduce_op.output(0).shape
        
        # special case for mi250 device's transpose mean 
        if len(block) == 3 and len(self.output_args) != 1 and node_shape[-1] == 1:
            print("special case for mi250 device's transpose mean")
            block = [4, 8, 1]
            reduce_thread = [16]
            thread = [4, 8, 1]
            rstep = [128]
        #  256 * 16 * 128
        # assign grid_size
        grid_size = 1
        for t, n in zip(block, node_shape):
            grid_size *= (n + t - 1) // t
        self.grid_size[0] = grid_size
        
        out = self.output_op
        reg_tile = self.reduce_op.output(0)
        self.block_size[0] = int(np.prod(reduce_thread))
        self.block_size[1] = int(np.prod(thread))
        
        # For inter thread reduction case, one thread must only compute one element
        assert thread == block

        blck_axis = []
        thrd_axis = []

        for i, axis in enumerate(sch[out].op.axis):
            bx, tx = sch[out].split(axis, factor=block[i])
            blck_axis.append(bx)
            thrd_axis.append(tx)
        
        axis_order = blck_axis + thrd_axis
        sch[out].reorder(*axis_order)
        blck_fused = sch[out].fuse(*blck_axis)
        thrd_fused = sch[out].fuse(*thrd_axis)
        sch[out].bind(blck_fused, te.thread_axis("blockIdx.x"))
        sch[out].bind(thrd_fused, te.thread_axis("threadIdx.y"))
        if out is not self.reduce_op:
            sch[reg_tile].compute_at(sch[out], thrd_fused)
        write_code(
            str(tvm.lower(sch, self.args, simple_mode=True)), log_path, 'split.py')
        reduce_outer_axis, reduce_inner_axis, reduce_inter_threads = [], [], []

        for i in self.config.raxis_order:
            axis = sch[reg_tile].op.reduce_axis[i]
            ro, _t = sch[reg_tile].split(axis, factor=rstep[i])
            ri, thd = sch[reg_tile].split(_t, factor=reduce_thread[i])
            reduce_inter_threads.append(thd)
            reduce_outer_axis.append(ro)
            reduce_inner_axis.append(ri)
        axis_order = reduce_inter_threads + reduce_outer_axis + reduce_inner_axis
        sch[reg_tile].reorder(*axis_order)
        fused_reduce_inter_threads = sch[reg_tile].fuse(*reduce_inter_threads)
        sch[reg_tile].bind(fused_reduce_inter_threads, te.thread_axis("threadIdx.x"))

        for input_tensor in self.reduce_op.input_tensors:
            shared_tensor = sch.cache_read(input_tensor, "shared", [reg_tile])
            if input_tensor in self.shared_inputs:
                sch[shared_tensor].compute_at(sch[out], blck_fused)
                strides = self.shared_inputs_strides[input_tensor]
            else:
                sch[shared_tensor].compute_at(sch[reg_tile], reduce_outer_axis[-1])
                strides = Stride()
            if input_tensor.name in self.config.vectorize and not self._is_from_shared(input_tensor):
                vectorize = self.config.vectorize[input_tensor.name]
                if input_tensor not in self.args:
                    vectorize = min(4 , vectorize) # tvm not supporting ramp for 8 elements
            else:
                vectorize = 1
            self.cooperative_fetch(shared_tensor, strides, vectorize)

        cache_plan = {}
        for op in self.none_reduce_ops:
            for tensor in op.input_tensors:
                if self.requires_cache(tensor, op):
                    if tensor not in cache_plan:
                        cache_plan[tensor] = []
                    cache_plan[tensor].append(op)

        for tensor, consumers in cache_plan.items():
            tensor_shared = sch.cache_read(tensor, "shared", consumers)
            sch[tensor_shared].compute_at(sch[out], thrd_fused)
            if tensor in self.shared_inputs_strides:
                strides = self.shared_inputs_strides[tensor]
            else:
                strides = Stride()
            self.cooperative_fetch(tensor_shared, strides)
            # This is a hack, TVM cannot handle cached_local_read when padding on a shared input
            consumers = list(filter(lambda x: x.output(0) not in self.reduce_op.input_tensors, consumers))
            if len(consumers) == 0 or len(self.shared_outputs) == 0: continue
            tensor_local = sch.cache_read(tensor_shared, "local", consumers)
            sch[tensor_local].compute_at(sch[out], thrd_fused)
        write_code(
            str(tvm.lower(sch, self.args, simple_mode=True)), log_path, 'cached.py')
        return sch
