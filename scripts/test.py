import torch
from lightglue import LightGlue, SuperPoint, match_pair

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision('high')

extractor = SuperPoint(max_num_keypoints=512).eval().cuda()
matcher = LightGlue(features="superpoint").eval().cuda()

# same size as your ros images
dummy0 = torch.zeros(3, 480, 640).cuda()
dummy1 = torch.zeros(3, 480, 640).cuda()

# warmup
for _ in range(10):
    match_pair(extractor, matcher, dummy0, dummy1)

# time it
starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)
times = []
for _ in range(50):
    starter.record()
    match_pair(extractor, matcher, dummy0, dummy1)
    ender.record()
    torch.cuda.synchronize()
    times.append(starter.elapsed_time(ender))

import numpy as np
print(f"mean: {np.mean(times):.1f}ms")
print(f"min: {min(times):.1f}ms")
