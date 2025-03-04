import matplotlib.pyplot as plt
import numpy as np
import torch
from plotting import plot_results

plot_results(Lgx_Array, Lgu_Array, L3_Array, L4_Array, L5_Array, L6_Array,
             Lowest_test_loss_index, model, val_tensor, train_tensor,
             Num_meas, r, enc_self_feeding)
