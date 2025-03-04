import matplotlib.pyplot as plt
from help_func import enc_self_feeding

def plot_results(Lgx_Array, Lgu_Array, L3_Array, L4_Array, L5_Array, L6_Array, Lowest_test_loss_index, model, val_tensor, train_tensor, Num_meas, r, enc_self_feeding):

    # Plot style parameters
    title_fontsize = 14
    label_fontsize = 12
    legend_fontsize = 10

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # --- First Figure: Loss Plotting ---
    fig = plt.figure(figsize=(18, 8))
    
    # Top subplot: spans both columns of a 4x2 grid (first row)
    ax_top = plt.subplot2grid((4, 2), (0, 0), colspan=2)
    ax_top.plot(Lgx_Array[int(Lowest_test_loss_index)], label='Lgx', color=colors[0])
    ax_top.plot(Lgu_Array[int(Lowest_test_loss_index)], label='Lgu', color=colors[1])
    ax_top.plot(L3_Array[int(Lowest_test_loss_index)], label='L3', color=colors[2])
    ax_top.plot(L4_Array[int(Lowest_test_loss_index)], label='L4', color=colors[3])
    ax_top.plot(L5_Array[int(Lowest_test_loss_index)], label='L5', color=colors[4])
    ax_top.plot(L6_Array[int(Lowest_test_loss_index)], label='L6', color=colors[5])
    ax_top.legend(loc='upper right', fontsize=legend_fontsize)
    ax_top.set_xlabel('Epochs', fontsize=label_fontsize)
    ax_top.set_ylabel('Loss', fontsize=label_fontsize)
    ax_top.set_title('Losses', fontsize=title_fontsize)
    
    # Lower subplots: arranged as a 3x2 grid.
    ax1 = plt.subplot2grid((4, 2), (1, 0))
    ax1.plot(Lgx_Array[int(Lowest_test_loss_index)], label='Lgx', color=colors[0])
    ax1.legend(loc='upper right', fontsize=legend_fontsize)
    ax1.set_xlabel('Epochs', fontsize=label_fontsize)
    ax1.set_ylabel('Loss', fontsize=label_fontsize)
    ax1.set_title('Lgx', fontsize=title_fontsize)
    
    ax2 = plt.subplot2grid((4, 2), (1, 1))
    ax2.plot(Lgu_Array[int(Lowest_test_loss_index)], label='Lgu', color=colors[1])
    ax2.legend(loc='upper right', fontsize=legend_fontsize)
    ax2.set_xlabel('Epochs', fontsize=label_fontsize)
    ax2.set_ylabel('Loss', fontsize=label_fontsize)
    ax2.set_title('Lgu', fontsize=title_fontsize)
    
    ax3 = plt.subplot2grid((4, 2), (2, 0))
    ax3.plot(L3_Array[int(Lowest_test_loss_index)], label='L3', color=colors[2])
    ax3.legend(loc='upper right', fontsize=legend_fontsize)
    ax3.set_xlabel('Epochs', fontsize=label_fontsize)
    ax3.set_ylabel('Loss', fontsize=label_fontsize)
    ax3.set_title('L3', fontsize=title_fontsize)
    
    ax4 = plt.subplot2grid((4, 2), (2, 1))
    ax4.plot(L4_Array[int(Lowest_test_loss_index)], label='L4', color=colors[3])
    ax4.legend(loc='upper right', fontsize=legend_fontsize)
    ax4.set_xlabel('Epochs', fontsize=label_fontsize)
    ax4.set_ylabel('Loss', fontsize=label_fontsize)
    ax4.set_title('L4', fontsize=title_fontsize)
    
    ax5 = plt.subplot2grid((4, 2), (3, 0))
    ax5.plot(L5_Array[int(Lowest_test_loss_index)], label='L5', color=colors[4])
    ax5.legend(loc='upper right', fontsize=legend_fontsize)
    ax5.set_xlabel('Epochs', fontsize=label_fontsize)
    ax5.set_ylabel('Loss', fontsize=label_fontsize)
    ax5.set_title('L5', fontsize=title_fontsize)
    
    ax6 = plt.subplot2grid((4, 2), (3, 1))
    ax6.plot(L6_Array[int(Lowest_test_loss_index)], label='L6', color=colors[5])
    ax6.legend(loc='upper right', fontsize=legend_fontsize)
    ax6.set_xlabel('Epochs', fontsize=label_fontsize)
    ax6.set_ylabel('Loss', fontsize=label_fontsize)
    ax6.set_title('L6', fontsize=title_fontsize)
    
    plt.tight_layout()
    plt.show()

    # --- Second Figure: Validation Predictions ---
    # Choose three distinct sample indices from validation tensor
    sample_indices = r.sample(range(val_tensor.shape[0]), 3)
    Val_pred_traj, val_loss = enc_self_feeding(model, val_tensor, Num_meas)
    print(f"Running loss for validation: {val_loss:.3e}")

    fig, axs = plt.subplots(2, 3, figsize=(18, 8), sharex=True, sharey=True)
    for i, idx in enumerate(sample_indices):
        predicted_traj = Val_pred_traj[idx]
        actual_traj = val_tensor[idx]
        time_steps = range(val_tensor.shape[1])
        
        # Plot x1 in the first row
        axs[0, i].plot(time_steps, actual_traj[:, 0].cpu().numpy(), 'o-', label='True x1')
        axs[0, i].plot(time_steps, predicted_traj[:, 0].detach().cpu().numpy(), 'x--', label='Predicted x1')
        axs[0, i].set_title(f"Validation Sample {idx} (x1)")
        axs[0, i].set_xlabel("Time step")
        axs[0, i].set_ylabel("x1")
        axs[0, i].legend()
        
        # Plot x2 in the second row
        axs[1, i].plot(time_steps, actual_traj[:, 1].cpu().numpy(), 'o-', label='True x2')
        axs[1, i].plot(time_steps, predicted_traj[:, 1].detach().cpu().numpy(), 'x--', label='Predicted x2')
        axs[1, i].set_title(f"Validation Sample {idx} (x2)")
        axs[1, i].set_xlabel("Time step")
        axs[1, i].set_ylabel("x2")
        axs[1, i].legend()
    
    plt.tight_layout()
    plt.show()

    # --- Third Figure: Training Predictions ---
    # Choose three distinct sample indices from training tensor
    sample_indices = r.sample(range(train_tensor.shape[0]), 3)
    train_pred_traj, train_loss = enc_self_feeding(model, train_tensor, Num_meas)
    print(f"Running loss for training: {train_loss:.3e}")

    fig, axs = plt.subplots(2, 3, figsize=(18, 8), sharex=True, sharey=True)
    for i, idx in enumerate(sample_indices):
        predicted_traj = train_pred_traj[idx]
        actual_traj = train_tensor[idx]
        time_steps = range(train_tensor.shape[1])
        
        # Plot x1 in the first row
        axs[0, i].plot(time_steps, actual_traj[:, 0].cpu().numpy(), 'o-', label='True x1')
        axs[0, i].plot(time_steps, predicted_traj[:, 0].detach().cpu().numpy(), 'x--', label='Predicted x1')
        axs[0, i].set_title(f"Train Sample {idx} (x1)")
        axs[0, i].set_xlabel("Time step")
        axs[0, i].set_ylabel("x1")
        axs[0, i].legend()
        
        # Plot x2 in the second row
        axs[1, i].plot(time_steps, actual_traj[:, 1].cpu().numpy(), 'o-', label='True x2')
        axs[1, i].plot(time_steps, predicted_traj[:, 1].detach().cpu().numpy(), 'x--', label='Predicted x2')
        axs[1, i].set_title(f"Train Sample {idx} (x2)")
        axs[1, i].set_xlabel("Time step")
        axs[1, i].set_ylabel("x2")
        axs[1, i].legend()
    
    plt.tight_layout()
    plt.show()

