import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import os

from help_func import self_feeding, enc_self_feeding
from loss_func import total_loss
from nn_structure import AUTOENCODER

def get_model_path(i):
    path1 = f"/home/trarity/master/koopman_control/data/Autoencoder_model_params{i}.pth"
    path2 = f"C:/Users/jokin/Desktop/Uni/Aalborg/Master/Masters_Thesis/Path/Autoencoder_model_params{i}.pth"
    path3 = f"/content/drive/My Drive/Colab Notebooks/Autoencoder_model_params{i}.pth"
    path4 = f"/content/drive/MyDrive/Colab Notebooks/Autoencoder_model_params{i}.pth"
    if os.path.exists(path1):
        return path1
    elif os.path.exists(path2):
        return path2
    elif os.path.exists(path3):
        return path3
    else:
        return path4
def trainingfcn(eps, lr, batch_size, S_p, T, alpha, Num_meas, Num_inputs, Num_x_Obsv, Num_x_Neurons, Num_u_Obsv, Num_u_Neurons, Num_hidden_x_encoder, Num_hidden_x_decoder, Num_hidden_u_encoder, Num_hidden_u_decoder, train_tensor, test_tensor, M, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set pin_memory=True if using a CUDA device
    pin_memory = True if device.type == "cuda" else False

    # Prepare data loaders (datasets remain on CPU)
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    test_dataset = TensorDataset(test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    # Precompute lengths of loaders
    len_train = len(train_loader)
    len_test = len(test_loader)

    # Set up model paths and pre-allocated metric tensors
    Model_path = [get_model_path(i) for i in range(M)]
    Models_loss_list = torch.zeros(M)
    Test_loss_list = torch.zeros(M)
    Running_Losses_Array, Lgx_Array, Lgu_Array, L3_Array, L4_Array, L5_Array, L6_Array = [torch.zeros(M, eps) for _ in range(7)]

    for c_m in range(M):
        model_path_i = Model_path[c_m]
        training_attempt = 0
        while True:  # Re-run training until no NaN is encountered
            training_attempt += 1

            # Instantiate model and optimizer afresh; move model to device
            model = AUTOENCODER(Num_meas, Num_inputs, Num_x_Obsv, Num_x_Neurons, Num_u_Obsv, Num_u_Neurons, Num_hidden_x_encoder, Num_hidden_x_decoder, Num_hidden_u_encoder, Num_hidden_u_decoder).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Pre-allocate per-epoch metric tensors (on CPU)
            loss_list, running_loss_list, Lgx_list, Lgu_list, L3_list, L4_list, L5_list, L6_list = [torch.zeros(eps) for _ in range(8)]

            nan_found = False

            model.train()  # Set model to training mode
            for e in range(eps):
                running_loss, running_Lgx, running_Lgu, running_L3, running_L4, running_L5, running_L6 = [0.0] * 7

                for (batch_x,) in train_loader:
                    # Move the batch to the device
                    batch_x = batch_x.to(device, non_blocking=True)
                    optimizer.zero_grad()
                    loss, L_gx, L_gu, L_3, L_4, L_5, L_6 = total_loss(alpha, xuk, Num_meas, Num_x_Obsv, T, S_p, model)

                    if torch.isnan(loss):
                        nan_found = True
                        break

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    running_Lgx += L_gx.item()
                    running_Lgu += L_gu.item()
                    running_L3 += L_3.item()
                    running_L4 += L_4.item()
                    running_L5 += L_5.item()
                    running_L6 += L_6.item()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

                if nan_found:
                    break

                avg_loss = running_loss / len_train
                loss_list[e] = avg_loss
                running_loss_list[e] = running_loss
                Lgx_list[e] = running_Lgx
                Lgu_list[e] = running_Lgu
                L3_list[e] = running_L3
                L4_list[e] = running_L4
                L5_list[e] = running_L5
                L6_list[e] = running_L6
                print(f'Model: {c_m}, Epoch {e+1}, Avg Loss: {avg_loss:.3e}, Running loss: {running_loss:.3e}')

                # Save model parameters at the end of each epoch if needed
                torch.save(model.state_dict(), model_path_i)

            if not nan_found:
                break
            else:
                print("Restarting training loop due to NaN encountered.\n")

        # Save per-model metrics
        Models_loss_list[c_m] = running_loss  # Last epoch's running loss
        Running_Losses_Array[c_m, :] = running_loss_list
        Lgx_Array[c_m, :] = Lgx_list
        Lgu_Array[c_m, :] = Lgu_list
        L3_Array[c_m, :] = L3_list
        L4_Array[c_m, :] = L4_list
        L5_Array[c_m, :] = L5_list
        L6_Array[c_m, :] = L6_list

        # Final model save
        torch.save(model.state_dict(), model_path_i)

        # Evaluate on test data with no gradient computation
        model.eval()
        test_running_loss = 0.0
        with torch.no_grad():
            for (batch_x,) in test_loader:
                # Move test batch to device
                batch_x = batch_x.to(device, non_blocking=True)
                _, loss = enc_self_feeding(model, batch_x, Num_meas)
                test_running_loss += loss.item()
            print(f'Test Data w/Model {c_m}, Running loss: {test_running_loss:.3e}')
        Test_loss_list[c_m] = test_running_loss

    Lowest_loss = Models_loss_list.min().item()
    Lowest_test_loss = Test_loss_list.min().item()

    # Determine the best models using tensor operations
    Lowest_test_loss_index = int((Test_loss_list == Test_loss_list.min()).nonzero(as_tuple=False)[0].item())
    Best_Model = Model_path[Lowest_test_loss_index]

    return (Lowest_loss, Lowest_test_loss, Best_Model, Lowest_test_loss_index, Lgx_Array, Lgu_Array, L3_Array, L4_Array, L5_Array, L6_Array)
