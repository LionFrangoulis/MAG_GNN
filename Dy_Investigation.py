import torch
import torch.optim as optim 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import GNN

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def R2(y_exp,y_pred):
    assert len(y_exp)==len(y_pred)
    resid=[(y_exp[i]-y_pred[i])**2 for i in range(len(y_exp))]
    var=[[(y_exp[i]-np.mean(y_exp))**2 for i in range(len(y_exp))]]
    R2=1-np.sum(resid)/np.sum(var)
    return R2
    

def analyse_model(size, run, model,train_loader, test_loader, save_on_disc=False):
    # Loop over the test set
    y_train=torch.tensor([])
    y_pred_train=torch.tensor([])
    identifiers_train=[]
    with torch.no_grad():  # Disable gradient calculation for testing
        for batch_data in train_loader:
            x_train_batch, elements_train, y_train_batch, identifiers_train_batch, ligands, train_atom_numbers = batch_data
            mask=torch.tensor([[1]*atom_number+[0]*(train_atom_numbers[-1]-atom_number) for atom_number in train_atom_numbers]).to(device)
            identifiers_train=np.concatenate([identifiers_train,identifiers_train_batch])
            x_train_batch=x_train_batch.to(device)
            y_train_batch=y_train_batch.to(device)
            y_pred_train_batch = model(elements_train,x_train_batch, train_atom_numbers[-1], mask)
            y_train=torch.cat([y_train, y_train_batch])
            y_pred_train=torch.cat([y_pred_train, y_pred_train_batch])
    
    train_loss=dnn_loss(y_train,y_pred_train)
    Mean_error_train=torch.mean(torch.abs(y_train-y_pred_train))
    train_R2=R2(y_train.detach().numpy(), y_pred_train.detach().numpy())
    
    y_test=torch.tensor([])
    y_pred_test=torch.tensor([])
    identifiers_test=[]
    with torch.no_grad():  # Disable gradient calculation for testing
        for batch_data in test_loader:
            x_test_batch, elements_test, y_test_batch, identifiers_test_batch, ligands, test_atom_numbers = batch_data
            mask=torch.tensor([[1]*atom_number+[0]*(test_atom_numbers[-1]-atom_number) for atom_number in test_atom_numbers]).to(device)
            identifiers_test=np.concatenate([identifiers_test,identifiers_test_batch])
            x_test_batch=x_test_batch.to(device)
            y_test_batch=y_test_batch.to(device)
            y_pred_test_batch = model(elements_test,x_test_batch, test_atom_numbers[-1], mask)
            y_test=torch.cat([y_test, y_test_batch])
            y_pred_test=torch.cat([y_pred_test, y_pred_test_batch])
            
    test_loss=dnn_loss(y_test,y_pred_test)
    Mean_error_test=torch.mean(torch.abs(y_test-y_pred_test))
    test_R2=R2(y_test.detach().numpy(), y_pred_test.detach().numpy())
    
    if save_on_disc:
        with open("./results_{}_{}.txt".format(size,run),"w") as f:
            
            f.write("RESULTS FOR SIZE {} RUN {}\n".format(size,run))
            f.write("normalised train loss: {}\n".format(train_loss.item()))
            f.write("rescaled train loss: {}\n".format(train_loss.item()*std**2))
            f.write("Mean train error: {}\n".format(Mean_error_train.item()))
            f.write("Mean rescaled train error: {}\n".format(Mean_error_train.item()*std))
            f.write("R2 train: {}\n".format(train_R2))
            f.write("normalised test loss: {}\n".format(test_loss.item()))
            f.write("rescaled test loss: {}\n".format(test_loss.item()*std**2))
            f.write("Mean test error: {}\n".format(Mean_error_test.item()))
            f.write("Mean rescaled test error: {}\n".format(Mean_error_test.item()*std))
            f.write("R2 test: {}\n".format(test_R2))
    else:
        print("RESULTS FOR SIZE {} RUN {}\n".format(size,run))
        print("normalised train loss: {}\n".format(train_loss.item()))
        print("rescaled train loss: {}\n".format(train_loss.item()*std**2))
        print("Mean train error: {}\n".format(Mean_error_train.item()))
        print("Mean rescaled train error: {}\n".format(Mean_error_train.item()*std))
        print("R2 train: {}\n".format(train_R2))
        print("normalised test loss: {}\n".format(test_loss.item()))
        print("rescaled test loss: {}\n".format(test_loss.item()*std**2))
        print("Mean test error: {}\n".format(Mean_error_test.item()))
        print("Mean rescaled test error: {}\n".format(Mean_error_test.item()*std))
        print("R2 test: {}\n".format(test_R2))

    label_list=["N-","O-","N","O"]
    line_x=[torch.min(y_train).item()*std+mean,torch.max(y_train).item()*std+mean]
    for i in range(len(label_list)):
    
        y_test_element=np.array([y_test[j].cpu() for j in range(len(y_test.cpu())) if identifiers_test[j].item()==i])
        y_test_pred_element=np.array([y_pred_test[j].cpu() for j in range(len(y_pred_test.cpu())) if identifiers_test[j].item()==i])
        
        plt.scatter(y_test_element*std+mean, y_test_pred_element*std+mean, label="test "+label_list[i], s=2, marker=".")
    plt.xlabel("real energy normalised")
    plt.ylabel("predicted energy normalised")
    plt.plot(line_x,line_x, c='black', lw = 1.1)#, zorder=0)
    plt.legend()
    plt.title("Test error")
    if save_on_disc:
        plt.savefig("Parity_Train.png",dpi=300)
    else:  
        plt.show()
    plt.close()
    
    for i in range(len(label_list)):
        y_train_element=np.array([y_train[j].cpu() for j in range(len(y_train.cpu())) if identifiers_train[j].item()==i])
        y_train_pred_element=np.array([y_pred_train[j].cpu() for j in range(len(y_pred_train.cpu())) if identifiers_train[j].item()==i])
        
        plt.scatter(y_train_element*std+mean, y_train_pred_element*std+mean, label="train "+label_list[i], s=2, marker=",")
    plt.xlabel("real energy normalised")
    plt.ylabel("predicted energy normalised")
    plt.plot(line_x,line_x, c='black', lw = 1.1)#, zorder=0)
    plt.legend()
    plt.title("Train error")
    if save_on_disc:
        plt.savefig("Parity_Test.png",dpi=300)
    else:  
        plt.show()
    plt.close()



def load_train_dataset(blocks, filter_number, location, masked=False):
    '''
    All info used as features are previously generated and stored. This then load in those info for the model.
    QUESTION IS WHY IS DATA SPLITTED IN TRAIN AND TEST AND LATER LOADED INTO THE MODEL? 
    '''
    dataset=[]
    for block in blocks:
        atom_numbers=np.loadtxt("{}/Train_Block_{}_atom_numbers.txt".format(location,block), dtype="int")
        elements=np.load("{}/Train_Block_{}_elements.npy".format(location,block))
        energies=np.loadtxt("{}/Train_Block_{}_energies.txt".format(location,block))
        identifiers=np.loadtxt("{}/Train_Block_{}_identifiers.txt".format(location,block), dtype="int")
        ligands=np.loadtxt("{}/Train_Block_{}_ligands.txt".format(location,block), dtype="str")
        if masked:
            mask=torch.tensor([[True]*atom_numbers[i]+[False]*(atom_numbers[-1]-atom_numbers[i]) for i in range(500)])
            filters=torch.masked.masked_tensor(torch.tensor(np.load("{}/Train_Block_{}_Filters_number_{}.npy".format(location,block, filter_number)),dtype=torch.float32), mask)
        else:
            filters=torch.tensor(np.load("{}/Train_Block_{}_Filters_number_{}.npy".format(location,block, filter_number)),dtype=torch.float32)
        dataset.append((filters, elements, torch.tensor(energies,dtype=torch.float32), identifiers, ligands, atom_numbers))
    return(dataset)

def load_test_dataset(blocks, filter_number, location,masked=False):
    dataset=[]
    for block in blocks:
        atom_numbers=np.loadtxt("{}/Test_Block_{}_atom_numbers.txt".format(location,block), dtype="int")
        elements=np.load("{}/Test_Block_{}_elements.npy".format(location,block))
        energies=np.loadtxt("{}/Test_Block_{}_energies.txt".format(location,block))
        identifiers=np.loadtxt("{}/Test_Block_{}_identifiers.txt".format(location,block), dtype="int")
        ligands=np.loadtxt("{}/Test_Block_{}_ligands.txt".format(location,block), dtype="str")
        if masked:
            mask=torch.tensor([[True]*atom_numbers[i]+[False]*(atom_numbers[-1]-atom_numbers[i]) for i in range(500)])
            filters=torch.masked.masked_tensor(torch.tensor(np.load("{}/Test_Block_{}_Filters_number_{}.npy".format(location,block, filter_number)),dtype=torch.float32), mask)
        else:
            filters=torch.tensor(np.load("{}/Test_Block_{}_Filters_number_{}.npy".format(location,block, filter_number)),dtype=torch.float32)
        dataset.append((filters, elements, torch.tensor(energies,dtype=torch.float32), identifiers, ligands, atom_numbers))
    return(dataset)

        
def dnn_loss(y_pred, y_true):
    return nn.MSELoss(reduction='mean')(y_pred, y_true)

def plot_loss_growth(history, filename=None):
    history = np.array(history)
    plt.figure(figsize=(8, 5))
    plt.plot(history[:,1], label="Training Loss")
    plt.plot(history[:,2], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.yscale("log")
    if filename==None:
        plt.show()
    else:
        plt.savefig(filename, dpi=300)
        plt.close()

def train_SchNet(feature_number, element_number, filter_number, layer_number, epochs, train_loader, test_loader, start_learning_rate,pooling, factor=0.5, patience=3, threshold=1e-4, min_lr=1e-6):
    history = []
    
    model = GNN.SchNet(feature_number,element_number,filter_number, layer_number,pooling,ghost_killer=True).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=start_learning_rate)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',        # Use 'min' for loss, 'max' for accuracy
        factor=factor,        # Multiplier for LR. new_lr = lr * factor
        patience=patience,
        threshold=threshold,
        min_lr=min_lr
    )
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        total_samples = 0
        losses=[]
        for batch_data in train_loader:
            x_train, elements_train, y_train, identifiers, ligands, atom_numbers = batch_data
            mask=torch.tensor([[1]*atom_number+[0]*(atom_numbers[-1]-atom_number) for atom_number in atom_numbers]).to(device)
            #mask=None
            x_train=x_train.to(device)
            y_train=y_train.to(device)
            y_pred = model(elements_train,x_train,atom_numbers[-1], mask)
            loss = dnn_loss(y_pred, y_train.view(-1))
            optimizer.zero_grad() 
            loss.backward()
            losses.append(loss)
            train_loss += loss.detach().item() * x_train.size(0)
            total_samples += x_train.size(0)  # Count samples in the batch
            optimizer.step() # update the parameters 
        avg_loss = train_loss / total_samples  # Normalize by number of batches
        scheduler.step(avg_loss)
        
        model.eval()
        val_loss = 0
        total_val_samples = 0 
        val_losses=[]
        with torch.no_grad():
            for batch_data in test_loader:
                x_val, elements_val, y_val, _, _, atom_numbers = batch_data
                # x_val the RBF filters (N,F)
                # elements_val the elements in each mol (N,A)
                # atoms_no the original no of atoms in each mol (N)
                
                x_val=x_val.to(device)
                y_val=y_val.to(device)
                mask=torch.tensor([[1]*atom_number+[0]*(atom_numbers[-1]-atom_number) for atom_number in atom_numbers]).to(device)
                #mask=None
                yval_pred = model(elements_val,x_val,atom_numbers[-1], mask) # the last one has the largest size 
                loss = dnn_loss(yval_pred, y_val.view(-1))
                val_losses.append(loss)
                val_loss += loss.detach().item() * x_val.size(0)
                total_val_samples += x_val.size(0)  # Count samples in the batch
        avg_val_loss = val_loss / total_val_samples  # Normalize by number of batches
        history.append([epoch,avg_loss, avg_val_loss])

        current_lr = get_lr(optimizer)
        if (epoch%1)==0:
            print("epoch: {} -- lr:{}".format(epoch, current_lr))
            print("train loss: {}".format(avg_loss))
            print("test loss: {}".format(avg_val_loss))
            print()
            
    #del x_val
    del x_train
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"train Loss: {avg_loss:.4f}")
    print("train average",np.average(y_train.cpu()))
    print("prediction: ",y_pred[0:10])
    print("target: ",y_train[0:10])
    print(f"test Loss: {avg_val_loss:.4f}")
    print(f"learning rate: {optimizer.param_groups[0]["lr"]}")
    history = np.array(history)
    return(history, model)



if __name__=="__main__":
    #FOLDERS/Files:
    xyz_folder="/home/lion/Documents/GNN_Clean/Data/raw_data/xyz_files_relaxed/"
    Energy_File="/home/lion/Documents/GNN_Clean/Data/raw_data/relaxed_Kramer_Energies.txt"
    data_block_location="/home/lion/Documents/GNN_Clean/Data/Dy_Relaxed_Block_Data_Cutoff_12/"
    output_folder_dir = '/home/lion/Documents/GNN_Clean/Results/'
    
    assert os.path.isdir(output_folder_dir)
    
    torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    full_ligands, full_coordinates, full_atom_numbers, full_elements, full_identifiers, element_dictionary, normalised_energies, mean, std=GNN.get_compound_data(Energy_File, xyz_folder)
    
    #Dataset parameters
    filter_number=128
    element_number=len(element_dictionary.keys()) # +1 is for the dummy or ghost atoms  
    run=0#Modifiable
    block_number=16#Modifiable  
    
    
    #Training parameters:
    feature_number=32#Modifiable
    layer_number=2#Modifiable
    LR=0.0025
    epochs=75
    
    train_loader = load_train_dataset(range(block_number),filter_number, data_block_location)
    test_loader = load_test_dataset(range(8),filter_number, data_block_location)
    
    
    history, model=train_SchNet(feature_number, element_number, filter_number, layer_number, epochs, train_loader, test_loader, LR,"mean")

    np.savetxt(output_folder_dir+"history_layer_number_{}_feature_number_{}_block_number_{}_run_{}.txt".format(layer_number, 
                                                                                                                  feature_number,
                                                                                                                  block_number,
                                                                                                                  run),history)
    torch.save(model.state_dict(), output_folder_dir+"model_weights_layer_number_{}_feature_number_{}_block_number_{}_run_{}.pyt".format(layer_number, 
                                                                                                                                         feature_number,
                                                                                                                                         block_number,
                                                                                                                                         run))
    plot_loss_growth(history)
    analyse_model(16,0,model,train_loader, test_loader)
    del(train_loader)
    del(test_loader)
    print("DONE")
