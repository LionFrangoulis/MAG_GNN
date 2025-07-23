import numpy as np
import os
import random
from operator import itemgetter
import GNN

def distances_padding(coordinate_data, atom_number, max_atom_number):
    dummy=100
    
    if max_atom_number==-1:
        max_atom_number=atom_number
    assert max_atom_number >= atom_number
    coordinate_data=np.reshape(coordinate_data,(-1,atom_number,3))
    distances=[]
    for counter, atom_coords in enumerate(coordinate_data):
        disto=[]
        for i in range(atom_number):
            atom_distances=[]
            for j in range(atom_number):
                coord1=atom_coords[i]
                coord2=atom_coords[j]
                dist=np.sqrt((coord2[0]-coord1[0])**2+(coord2[1]-coord1[1])**2+(coord2[2]-coord1[2])**2)
                atom_distances.append(dist)
            atom_distances+=[dummy]*(max_atom_number-atom_number)
            disto.append(atom_distances)
        for k in range(max_atom_number-atom_number):
            disto.append([dummy]*max_atom_number)
        distances.append(disto)
    distances=np.reshape(np.array(distances,dtype=np.float32),(-1,max_atom_number**2))
    return distances

def distances_to_filters(distances, atom_number, filter_number):
    start=0
    end=3.3
    steps=filter_number
    gamma=10
    stepsize=(end-start)/steps
    
    distances=np.reshape(distances,(-1))
    filter_distances=np.arange(start,end,stepsize)
    all_filters=[]
    i=0
    for dist in distances:
        if i%int(len(distances)/20)==0:
            print(i/int(len(distances)/20))
        if dist>(end+2):
            filters=[0]*steps
        else:
            filters=[np.exp(-gamma*(dist-filter_distance)**2) for filter_distance in filter_distances]
        all_filters.append(filters)
        i+=1
    all_filters=np.reshape(np.array(all_filters,dtype=np.float32),(-1,atom_number,atom_number,steps))
    return all_filters

if __name__=="__main__":
    seed=42
    Energy_file="./Dy_prediction/relaxed_Kramer_Energies.txt"
    xyz_location="./Dy_prediction/xyz_files_relaxed/"
    filter_location="./Dy_Relaxed_Block_Data_Small/"
    
    filter_numbers=[64]
    test_number=2000
    train_block_size=500
    test_block_size=200
    assert os.path.isfile(Energy_file)
    assert os.path.isdir(xyz_location)
    if not os.path.isdir(filter_location):
        os.mkdir(filter_location)
    
    full_ligands, full_coordinates, full_atom_numbers, full_elements, full_identifiers, element_dictionary, normalised_energies, energy_mean, energy_std=GNN.get_compound_data(Energy_file, xyz_location)
    
    compound_number=len(full_ligands)
    max_atom_number=np.max(full_atom_numbers)
    
    
        
    random.seed(seed)
    test_IDs=random.sample(list(range(compound_number)),test_number)
    train_IDs=list(range(compound_number))
    for ID in test_IDs:
        train_IDs.remove(ID)
    
    test_coordinates=[full_coordinates[ID] for ID in test_IDs]
    test_ligands=[full_ligands[ID] for ID in test_IDs]
    test_atom_numbers=[full_atom_numbers[ID] for ID in test_IDs]
    test_elements=[full_elements[ID] for ID in test_IDs]
    test_energies=[normalised_energies[ID] for ID in test_IDs]
    test_identifiers=[full_identifiers[ID] for ID in test_IDs]
    
    train_coordinates=[full_coordinates[ID] for ID in train_IDs]
    train_ligands=[full_ligands[ID] for ID in train_IDs]
    train_atom_numbers=[full_atom_numbers[ID] for ID in train_IDs]
    train_elements=[full_elements[ID] for ID in train_IDs]
    train_energies=[normalised_energies[ID] for ID in train_IDs]
    train_identifiers=[full_identifiers[ID] for ID in train_IDs]
    
    
    starts=list(range(0,compound_number-test_number,train_block_size))
    ends=list(range(train_block_size,compound_number-test_number,train_block_size))
    if len(ends)<len(starts):
        ends.append(compound_number-1-test_number)
    sorted_train_atom_numbers,sorted_train_IDs = (list(t) for t in zip(*sorted(zip(train_atom_numbers, train_IDs),key=itemgetter(0))))
    for counter in range(len(ends)):
        block=sorted_train_IDs[starts[counter]:ends[counter]]
        block_coordinates=[full_coordinates[ID] for ID in block]
        block_ligands=[full_ligands[ID] for ID in block]
        block_atom_numbers=[full_atom_numbers[ID] for ID in block]
        block_elements=[full_elements[ID] for ID in block]
        block_energies=[normalised_energies[ID] for ID in block]
        block_identifiers=[full_identifiers[ID] for ID in block]
        block_translated_elements=GNN.translate_pad_elements(block_elements, block_atom_numbers[-1], element_dictionary)
        np.savetxt("{}/Train_Block_{}_ligands.txt".format(filter_location,counter),block_ligands, fmt="%s")
        np.savetxt("{}/Train_Block_{}_energies.txt".format(filter_location,counter),block_energies)
        np.savetxt("{}/Train_Block_{}_atom_numbers.txt".format(filter_location,counter),block_atom_numbers, fmt="%i")
        np.savetxt("{}/Train_Block_{}_identifiers.txt".format(filter_location,counter),block_identifiers, fmt="%i")
        np.save("{}/Train_Block_{}_elements.npy".format(filter_location,counter),block_translated_elements)
        distances_Dy = np.array([distances_padding(block_coordinates[i], block_atom_numbers[i], block_atom_numbers[-1]) for i in range(len(block))]).reshape(len(block),-1)
        for filter_number in filter_numbers:
            filters_Dy = distances_to_filters(distances_Dy, block_atom_numbers[-1], filter_number)
            np.save("{}/Train_Block_{}_Filters_number_{}.npy".format(filter_location,counter, filter_number),filters_Dy)
            print("done filters {} size {}".format(counter, filters_Dy.shape))
    
    starts=list(range(0,test_number,test_block_size))
    ends=list(range(test_block_size,test_number,test_block_size))
    if len(ends)<len(starts):
        ends.append(test_number-1)
    sorted_test_atom_numbers,sorted_test_IDs = (list(t) for t in zip(*sorted(zip(test_atom_numbers, test_IDs),key=itemgetter(0))))
    for counter in range(len(ends)):
        block=sorted_test_IDs[starts[counter]:ends[counter]]
        block_coordinates=[full_coordinates[ID] for ID in block]
        block_ligands=[full_ligands[ID] for ID in block]
        block_atom_numbers=[full_atom_numbers[ID] for ID in block]
        block_elements=[full_elements[ID] for ID in block]
        block_energies=[normalised_energies[ID] for ID in block]
        block_identifiers=[full_identifiers[ID] for ID in block]
        block_translated_elements=GNN.translate_pad_elements(block_elements, block_atom_numbers[-1], element_dictionary)
        np.savetxt("{}/Test_Block_{}_ligands.txt".format(filter_location,counter),block_ligands, fmt="%s")
        np.savetxt("{}/Test_Block_{}_energies.txt".format(filter_location,counter),block_energies)
        np.savetxt("{}/Test_Block_{}_atom_numbers.txt".format(filter_location,counter),block_atom_numbers, fmt="%i")
        np.savetxt("{}/Test_Block_{}_identifiers.txt".format(filter_location,counter),block_identifiers, fmt="%i")
        np.save("{}/Test_Block_{}_elements.npy".format(filter_location,counter),block_translated_elements)
        distances_Dy = np.array([distances_padding(block_coordinates[i], block_atom_numbers[i], block_atom_numbers[-1]) for i in range(len(block))]).reshape(len(block),-1)
        for filter_number in filter_numbers:
            filters_Dy = distances_to_filters(distances_Dy, block_atom_numbers[-1], filter_number)
            np.save("{}/Test_Block_{}_Filters_number_{}.npy".format(filter_location,counter, filter_number),filters_Dy)
            print("done filters {} size {}".format(counter, filters_Dy.shape))