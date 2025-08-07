import sys
import math
import time
import numpy as np
import os
import random
from operator import itemgetter
import GNN
import matplotlib.pyplot as plt

def distance(r1,r2):
    return(np.sqrt((r1[0]-r2[0])**2+(r1[1]-r2[1])**2+(r1[2]-r2[2])**2))

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

def distances_to_filters(distances, atom_number, filter_number, rbf_cutoff):
    start=0
    end=rbf_cutoff
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

def spherical_basis_expansion(r1,r2,r3):
    #r1: middle atom
    #r2: first bond end
    #r3: second bond end
    #if np.linalg.norm(r2-r1)<0.1 or np.linalg.norm(r3-r1)<0.1:
    #    print(r1,r2,r3)
    x=np.dot(r1-r2,r1-r3)/(np.linalg.norm(r3-r1)*np.linalg.norm(r2-r1))
    expansion=np.array([x,0.5*(3*x**2-1),0.5*(5*x**3-3*x),1/8*(35*x**4-30*x**2+3),1/8*(63*x**5-70*x**3+15*x), 1/16*(231*x**6-315*x**4+105*x**2-5), 1/16*(429*x**7-315*x**4+315*x**3-35*x)])
    return expansion

def coordinates_to_padded_sphericals(coordinates, atom_number, cutoff, max_angle=6):
    all_sbf=[]
    total_number_angles=0
    empty_sbf=[0]*7*max_angle
    for i,r1 in enumerate(coordinates):#i
        for j,r2 in enumerate(coordinates):#j
            if not i==j:
                distances=[]
                IDs=[]
                for k in range(len(coordinates)):
                    if not k==i and not k==j and not i==j and distance(r1,coordinates[k])<cutoff:
                        distances.append(distance(r1,coordinates[k]))
                        IDs.append(k)
                sorted_IDs=[x for _, x in sorted(zip(distances, IDs))]
                if len(sorted_IDs)>max_angle:
                    sorted_IDs=np.sort(sorted_IDs[:max_angle])
                total_number_angles+=len(sorted_IDs)
                expansion=np.array([spherical_basis_expansion(r1,r2,coordinates[ID]) for ID in sorted_IDs]+[[0]*7]*(max_angle-len(sorted_IDs))).flatten()
                all_sbf.append(expansion)
            else:
                all_sbf.append(empty_sbf)
        for space in range(atom_number-len(coordinates)):
            all_sbf.append(empty_sbf)#spaceholder_i_ghost
    for space1 in range(atom_number-len(coordinates)):
        for space2 in range(atom_number):
            all_sbf.append(empty_sbf)
    all_sbf=np.array(all_sbf).reshape(atom_number, atom_number, max_angle*7)
    return(all_sbf)
    
def coordinates_to_atom_centred_sphericals(coordinates, atom_number, cutoff, max_angle=6):
    all_sbf=[]
    total_number_angles=0
    empty_sbf=[0]*7*(max_angle-1)*max_angle
    for i,r1 in enumerate(coordinates):
        distances=[distance(r1,r2) for j,r2 in enumerate(coordinates) if not j==i and not distance(r1,r2) > cutoff]
        IDs=[j for j,r2 in enumerate(coordinates) if not j==i and not distance(r1,r2) > cutoff]
        sorted_IDs=[x for _, x in sorted(zip(distances, IDs))]
        if len(sorted_IDs)>max_angle:
            sorted_IDs=sorted_IDs[:max_angle]
        expansion=[]
        for ID1 in sorted_IDs:
            for ID2 in sorted_IDs:
                if not ID1==ID2:
                    expansion.append(spherical_basis_expansion(r1,coordinates[ID1],coordinates[ID2]))
        expansion+=[[0]*7]*(max_angle*(max_angle-1)-len(expansion))
        expansion=np.array(expansion).flatten()
        all_sbf.append(expansion)
    for space in range(atom_number-len(coordinates)):
        all_sbf.append(empty_sbf)
    all_sbf=np.array(all_sbf).reshape(atom_number, max_angle*(max_angle-1)*7)
    return(all_sbf)

def coordinates_to_atom_centred_sphericals_no_water(coordinates, elements, atom_number, cutoff, max_angle=7):
    all_sbf=[]
    total_number_angles=0
    empty_sbf=[0]*7*(max_angle-1)*max_angle
    for i,r1 in enumerate(coordinates):
        if elements[i]=="H":
            all_sbf.append(empty_sbf)
        else:
            distances=[distance(r1,r2) for j,r2 in enumerate(coordinates) if not j==i and not distance(r1,r2) > cutoff and not elements[j]=="H"]
            IDs=[j for j,r2 in enumerate(coordinates) if not j==i and not distance(r1,r2) > cutoff and not elements[j]=="H"]
            sorted_IDs=[x for _, x in sorted(zip(distances, IDs))]
            print(len(sorted_IDs))
            if len(sorted_IDs)>max_angle:
                sorted_IDs=sorted_IDs[:max_angle]
            expansion=[]
            for ID1 in sorted_IDs:
                for ID2 in sorted_IDs:
                    if not ID1==ID2:
                        expansion.append(spherical_basis_expansion(r1,coordinates[ID1],coordinates[ID2]))
            expansion+=[[0]*7]*(max_angle*(max_angle-1)-len(expansion))
            expansion=np.array(expansion).flatten()
            all_sbf.append(expansion)
    for space in range(atom_number-len(coordinates)):
        all_sbf.append(empty_sbf)
    all_sbf=np.array(all_sbf).reshape(atom_number, max_angle*(max_angle-1)*7)
    return(all_sbf)
    
    

if __name__=="__main__":
    seed=42
    Energy_file="/home/lion/Documents/GNN_Clean/Data/raw_data/relaxed_Kramer_Energies.txt"
    xyz_location="/home/lion/Documents/GNN_Clean/Data/raw_data/xyz_files_relaxed/"
    filter_location="/home/lion/Documents/GNN_Clean/Data/Dy_Relaxed_Block_Data/"
    sbf_location="/home/lion/Documents/GNN_Clean/Data/Dy_Relaxed_sbf_no_H/"
    
    filter_numbers=[128]
    sbf_cutoff=3
    test_number=2000
    train_block_size=500
    test_block_size=200
    rbf_cutoff=12
    max_angle=7
    
    assert os.path.isfile(Energy_file)
    assert os.path.isdir(xyz_location)
    if not os.path.isdir(filter_location):
        os.mkdir(filter_location)
    if not os.path.isdir(sbf_location):
        os.mkdir(sbf_location)
    
    full_ligands, full_coordinates, full_atom_numbers, full_elements, full_identifiers, element_dictionary, normalised_energies, energy_mean, energy_std=GNN.get_compound_data(Energy_file, xyz_location)
    connectors={}
    with open(xyz_location+"connectors.txt","r") as f:
        lines=f.readlines()
    for line in lines:
        connectors[line.split()[0]]=int(line.split()[1])
    print(element_dictionary)
    compound_number=len(full_ligands)
    max_atom_number=np.max(full_atom_numbers)
    
    
    #input_ID=int(sys.argv[1])
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
    axial_distances=[]
    planar_distances=[]
    for i in range(len(full_coordinates)):
        coords=full_coordinates[i]
        connector=connectors[full_ligands[i]]
        planar_distances.append(distance(coords[0],coords[1]))
        planar_distances.append(distance(coords[0],coords[4]))
        planar_distances.append(distance(coords[0],coords[7]))
        planar_distances.append(distance(coords[0],coords[10]))
        planar_distances.append(distance(coords[0],coords[13]))
        axial_distances.append(distance(coords[0],coords[15+connector]))
    plt.hist(planar_distances)
    plt.show()
    plt.hist(axial_distances)
    plt.show()
    sbf=coordinates_to_atom_centred_sphericals_no_water(full_coordinates[0], full_elements[0], full_atom_numbers[0], sbf_cutoff, max_angle=max_angle)
    
    
    
    starts=list(range(0,compound_number-test_number,train_block_size))
    ends=list(range(train_block_size,compound_number-test_number,train_block_size))
    if len(ends)<len(starts):
        ends.append(compound_number-test_number)
    sorted_train_atom_numbers,sorted_train_IDs = (list(t) for t in zip(*sorted(zip(train_atom_numbers, train_IDs),key=itemgetter(0))))
    RBF=False
    SBF=True
    for counter in range(0):
        print("train", counter)
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
        if SBF:
            start=time.time()
            #sbf=np.array([coordinates_to_atom_centred_sphericals(block_coordinates[i], block_atom_numbers[-1], sbf_cutoff) for i in range(len(block))]).reshape(len(block),block_atom_numbers[-1],7*max_angle*(max_angle-1))
            sbf=np.array([coordinates_to_atom_centred_sphericals_no_water(block_coordinates[i], block_elements[i], block_atom_numbers[-1], sbf_cutoff, max_angle=max_angle) for i in range(len(block))]).reshape(len(block),block_atom_numbers[-1],7*max_angle*(max_angle-1))
            np.save("{}/Train_Block_{}_atomic_sbf_cutoff_{}_max_angle_{}_L_7.npy".format(sbf_location,counter, sbf_cutoff, max_angle), sbf)
            end=time.time()
            print("done sbf {}".format(counter), end-start)
        if RBF:
            distances_Dy = np.array([distances_padding(block_coordinates[i], block_atom_numbers[i], block_atom_numbers[-1]) for i in range(len(block))]).reshape(len(block),-1)
            for filter_number in filter_numbers:
                filters_Dy = distances_to_filters(distances_Dy, block_atom_numbers[-1], filter_number, rbf_cutoff)
                np.save("{}/Train_Block_{}_Filters_number_{}_cutoff_{}.npy".format(filter_location,counter, filter_number, rbf_cutoff),filters_Dy)
                print("done filters {} size {}".format(counter, filters_Dy.shape))
    
    starts=list(range(0,test_number,test_block_size))
    ends=list(range(test_block_size,test_number,test_block_size))
    if len(ends)<len(starts):
        ends.append(test_number)
    sorted_test_atom_numbers,sorted_test_IDs = (list(t) for t in zip(*sorted(zip(test_atom_numbers, test_IDs),key=itemgetter(0))))
    for counter in range(0):
        print("test",counter)
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
        if SBF:
            start=time.time()
            print(block_atom_numbers[-1])
            #sbf=np.array([coordinates_to_padded_sphericals(block_coordinates[i], block_atom_numbers[-1], sbf_cutoff) for i in range(len(block))]).reshape(-1,block_atom_numbers[-1],block_atom_numbers[-1],210)
            #sbf=np.array([coordinates_to_atom_centred_sphericals(block_coordinates[i], block_atom_numbers[-1], sbf_cutoff) for i in range(len(block))]).reshape(len(block),block_atom_numbers[-1],7*max_angle*(max_angle-1))
            sbf=np.array([coordinates_to_atom_centred_sphericals_no_water(block_coordinates[i], block_elements[i], block_atom_numbers[-1], sbf_cutoff, max_angle=max_angle) for i in range(len(block))]).reshape(len(block),block_atom_numbers[-1],7*max_angle*(max_angle-1))
            print(sbf.shape)
            np.save("{}/Test_Block_{}_atomic_sbf_cutoff_{}_max_angle_{}_L_7.npy".format(sbf_location,counter, sbf_cutoff, max_angle), sbf)
            end=time.time()
            print("done sbf {}".format(counter), end-start)
        if RBF:
            distances_Dy = np.array([distances_padding(block_coordinates[i], block_atom_numbers[i], block_atom_numbers[-1]) for i in range(len(block))]).reshape(len(block),-1)
            for filter_number in filter_numbers:
                filters_Dy = distances_to_filters(distances_Dy, block_atom_numbers[-1], filter_number, rbf_cutoff)
                np.save("{}/Test_Block_{}_Filters_number_{}_cutoff_{}.npy".format(filter_location,counter, filter_number, rbf_cutoff),filters_Dy)
                print("done filters {} size {}".format(counter, filters_Dy.shape))