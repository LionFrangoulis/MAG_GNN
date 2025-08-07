import numpy as np

def read_xyz(file_path):
    '''
    Reads in the atomic coordinates and the atomic labels to use as features later in the model. 
    Also stores the connecting atom info, e.g. if it is N or O, and if it is negatively charged.
    '''
    with open(file_path, 'r') as f:
        lines = f.readlines()
        atom_number=int(lines[0]) # number of atoms 
        block_data = []
        elements=[]
        for j in range(len(lines)-2):  # Read the three atom lines
            parts = lines[2 + j].split()
            block_data.append([float(value) for value in parts[1:]])  # Store only the coordinates
            elements.append(parts[0]) # so this is the dummy atom we used instead of Dy 
    '''
    This block is for setting the identifiers and it highly depends on the xyz implementation. 
    Rn it read the line 16 of xyz as it sores the info about the QM9 ligand.
    It needs to be fixed for new uses. 
    '''
    identifier=0#int(lines[1])
    return(np.array(block_data),elements,atom_number, identifier)

def translate_pad_elements(elements, max_atom_number, element_dictionary):
    padded_elements=np.array([ele+["placeholder"]*(max_atom_number-len(ele)) for ele in elements]).flatten()
    elements_translated=np.array([element_dictionary[element] for element in padded_elements]).reshape(-1,max_atom_number)
    return(elements_translated)

def get_compound_data(Energy_file, xyz_location):
    full_elements=[]
    full_atom_numbers=[]
    full_coordinates=[]
    full_identifiers=[]
    
    with open(Energy_file,"r") as f:
        lines=f.readlines()
    
    full_ligands=[line.split()[0] for line in lines]
    energy_data=np.array([float(line.split()[1]) for line in lines])
    normalised_energies=(energy_data-np.mean(energy_data))/np.std(energy_data)
    energy_mean=np.mean(energy_data)
    energy_std=np.std(energy_data)
    
    
    
    
    for ligand in full_ligands:
        c,e,a,i=read_xyz("{}/{}.xyz".format(xyz_location,ligand))
        full_coordinates.append(c)
        full_atom_numbers.append(a)
        full_elements.append(e)
        full_identifiers.append(i)
    
    all_elements=list(set([x for xs in full_elements for x in xs]))
    element_dictionary={"placeholder":0}
    for ele_i in range(len(all_elements)):
        element_dictionary[all_elements[ele_i]] = ele_i+1
        
    return(full_ligands, full_coordinates, full_atom_numbers, full_elements, full_identifiers, element_dictionary, normalised_energies, energy_mean, energy_std)