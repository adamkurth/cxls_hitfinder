import os 
from pkg import u

def main():
    clen_values, photon_energy_values = [1.5, 2.5, 3.5], [6000, 7000, 8000]
    param_matrix = u.parameter_matrix(clen_values, photon_energy_values)
    print(param_matrix, '\n')

    dataset_dict = {
        '01': [clen_values[0], photon_energy_values[0]],
        '02': [clen_values[0], photon_energy_values[1]],
        '03': [clen_values[0], photon_energy_values[2]],
        '04': [clen_values[1], photon_energy_values[0]],
        '05': [clen_values[1], photon_energy_values[1]],
        '06': [clen_values[1], photon_energy_values[2]],
        '07': [clen_values[2], photon_energy_values[0]],
        '08': [clen_values[2], photon_energy_values[1]],
        '09': [clen_values[2], photon_energy_values[2]],
    }
    
    # parameters
    # dataset = '01'
    dataset_number = input("Enter dataset number: ")  # user input
    dataset = dataset_number.zfill(2)  # convert to string with leading zero if necessary
    print(f'Parameter values of dataset {dataset}: {dataset_dict[dataset]}')

    clen, photon_energy = dataset_dict[dataset]
    threshold = 1
    
    # instances
    pm = u.PathManager()
    peak_paths, water_peak_paths, labels, water_background_path = pm.select_dataset(dataset=dataset)
    p = u.Processor(paths=pm, dataset=dataset)
    dm = u.DatasetManager(paths=pm, dataset=dataset, transform=None)

    p.process_directory(dataset=dataset, clen=clen, photon_energy=photon_energy)
    
if __name__ == "__main__":
    main()
    
        
        