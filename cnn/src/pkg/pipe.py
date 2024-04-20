import torch
import torch.nn as nn
import torch.optim as optim
# from pkg import *
from pkg.waterbackground_subtraction.finder import background
import numpy as np

class ModelPipeline:
    def __init__(self, peak_conf: nn.Module, energy_conf: nn.Module, clen_conf: nn.Module, device) -> None:
        """
        This class represents a pipeline for analyzing Bragg peak images.
        It combines three models for peak detection, energy estimation, and clen calculation.

        Args:
            peak_model (nn.Module): Convolutional neural network for peak detection.
            energy_model (nn.Module): Convolutional neural network for energy estimation. This model is used when a peak is detected.
            clen_model (nn.Module): Convolutional neural network for clen calculation. This model is used when a peak is detected after energy_model.
        """
        
        self.peak_conf = peak_conf
        self.energy_conf = energy_conf
        self.clen_conf = clen_conf
        
        peak_model_path = peak_conf.get_save_path()
        energy_model_path = energy_conf.get_save_path()
        clen_model_path = clen_conf.get_save_path() 
        
        peak_state_dict = torch.load(peak_model_path)
        energy_state_dict = torch.load(energy_model_path)
        clen_state_dict = torch.load(clen_model_path)

        peak_model = peak_conf.get_model()
        energy_model = energy_conf.get_model()
        clen_model = clen_conf.get_model()

        peak_model.load_state_dict(peak_state_dict)
        energy_model.load_state_dict(energy_state_dict)
        clen_model.load_state_dict(clen_state_dict)

        self.peak_model = peak_model.eval() 
        self.energy_model = energy_model.eval()
        self.clen_model = clen_model.eval()
        
        self.peak_model.to(device)
        self.energy_model.to(device)
        self.clen_model.to(device)
        
        self.water_background_subtraction = background.BackgroundSubtraction(threshold=20)

        self.pipeline_results = {
            'photon_energy': None,
            'clen': None
        }
        
        self.attributes = (0, 0)  
        self.atributes = (0,0)

    def run(self, image: torch.tensor) -> tuple:
        """ 
        This function runs the analysis pipeline on a given image.

        Args:
            image (torch.Tensor): This file shcxls_hitfinder/images/peaks/01/img_6keV_clen01_00062.h5ould be a 2D tensor representing the image of the Bragg peak .h5 file. 

        Returns:
            tuple: This function returns the estimated x-ray energy and clen value if a peak is detected, otherwise None.
        """
        
        with torch.no_grad():  
            peak_detected = (torch.sigmoid(self.peak_model(image)) > 0.5).long()
              
            if peak_detected == 1:
                _, x_ray_energy = torch.max(self.energy_model(image),1)
                x_ray_energy = x_ray_energy * 1000 + 5000       
                 
                _, clen = torch.max(self.clen_model(image),1)
                clen = clen * 0.1 + 0.05
                
                self.pipeline_results['photon_energy'] = x_ray_energy
                self.pipeline_results['clen'] = clen
                
                dataframe = self.water_background_subtraction.main(image)
                
                print(dataframe)
                
                # literally does not work
                self.water_background_subtraction.visualize_peaks(image, dataframe) 
                
                return self.pipeline_results
            else:
                return None 

    # def evaluate_results(self, image_path: str) -> None:
    #     """
    #     This function compares the pipeline results with the true attributes of the image.

    #     Args:
    #         image_path (str): This is the path to the .h5 image that was used in run_pipeline.

    #     Returns:
    #         str: The message telling us if the atributes are matching or not.
    #     """
        
    #     clen, photon_energy = f.retrieve_attributes(image_path)
    #     self.atributes = (clen, photon_energy)
        
    #     if self.pipeline_results == self.atributes:
    #         print("The pipeline results match the true attributes.")
    #     else:
    #         print("The pipeline results do not match the true attributes.")