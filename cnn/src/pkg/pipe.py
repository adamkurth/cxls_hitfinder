import torch
import torch.nn as nn
import torch.optim as optim
from pkg import *
import numpy as np

class ModelPipeline:
    def __init__(self, peak_model_path: str, energy_model_path: str, clen_model_path: str) -> None:
        """
        This class represents a pipeline for analyzing Bragg peak images.
        It combines three models for peak detection, energy estimation, and clen calculation.

        Args:
            peak_model (nn.Module): Convolutional neural network for peak detection.
            energy_model (nn.Module): Convolutional neural network for energy estimation. This model is used when a peak is detected.
            clen_model (nn.Module): Convolutional neural network for clen calculation. This model is used when a peak is detected after energy_model.
        """
        
        self.binary_model = torch.load(peak_model_path)
        self.energy_model = torch.load(energy_model_path)
        self.clen_model = torch.load(clen_model_path)
        
        self.binary_model.eval()
        self.energy_model.eval()
        self.clen_model.eval()
        
        self.pipeline_results = (0,0)
        self.atributes = (0,0)

    def run_pipeline(self, image: torch.tensor) -> tuple:
        """ 
        This function runs the analysis pipeline on a given image.

        Args:
            image (torch.Tensor): This file shcxls_hitfinder/images/peaks/01/img_6keV_clen01_00062.h5ould be a 2D tensor representing the image of the Bragg peak .h5 file. 

        Returns:
            tuple: This function returns the estimated x-ray energy and clen value if a peak is detected, otherwise None.
        """
        
        with torch.no_grad():  
            peak_detected = self.binary_model(image).argmax(dim=1).item() == 1
              
            if peak_detected:
                x_ray_energy = self.energy_model(image).item()
                clen = self.clen_model(image).item()
                
                self.pipeline_results = (x_ray_energy, clen)
                return self.pipeline_results
            else:
                return None 

    def compare_results(self, image_path: str) -> None:
        """
        This function compares the pipeline results with the true attributes of the image.

        Args:
            image_path (str): This is the path to the .h5 image that was used in run_pipeline.

        Returns:
            str: The message telling us if the atributes are matching or not.
        """
        
        clen, photon_energy = f.retrieve_attributes(image_path)
        self.atributes = (clen, photon_energy)
        
        if self.pipeline_results == self.atributes:
            print("The pipeline results match the true attributes.")
        else:
            print("The pipeline results do not match the true attributes.")