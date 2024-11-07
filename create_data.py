
import core.IPW as IPW
import core.SEMs as SEMs
import core.global_params as global_params
import core.utile as utile
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import copy
from tqdm import tqdm as tqdm
import h5py
import json

seed = global_params.SEED
np.random.seed(seed)

# create path to folder of this script, where the data will be saved.
import os
path = os.path.join("data","samplesize")

BATCH_SIZE = 700 #dataset size for estimation
ASYMPTOTIC_BATCH_SIZE = 10**4 # Number of simulations to calculate true mean under intervention

#Define Nodes and their Structural Equations
speed = SEMs.node(name = "Speed",
                  SE = SEMs.identity_SE(),
                  intervenable = True,
                  measurable = True,
                  is_treatment = False,
                  is_target = False
                  )


class chip_flow_SE(SEMs.StructuralEquation):
    def __init__(self, coefficients_dict,incorporate_wear:bool = False):
        """
        :param incorporate_wear: boolean, whether to incorporate wear into the structural equation. By default False.
        Note that this requires an arrow from Wear to Chip Flow.
        :param coefficients_dict: dictionary with the coefficients of the structural equation. 
        May contain "d". Must contain "xi" if incorporate_wear is True.
        The equation is Speed*FeedRate*d*(Noise U_Q)(1.0-xi*Wear) if incorporate_wear is True, else Speed*FeedRate*d*(Noise U_Q)
        """
        self.d = 1.0
        if "d" in coefficients_dict.keys():
            self.d = coefficients_dict["d"]
        self.incorporate_wear = incorporate_wear
        if incorporate_wear:
            self.xi = coefficients_dict["xi"]
        
    def __call__(self, input_dict):
        Noise = input_dict["U"]
        assert np.all(Noise>=0)
        Speed = input_dict["Speed"]
        FeedRate = input_dict["FeedRate"]
        if self.incorporate_wear:
            Wear = input_dict["Wear"]
            result = Speed*FeedRate*self.d*(1.0-self.xi*Wear)*(Noise)
        else:
            result = Speed*FeedRate*self.d*(Noise)
        return result

class cutting_force_SE(SEMs.StructuralEquation):
    """
    The equation is K0*Speed**{-m}*FeedRate*d*(Noise U_F)(1.0+beta*Wear)
    """
    def __init__(self, coefficients_dict):
        """
        :param coefficients_dict: dictionary with the coefficients of the structural equation.
        Must contain "K0", "d", "m", "beta".
        """
        self.coeff_names = ["K0","d","m","beta"]
        self.K0 = coefficients_dict["K0"]
        self.d = coefficients_dict["d"]
        self.m = coefficients_dict["m"]
        self.beta = coefficients_dict["beta"]

        
    def __call__(self, input_dict):
        
        Noise = input_dict["U"] #BUG noticed here. Work in progress. This code itself is fine.
        assert np.all(Noise>=0)
        Speed = input_dict["Speed"]
        FeedRate = input_dict["FeedRate"]
        Wear = input_dict["Wear"]
        result = self.K0*Speed**(-self.m)*FeedRate*self.d*(Noise)*(1.0+self.beta*Wear)
        return result

class Drift_SE(SEMs.SE_with_coefficients):
    """
    Structural Equation of the form Y=(aQ+bF)(d U)
    Very similar to linear_inhomogeneous_SE, just without the +1 so its explicit.
    Expect U to be half-normal here, or sqrt chisq(2).
    """   
    def __init__(self, coefficients_dict, number_incoming_edges:int):
        correspondence  = {"a_chip":"ChipFlow","a_force":"CuttingForce"}
        modified_coefficients_dict = {correspondence[key]:value for key,value in coefficients_dict.items()}
        super().__init__(modified_coefficients_dict, number_incoming_edges)
        self.coeff_names = ["ChipFlow","CuttingForce"] #believe this is actually not used.

    def __call__(self, input_dict:dict)->np.array:
        random_variables, exogeneous_U = super().__call__(input_dict)
        return random_variables@self.coefficients*(self.coefficient_for_U*exogeneous_U)

import numpy as np
import itertools

class Hyperparameters:
    """
    An iterable class that returns the next hyperparameter combination for a given set of hyperparameters, in form of a dictionary.
    :param dict_of_grids: dictionary of numpy arrays, where the keys are the hyperparameter names and the values are the numpy vectors of values they can take.
    :param n_values_per_parameter: int, the number of values each hyperparameter takes if the used does not specify that parameter. Only used if grid_search = True.
    :grid_search: bool, whether to create all possible combinations of hyperparameters (default). If False, the user has to the combinations. Parameters not specified by the user will be set to the default (middle of the range) value.
    Specifying the combos is done by supplying a vector of values for each hyperparameter (in dict_of_grids), of the same length (repetition possible). The first combo is then the first value from every vector, and so on.
    """
    def __init__(self, dict_of_grids=None, n_values_per_parameter=1, grid_search = True):
        self.verbose = True
        self.keys = [ "d", "xi", "K0", "m", "beta", "sigma_force", "sigma_chip", "a_chip", "a_force","sigma_drift"]
        self.grid = {key: None for key in self.keys}
        
        self.ranges = { # Used to automatically generate values if not supplied by user.
            "d": (5, 5), #mm
            "xi": (0.0, 0.5),
            "K0": (1000, 2000), # N/mm^2
            "m": (0.3, 0.3), 
            "beta": (0.0, 0.1),
            "sigma_force": (0.1, 0.5),
            "sigma_chip": (0.1,0.5),
            "a_chip": (0.0005,0.001),
            "a_force": (0.001,0.005),
            "sigma_drift": (1.0, 1.0)
        }

        # Hyperparameters that are relevant to the structural equation of each node, used in calculate_all_data
        self.node_hyperparameters = {
            "ChipFlow": ["d","xi","sigma_chip"],
            "CuttingForce": ["sigma_force","K0","m","beta","d"],
            "Drift": ["a_chip","a_force","sigma_drift"]
        }
        
        # Whether the hyperparameter is structural (parameter in the SE) 
        #or whether it belongs to the noise 
        #(which is supplied by the user and hence not explicitly encoded in the SE)
        self.is_structural = {
            "d": True,
            "xi": True,
            "sigma_chip": False,
            "sigma_force": False,
            "sigma_drift": False,
            "K0": True,
            "m": True,
            "beta": True,
            "a_chip": True,
            "a_force": True
        }

        if grid_search:
            # Create the grid, either supplied by the user or linearly spaced among the ranges
            for key in self.keys:
                if dict_of_grids is not None and key in dict_of_grids:
                    assert isinstance(dict_of_grids[key], np.ndarray), "Grid should be a numpy array"
                    assert dict_of_grids[key].size > 0, "Grid cannot be empty"
                    self.grid[key] = dict_of_grids[key]
                else:
                    if key == "sigma_drift": # Special case for sigma_drift
                        self.grid[key] = np.array([1.0])
                    elif self.verbose:
                        print(f"Creating grid for {key} with {n_values_per_parameter} values between {self.ranges[key][0]} and {self.ranges[key][1]}.")
                    self.grid[key] = np.linspace(self.ranges[key][0], self.ranges[key][1], n_values_per_parameter)

            # Prepare the Cartesian product of hyperparameters
            self.combinations = list(itertools.product(*[self.grid[key] for key in self.keys]))
        
        else:
            # The user supplies the combinations, by supplying vectors of same length for each hyperparameter.
            #Check that the supplied values are reasonable
            length = None
            assert len(dict_of_grids) >0, "dict_of_grids cannot be empty."

            for key_value in dict_of_grids.items():
                assert isinstance(key_value[1], np.ndarray), "Grid should be a numpy array"
                assert key_value[1].size > 0, "Grid cannot be empty"
                if not length is None:
                    assert length == key_value[1].size, "All grids should have the same length."
                length = key_value[1].size
                self.grid[key_value[0]] = key_value[1]
            
            # If the user didn't supply everything, set rest to default.
            for key in self.keys:
                if not key in dict_of_grids:
                    print(f"Warning: Key {key} not supplied in dict_of_grids by user. Using default value of {key} = {self.default(key)}.")
                    self.grid[key] = np.full(shape= (length,), fill_value=self.default(key))
                
            # Create the combinations, taking the ith element from each vector.
            list_of_vectors = [self.grid[key] for key in self.keys]
            self.combinations = list(zip(*list_of_vectors))
              
        self.iteration = 0
        
    def default(self, key):
        return (self.ranges[key][0]+self.ranges[key][1])/2
        
    def hyperparameter_key(self, hyperparameter_dict):
        """
        In the hdf5 file hp key, the order of the hyperparameters matters. The idea is that the user can give a dict with one value per hyperparameter.
        This function will take care of ordering it correctly (by creating a new dict in the right order and calling utile.dict_to_str)
        It will also check that the hyperparameters all have valid values, by checking whether the combination is in self.combinations.
        """
        new_dict = {key:hyperparameter_dict[key] for key in self.keys}
        assert tuple(new_dict.values()) in self.combinations, "Invalid hyperparameter combination."
        return utile.dict_as_string(new_dict)

    def __next__(self):
        if self.iteration >= len(self.combinations):
            raise StopIteration("No more hyperparameter combinations available.")
        current_combination = self.combinations[self.iteration]
        self.iteration += 1
        return dict(zip(self.keys, current_combination))

    def reset(self):
        self.iteration = 0

    def __iter__(self):
        return self
    def __len__(self):
        return len(self.combinations)
    
"""
# Usage example
hyperparams = Hyperparameters(n_values_per_parameter=5)
for combo in hyperparams:
    print(combo)
"""


chip_flow = SEMs.node(name = "ChipFlow",
                      SE = None,
                      intervenable=False,
                      measurable = False,
                      is_treatment = False,
                      is_target = False
                      )

wear = SEMs.node(name = "Wear",
                 SE = None,
                 intervenable = False,
                 measurable = True,
                 is_treatment = False,
                 is_target = False
                 )

cutting_force = SEMs.node(name = "CuttingForce",
                          SE = None,
                          intervenable = False,
                          measurable = True,
                          is_treatment = True,
                          is_target = False
                          )

feed_rate = SEMs.node(  name = "FeedRate",
                        SE = None,
                        intervenable = True,
                        measurable = True,
                        is_treatment = True,
                        is_target = False
                        )

drift = SEMs.node(  name = "Drift",
                    SE = None,
                    intervenable = False,
                    measurable = True,
                    is_treatment = False,
                    is_target = True
                    )



# Create the edge matrix
vertices=[speed, chip_flow, wear, cutting_force, feed_rate, drift]
edges=[("Speed","ChipFlow"),
       ("Speed","CuttingForce"),
       ("CuttingForce","Drift"),
       ("FeedRate","CuttingForce"),
       ("FeedRate","ChipFlow"),
       ("ChipFlow","Drift"),
       ("Wear","CuttingForce")]
edge_matrix = utile.create_edge_matrix(vertices, edges)

# Create the graph
M=SEMs.graph(name = "ExperimentGraph",
             vertices = vertices,
             edge_matrix = edge_matrix,
             verbose = False)

def get_dict_of_us(batch_size,sigma_force = 1.0,sigma_chip = 1.0,sigma_drift = 1.0):
    #For Simulation, manually set the Us you want to set
    U_SPEED = np.random.uniform(low=2000,high=3000,size=batch_size).astype(np.float64)
    #150 m/min is 2500 mm/s
    #20-60 m/min #TODO
    U_DRIFT = np.abs(np.random.normal(loc=0.0,scale = sigma_drift,size=batch_size).astype(np.float64)) #half normal
    #TODO: make log-normal
    U_FEED_RATE = np.random.uniform(low=0.05,high=0.3,size=batch_size).astype(np.float64)
    #0.2 mm/rev to 0.4 mm/rev
    U_WEAR = np.random.uniform(low=0.0,high=0.2,size=batch_size).astype(np.float64)
    #0.0 mm to 0.2 mm

    U_CUTTING_FORCE = np.exp(np.random.normal(size=batch_size,loc=0.0,scale=sigma_force).astype(np.float64)) #lognormal
    U_CHIP_FLOW = np.exp(np.random.normal(size=batch_size,loc=0.0,scale=sigma_chip).astype(np.float64)) #lognormal
    return {"Speed": U_SPEED,
              "FeedRate": U_FEED_RATE,
              "Drift": U_DRIFT,
                "Wear":U_WEAR,
                "CuttingForce": U_CUTTING_FORCE,
                "ChipFlow": U_CHIP_FLOW}


def calculate_all_data(hp:Hyperparameters,treatment_name,B, grid_of_intervention_values, grid_of_sample_sizes,filename):
    """
    ...
    """
    size_milestones=8**np.arange(3,8)/1024 #from 0.5MB to 2GB, unit in megabytes
    milestone_step=0

    loop_counter = 0
    # Define a single HDF5 file for all batches
    hdf5_file = os.path.join(path, filename) 
    # Store metadata as attributes in the HDF5 file

    #Check whether previous data exists
    if os.path.exists(hdf5_file):
        print(f"Caution, file {hdf5_file} already exists. Proceeding will overwrite the file.")
        proceed = input("Do you want to proceed? (y/n)")
        if proceed.lower() != "y":
            print("Terminating the function calculate_all_data.")
            return None
        print("Proceeding to overwrite the file.")

    with h5py.File(hdf5_file, "w") as f: #this deletes previous data
        f.attrs["B"] = B
        f.attrs["grid_of_sample_sizes"] = grid_of_sample_sizes
        if not grid_of_intervention_values is None:
            f.attrs["grid_of_intervention_values"] = grid_of_intervention_values
        f.attrs["hyperparameter_names"] = hp.keys # list of hp names
        for key in hp.keys:
            assert key != "B", "Key 'B' is reserved for the number of batches"
            assert key not in f.attrs, f"Key {key} already exists as metadata in the HDF5 file"
            f.attrs[key] = hp.grid[key] 

    for hyperparameters in tqdm(hp,desc="Hyperparameters "): #hyperparameters is a dictionary
        
        # Encode the hyperparameters
        hyperparameter_key = utile.dict_as_string(hyperparameters)

        # Create the graph
        M=SEMs.graph(name = "ExperimentGraph",
                vertices = vertices,
                edge_matrix = edge_matrix,
                verbose = False)
        
        # Set the SE of all vertices
        for vertex in M.vertices:

            number_incoming_edges = vertex.number_incoming_edges
            #Update the SE using the hyperparameters
            dict_node_to_parameters = hp.node_hyperparameters
            if vertex.name in ["Speed","Wear","FeedRate"]:
                vertex.SE = SEMs.identity_SE()
            else:
                # get the subdict of hyperparameters that are relevant to the structural equation of this vertex
                coefficients_dict = {hp_name:hyperparameters[hp_name] for hp_name in dict_node_to_parameters[vertex.name] if hp.is_structural[hp_name]}
                if vertex.name == "ChipFlow":
                    vertex.SE = chip_flow_SE(coefficients_dict = coefficients_dict)
                elif vertex.name == "CuttingForce":
                    vertex.SE = cutting_force_SE(coefficients_dict = coefficients_dict)
                elif vertex.name == "Drift":
                    vertex.SE = Drift_SE(coefficients_dict = coefficients_dict, number_incoming_edges = number_incoming_edges)
                else:
                    raise ValueError(f"Vertex {vertex.name} not recognized.")
        

        #Set reasonable treatment values for the intervention in grid_of_intervention_values if it hasn't been set by the user.
        #This is a useful option because it is difficult to know what values are in the natural range the first time you define the hyperparameters, when you have no data to look at.
        # The idea is to simulate from the graph and then take reasonable quantiles of the treatment variable. This guess is made with the first combo of hyperparameters.
        
        if grid_of_intervention_values is None and loop_counter == 0:
            print("The user did not specify grid_of_intervention_variables.")
            lwr = 0.05
            upr = 0.95
            results = M.simulate(dict_of_Us = get_dict_of_us(ASYMPTOTIC_BATCH_SIZE, sigma_chip = hyperparameters["sigma_chip"], sigma_force = hyperparameters["sigma_force"], sigma_drift=hyperparameters["sigma_drift"]),
                                 batch_size=ASYMPTOTIC_BATCH_SIZE)
            treatment_values=results["endogeneous"][treatment_name]
            lwr_quantile = np.quantile(treatment_values,lwr)
            upr_quantile = np.quantile(treatment_values,upr)
            grid_of_intervention_values = np.linspace(lwr_quantile,upr_quantile,100)
            print(f"A grid has been automatically set for the intervention values. The variable {treatment_name} takes {len(grid_of_intervention_values)} values from {lwr_quantile} to {upr_quantile}.")
            with h5py.File(hdf5_file, "a") as f:
                f.attrs["grid_of_intervention_values"] = grid_of_intervention_values

        #Calculate true counterfactual means
        true_counterfactual_means = M.do(target_variable_name = "Drift",
            intervention_variable_name = treatment_name,
            grid_of_intervention_values = grid_of_intervention_values,
            dict_of_Us = get_dict_of_us(ASYMPTOTIC_BATCH_SIZE,
                                        sigma_chip=hyperparameters["sigma_chip"],
                                        sigma_force=hyperparameters["sigma_force"],
                                        sigma_drift=hyperparameters["sigma_drift"]),
                                        return_all_data = False)
        
        #Store the true counterfactual means for those hyperparameters (may depend on hyperparameters depending on the system)
        with h5py.File(hdf5_file, "a") as f:
            hp_group = f.create_group(hyperparameter_key)
            hp_group.attrs["hyperparameters"] = json.dumps(hyperparameters)
            f.create_dataset(f"{hyperparameter_key}/true_counterfactual_means", data=true_counterfactual_means)
        
        # Calculate the B batches of data for each sample size
        for n in grid_of_sample_sizes:
            
            #initialize lists to store batches of data
            endogeneous_data = []
            exogeneous_data = []

            # Calculate the B batches of data for a given sample size
            for b in range(B):
                dict_of_Us = get_dict_of_us(n, sigma_chip = hyperparameters["sigma_chip"], sigma_force = hyperparameters["sigma_force"],sigma_drift=hyperparameters["sigma_drift"])
                results = M.simulate(dict_of_Us = dict_of_Us, batch_size = n)
                endogeneous_df = results["endogeneous"]
                exogeneous_df = results["exogeneous"]

                endogeneous_df["batchnr"] = b
                exogeneous_df["batchnr"] = b
                endogeneous_data.append(endogeneous_df)
                exogeneous_data.append(exogeneous_df)
                
                loop_counter += 1
            
            # Store the batches of data for this sample size
            endogeneous_batch = pd.concat(endogeneous_data)
            exogeneous_batch = pd.concat(exogeneous_data)
            endogeneous_batch.to_hdf(hdf5_file, key=f"{hyperparameter_key}/n_{n}/endogeneous", mode='a')
            exogeneous_batch.to_hdf(hdf5_file, key=f"{hyperparameter_key}/n_{n}/exogeneous", mode='a')

            # Track file size, this can get quite big.
            file_size = os.path.getsize(hdf5_file)
            if milestone_step >= len(size_milestones):
                    Warning(f"File size {file_size} very large, greater than", size_milestones[-1], "Megabytes.")
                    next_milestone = size_milestones[-1]
            elif file_size/(1024**2) > size_milestones[milestone_step]:
                print("We hit the milestone", milestone_step, "of ", size_milestones[milestone_step], "Megabytes.")
                print(f"File size: {file_size} bytes, {file_size/(1024**2)} Megabytes.")
                milestone_step += 1
                
    file_size = os.path.getsize(hdf5_file)
    print(f"Final file size: {file_size} bytes, {file_size/(1024**2)} Megabytes.")





# Experiment DATASET
sigmas = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0])
a_force = np.array([0.01,0.03])
a_force = np.repeat(a_force,repeats = len(sigmas))
sigmas = np.tile(sigmas,reps = 2)

dict_of_grids = {"sigma_chip":sigmas,"sigma_force":sigmas,
                 "a_chip":np.full_like(a=sigmas,fill_value = 0.005),
                 "a_force":a_force,
                 "sigma_drift":np.full_like(sigmas,1.0)}


HYPERPARAMETERS = Hyperparameters(dict_of_grids = dict_of_grids,
                                   n_values_per_parameter = 1,
                                   grid_search = False)
B = 100
grid_of_intervention_values = np.linspace(100,400,100)
grid_of_sample_sizes = 25*2**np.arange(9) # 25, 50, 100, 200, 400, 800, 1600, 3200, 6400

"""
calculate_all_data(hp = HYPERPARAMETERS,
                   treatment_name = "CuttingForce",
                   B = B, 
                   grid_of_intervention_values = None, 
                   grid_of_sample_sizes = grid_of_sample_sizes, 
                   filename = "experiment_datav03_linear.h5")
"""

# TOY DATASET

sigmas = np.array([0.25])
dict_of_grids = {"sigma_chip":sigmas,"sigma_force":sigmas,
                 "a_chip":np.full_like(a=sigmas,fill_value = 0.005),
                 "a_force":np.full_like(sigmas,0.02),
                 "sigma_drift":np.full_like(sigmas,1.0)}


HYPERPARAMETERS = Hyperparameters(dict_of_grids = dict_of_grids,
                                   n_values_per_parameter = 1,
                                   grid_search = False)
B = 10
grid_of_intervention_values = np.linspace(90,200,100)
grid_of_sample_sizes = 25*2**np.arange(9) # 25, 50, 100, 200, 400, 800, 1600, 3200, 6400


calculate_all_data(hp = HYPERPARAMETERS,
                   treatment_name = "CuttingForce",
                   B = B, 
                   grid_of_intervention_values = None, 
                   grid_of_sample_sizes = grid_of_sample_sizes, 
                   filename = "toy_data.h5")












