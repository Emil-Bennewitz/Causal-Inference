import pandas as pd
import numpy as np
import h5py
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tqdm import tqdm as tqdm
import copy

import core.utile as utile
import core.IPW as IPW
import core.global_params as global_params
from abc import ABC, abstractmethod


def get_df_from_hdf5_experiment(hdf5_file_path, hyperparameter_dict:dict,sample_size:int, is_endogeneous="True",get_estimates = False):
    """
    Retrieve a batch of data with a certain samplesize and hyperparameters from the hdf5 file of simulated experiment data.
    The hyperparameter_dict is converted to the key, so is the sample size.
    :param hdf5_file_path: path to the hdf5 file
    :param hyperparameter_dict: dictionary of hyperparameters. Must contain one value for each hyperparameter in the hdf5 file.
    :param sample_size: int, the sample size of the data you want to retrieve.
    :param is_endogeneous: bool, whether you want to retrieve endogeneous or exogeneous data.
    """
    
    # Create a new dict where we've reordered the hyperparameters to match the order in the keys of the hdf5 file.
    #dicts are technically orderless but the order of insertion is preserved when you do dict.items() for example so I guess that's a lie.
    with h5py.File(hdf5_file_path, "r") as f:
        hp_order = f.attrs["hyperparameter_names"]

        #check if hp_order contains exactly the keys of hyperparameter_dict
        if set(hp_order) != set(hyperparameter_dict.keys()):
            raise ValueError(f"The hyperparameters provided do not exactly match the dataset. Expected {hp_order}, got {hyperparameter_dict.keys()}")

        new_dict = {hp_order[i]:hyperparameter_dict[hp_order[i]] for i in range(len(hp_order))} 
        
        #check the validity of the hyperparameters
        for key, value in new_dict.items():
            valid_values = f.attrs[key] #np array of valid values
            if value not in valid_values:
                raise ValueError(f"Invalid value for hyperparameter {key}. Valid values are {valid_values}")
        
        #check validity of sample size
        valid_sample_sizes = f.attrs["grid_of_sample_sizes"]
        if sample_size not in valid_sample_sizes:
            raise ValueError(f"Invalid sample size {sample_size}. Valid sample sizes are {valid_sample_sizes}")
        
        #Extract Data
        hyperparameter_key = utile.dict_as_string(new_dict)
        sample_size_key = "n_" + str(sample_size)
        endogeneous_key = "endogeneous" if is_endogeneous else "exogeneous"
        group = f[hyperparameter_key]
        sample_size_group = group[sample_size_key]
        endogeneous_group = sample_size_group[endogeneous_key]
        df = pd.read_hdf(hdf5_file_path, key = endogeneous_group.name)
        if get_estimates:
            df_estimates = pd.read_hdf(hdf5_file_path, key = endogeneous_group.name + "_estimates")
            return df,df_estimates
        else:
            return df

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# The outcome regression on the weighted data is an instance of this.
class WeightedModel(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def fit(self, dataset: IPW.dataset, weights: np.array):
        pass
    
    @abstractmethod
    def predict(self, intervention_values_grid: np.array):
        pass

    def _weighted_data(self, dataset:IPW.dataset, weights:np.array):
        """
        A very small function that takes the dataset and weights and outputs df ready for model.fit and model.predict.
        Also sets target and treatment names.
        """
        self.treatment = dataset.treatment
        self.target = dataset.target
        df = dataset.data
        df[dataset.target] = df[dataset.target] * weights
        return df
    

# Two subclasses: the simple linear model y ~ t and loess.
class LinearModel(WeightedModel):
    def __init__(self):
        super().__init__(model=smf.wls)  # Call superclass initializer
        
    def fit(self, dataset: IPW.dataset, weights: np.array):
        self.treatment_name = dataset.treatment
        self.target_name = dataset.target
        self.fitted_model = self.model(formula=f"{self.target_name} ~ {self.treatment_name}", data=dataset.data, weights = weights).fit()
    
    def predict(self, intervention_values_grid: np.array):
        # Predict values based on a grid of intervention values
        return self.fitted_model.predict(pd.DataFrame({self.treatment_name: intervention_values_grid}))

class LoessModel(WeightedModel):
    def __init__(self, bandwidth, span, degree):
        super().__init__(model=utile.loess)  # Call superclass initializer
        self.bandwidth = bandwidth
        self.span = span
        self.degree = degree
        
    def fit(self, dataset: IPW.dataset, weights: np.array):
        #for loess, there's no real fit method, so we just store everything to use in predict.
        self.treatment_name = dataset.treatment
        self.target_name = dataset.target
        self.weights = weights
        self.data = dataset.data

    def predict(self, intervention_values_grid: np.array):
        # Predict values based on a grid of intervention values
        results = self.model(x_eval = intervention_values_grid,
                             x= self.data[self.treatment_name],
                             y= self.data[self.target_name],
                             bandwidth = self.bandwidth,
                             span=self.span, 
                             degree=self.degree, 
                             sample_weights = self.weights)
        return results


class Estimator:
    """
    This class will make estimates for the entire experiment dataset and store them.
    At its core is the method estimate_distribution, which estimates E[Y|do(T=t)] for every batch in the df.
    """
    def __init__(self,
                 model:WeightedModel,
                 WeightCalculator:IPW.WeightCalculator,
                 name_of_treatment:str,
                 name_of_target:str,
                 name_of_adjustment_variables:list,
                 store_hdf5_file:str,
                 read_hdf5_file:str,
                 renormalize_to:int = None):
        
        self.store_hdf5_file = store_hdf5_file
        self.read_hdf5_file = read_hdf5_file
        self.model = model #instantiated model, instance of a subclass of WeightedModel
        #self.intervention_values = intervention_values #Should not be passed as an argument, already exists in the read file.
        with h5py.File(self.read_hdf5_file, "r") as rf:
            self.intervention_values = rf.attrs["grid_of_intervention_values"]

        self.WC = WeightCalculator
        self.name_of_treatment = name_of_treatment
        self.name_of_target = name_of_target
        self.name_of_adjustment_variables = name_of_adjustment_variables

        self.renormalize_to = renormalize_to #For continuous treatments especially, rescaling the weights to have this mean can help numerically.
    #This function estimates E[Y|do(T=t)] for every batch in the df. Returns a (batchsize,n_intervention_values) numpy array of estimates.
    def estimate_distribution(self, batch_df:pd.DataFrame):
        """
        This function estimates E[Y|do(T=t)] for every batch in the df. 

        The parameters are (used to be arguments, now class attributes)-
        :param batch_df: a pandas dataframe with a column "batch" that indicates the batch number.
        :param model: a WeightedModel instance that can fit and predict.
        :param intervention_values: a numpy array vector of intervention values.
        :param weight_method_name: str, the name of the method to use for calculating weights, e.g. "parameterized_mean_normal"
        :param stabilized: bool, whether to use stabilized weights.
        :param weight_nominator_method_name: str, the name of the method to use for calculating the nominator of the weights, e.g. "normal"

        :return: a (batchsize,n_intervention_values) numpy array of estimates,
        and a (batchsize,) numpy vector of the number of trimmed weights for each batch.
        """
        n_batches = max(batch_df["batchnr"]) + 1
        estimates = np.zeros((n_batches, len(self.intervention_values)), dtype=np.float64) 
        estimates_unweighted = np.zeros_like(estimates,dtype=np.float64)
        nb_trimmed_per_batch = np.zeros(n_batches, dtype=np.int32)

        # check divisibility
        assert len(batch_df) % n_batches == 0
        weight_matrix = np.zeros((int(len(batch_df)/n_batches),n_batches),dtype=np.float64)
        GPS_matrix = np.zeros_like(weight_matrix,dtype=np.float64)

        for i in range(n_batches):
            batch = batch_df[batch_df["batchnr"] == i]
            batch_as_dataset = IPW.dataset(df = batch,
                                        treatment = self.name_of_treatment,
                                        treatment_type="continuous",
                                        target = self.name_of_target,
                                        adjustment_variables = self.name_of_adjustment_variables)
            Wobj = self.WC(batch_as_dataset)
            weights = Wobj.get_weights(trimmed = True, renormalize_to=self.renormalize_to)
            GPS = Wobj.f_A_given_Z
            GPS_matrix[:,i] = GPS
            weight_matrix[:,i] = weights
            nb_trimmed_per_batch[i] = Wobj.number_of_trimmed_weights
            
            #make our estimates
            self.model.fit(dataset = batch_as_dataset, weights = weights)
            estimates[i,:] = self.model.predict(self.intervention_values)

            #also make unweighted estimates to compare to
            self.model.fit(batch_as_dataset,np.ones_like(weights))
            estimates_unweighted[i,:] = self.model.predict(self.intervention_values)
            assert np.nan not in estimates[i,:]
            assert np.nan not in estimates_unweighted[i,:]
            
        return {"estimates":estimates, "nb_trimmed_per_batch":nb_trimmed_per_batch,"unweighted_estimates":estimates_unweighted,"weights":weight_matrix, "f_A_given_Z":GPS_matrix}
    
    def store_data(self, df, estimate_dict, path_to_subgroup:str):
        """
        Stores the output of the estimate_distribution function in the hdf5 file, i.e. stores one batch with estimates.
        """

        estimates_array = estimate_dict["estimates"]
        number_of_trimmed_weights_vector = estimate_dict["nb_trimmed_per_batch"]
        unweighted_estimates_array = estimate_dict["unweighted_estimates"]
        GPS_matrix = estimate_dict["f_A_given_Z"]

                        
        estimates_df = pd.DataFrame(estimates_array,columns = [f"do(Treatment={i})" for i in self.intervention_values])
        df.to_hdf(self.store_hdf5_file,key = path_to_subgroup, mode = "a")

        unweighted_estimates_df = pd.DataFrame(unweighted_estimates_array,columns = [f"do(Treatment={i})" for i in self.intervention_values])
        
        n_trimmed_weights_series = pd.Series(number_of_trimmed_weights_vector, name = "n_trimmed")
        ones = pd.Series(np.ones_like(number_of_trimmed_weights_vector), name = "is_weighted")
        zeros = pd.Series(np.zeros_like(number_of_trimmed_weights_vector), name = "is_weighted")
        weighted_df = pd.concat([estimates_df, n_trimmed_weights_series,ones], axis = 1)
        unweighted_df = pd.concat([unweighted_estimates_df, n_trimmed_weights_series,zeros], axis = 1)
        new_df = pd.concat([weighted_df,unweighted_df], axis = 0)
        new_df.to_hdf(self.store_hdf5_file,key = path_to_subgroup+"_estimates", mode = "a")#We change the name of the key because we have already stored the data there.
        with h5py.File(self.store_hdf5_file, "a") as f:
            group=f.get(path_to_subgroup+"_weights")
            if group is not None:
                del f[path_to_subgroup+"_weights"]  
            f.create_dataset(path_to_subgroup+"_weights",data = estimate_dict["weights"])
            group=f.get(path_to_subgroup+"_f_A_given_Z")
            if group is not None:
                del f[path_to_subgroup+"_f_A_given_Z"]
            f.create_dataset(path_to_subgroup+"_f_A_given_Z",data = GPS_matrix)
            
    def run_experiment(self):

        #Check whether the store file already exists. If it does, we don't want to overwrite it automatically, but prompt the user.
        if os.path.exists(self.store_hdf5_file):
            print(f"Caution, file {self.store_hdf5_file} already exists. Proceeding will overwrite the file.")
            proceed = input("Do you want to proceed? (y/n)")
            if proceed.lower() != "y":
                print("Terminating the calculation.")
                return None
            print("Proceeding to overwrite the file.")

        with h5py.File(self.read_hdf5_file, "r") as rf, h5py.File(self.store_hdf5_file, "w") as sf:

            #Write attributes to the new file
            for key,value in rf.attrs.items():
                sf.attrs[key] = value
            sf.attrs["grid_of_intervention_values"] = self.intervention_values

            #Go get a batch of data
            hyperparameter_keys = list(rf.keys())
            print("Calculating estimates... ")
            with tqdm(total=len(hyperparameter_keys)*len(rf.attrs["grid_of_sample_sizes"])) as pbar:
                for hyperparameter_key in hyperparameter_keys:
                    hyperparameter_group = rf[hyperparameter_key]
                    sample_size_keys = list(hyperparameter_group.keys())
                    for sample_size_key in sample_size_keys:
                        if sample_size_key == "true_counterfactual_means":
                            #This key is present for every hyperparameter but is not a sample size. I forgot about that.
                            #We store and then continue.
                            
                            #store the counterfactual means in the new file.
                            hp_group = sf.require_group(hyperparameter_key)
                            #overwrite the dataset if it already exists
                            if "true_counterfactual_means" in hp_group:
                                del hp_group["true_counterfactual_means"]
                            sf[hyperparameter_key].create_dataset("true_counterfactual_means", data = hyperparameter_group["true_counterfactual_means"][:]) #TODO: We are taking the intervention values from the dataset! SEE BUG
                            #BUG: We are specifying intervention values twice, once when creating the dataset and once when estimating. This is wrong.
                            #There is confusion on which intervention values we do what with. The true counterfactual means are calculated with the old intervention values, stored in rf,
                            #but the estimates are calculated with the new intervention values, stored in self.
                            #Am in the process of fixing, by the time you read this it may be fixed.
                            #UPDATE: so for now the fix is that the grid is read from the file. Only problem: It is actually tricky to guess a good grid before seeing the data.
                            continue 

                        sample_size_group = hyperparameter_group[sample_size_key]
                        endogeneous_group = sample_size_group["endogeneous"]

                        #read the batch, create estimates, store the augmented batch.
                        df = pd.read_hdf(self.read_hdf5_file, key = endogeneous_group.name)
                        estimate_dict = self.estimate_distribution(df)
                        self.store_data(df, estimate_dict,endogeneous_group.name)
                        pbar.update(1)  

path = os.path.join(global_params.PROJECT_PATH,"data","samplesize")
model_loess = LoessModel(bandwidth = None, span = 1.0, degree = 2)
model_linear = LinearModel()
E_loess = Estimator(model = model_linear, 
                    WeightCalculator=IPW.WeightCalculator(method_name = "custom_linear_model",
                                                          stabilized = True, 
                                                          nominator_method_name = "normal",
                                                          additional_method_arguments={"formula": "I(np.log(CuttingForce)) ~ I(np.log(Speed)) + I(np.log(FeedRate))"}),
                    name_of_treatment = "CuttingForce",
                    name_of_target = "Drift",
                    name_of_adjustment_variables = ["Speed","FeedRate"],
                    store_hdf5_file = os.path.join(path,"toy_estimatesv01.h5"),
                    read_hdf5_file = os.path.join(path,"toy_data.h5"),
                    renormalize_to = 1.0) 

E_loess.run_experiment()




