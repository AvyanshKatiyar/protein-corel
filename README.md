# protein-corel
Predicts protein correlation maps using deep learning 


# Bash Scripts
## train6.sh 

This script helps with the automation of setting up training experiments, particularly useful when conducting many similar experiments with slightly different parameters. By changing the variables at the top of the script, you can quickly create new directories with the appropriate scripts and submit them as jobs to a job scheduler.

1. It declares variables for the experiment. For instance, `VAR` is set to 1000 and `PREFIX_CPU` is defined to form a unique identifier for the experiment. It's important to note that `PREFIX_CPU` is constructed by concatenating various strings and the value of `VAR`.

2. It creates a new directory with the name equal to the value of `PREFIX_CPU`. The `-p` flag ensures that the directory is created only if it doesn't already exist.

3. It then copies two files, `train7.sh` and `train_corel.py`, into the new directory, renaming them to `train.sh` and `train.py` respectively. These files likely contain the shell script for running the training and the Python script defining the training procedure, respectively.

4. It uses the `sed` command to replace placeholders in the copied `train.py` script. Specifically, it replaces `NUM_CHAINS` with the value of `VAR` and `NUM_CPUS` with the value of `PREFIX_CPU`.

5. Similarly, it replaces the `DIRECTORY` placeholder in `train.sh` script with the value of `PREFIX_CPU`.

6. It changes the working directory to the newly created directory.

7. Finally, it submits the `train.sh` script as a job to a job scheduler (likely PBS) using the `qsub` command. The job is submitted to the `low` queue, it requests one node with 8 CPUs, it's submitted under the `chemical` project, and it sets a walltime limit of 96 hours. The job is named `test$VAR`.


## train7.sh

This script is designed to navigate to a specific directory, load the Anaconda Python 3 module, and then execute a Python script named `train.py`. Here's a more detailed explanation:

1. `cd /home/chemical/btech/ch1190960/scratch/BTP/pdnet-master/DIRECTORY`: This command changes the current working directory to a specific path. Note that `DIRECTORY` should be replaced by the actual directory name.

2. `module load apps/anaconda/3`: This command loads the Anaconda Python 3 module, which is necessary to run Python scripts. Anaconda is a distribution of Python that comes pre-packaged with many useful libraries for scientific computing.

3. `python3 "train.py"`: Finally, this command runs the `train.py` Python script using Python 3.

# Python Scripts

## train_corel.py

### Functions

1. `load_list(file_lst, max_items = 1000000)`: This function loads a list of protein IDs from a given file. It takes as input the file path `file_lst` and an optional parameter `max_items` which defaults to `1,000,000`. The function returns a list of protein IDs, with length limited by `max_items`.

2. `summarize_channels(x, y)`: This function provides summary statistics about each channel in the input tensor `x` and the output tensor `y`. It calculates and prints the average, maximum, and sum of each channel in the tensor `x`. It also prints the minimum, mean, and maximum of the tensor `y`.

3. `get_bulk_output_maps_dist(pdb_id_list, all_dist_or_corel_paths, OUTL)`: This function creates and returns a tensor containing all output maps (distance maps) for the list of Protein Data Bank (PDB) IDs given by `pdb_id_list`. The size of each output map is given by `OUTL`. The function assumes that all values not specified in the map are at an infinite distance.

4. `get_bulk_output_maps_corel(pdb_id_list, all_dist_or_corel_paths, OUTL)`: Similar to `get_bulk_output_maps_dist`, but the function assumes that all values not specified in the map are not correlated (i.e., have a correlation of 0.0).

5. `get_input_output_dist(pdb_id_list, all_feat_paths, all_dist_or_corel_paths, pad_size, OUTL, expected_n_channels)`: This function generates input and output tensors for a list of PDB IDs. The function loads features for each protein, pads the features with zeros, and then crops or pads the resulting feature map to match the desired output length (`OUTL`). The function does the same for output maps, but pads with a distance of 100.0. The function returns the input and output tensors.

6. `get_input_output_corel(pdb_id_list, all_feat_paths, all_dist_or_corel_paths, pad_size, OUTL, expected_n_channels)`: Similar to `get_input_output_dist`, but for correlation maps rather than distance maps.

7. `get_sequence(pdb, feature_file)`: This function loads and returns the sequence for a given PDB ID from a given feature file.

8. `get_feature(pdb, all_feat_paths, expected_n_channels)`: This function loads and returns the feature map for a given PDB ID. The feature map is loaded from one of the paths listed in `all_feat_paths` and should have `expected_n_channels` channels.

9. `get_map(pdb, all_dist_or_corel_paths, expected_l = -1)`: This function loads and returns the distance map for a given PDB ID. The distance map is loaded from one of the paths listed in `all_dist_or_corel_paths` and should have a length of `expected_l`.

10. `five(a, n)`: This function sets to zero every fifth element in a 2D array `a` of size `n x n`. 

11. `zero_diagonal(a,n)`: This function sets to zero the diagonal of a 2D array `a` of size `n x n`.

12. `get_map_corel(pdb, all_dist_or_corel_paths, expected_l = -1)`: This function loads and returns the correlation map for a given PDB ID, replacing all elements with a distance less than or equal to 8.0 with -1.0 and all elements with a distance greater than 8 with


13. `save_dist_rr(pdb, all_feat_paths, pred_matrix, file_rr)`

This function saves distance prediction results into an RR format file. Here's a breakdown of the parameters:
  - `pdb`: String representing the Protein Data Bank (PDB) ID of the protein.
  - `all_feat_paths`: List of paths where the feature files can be found.
  - `pred_matrix`: The predicted distance matrix to be written to the file. This is a numpy array where element [i,j] represents the predicted distance between amino acids i and j.
  - `file_rr`: The path of the output RR file where the results will be written.

The function first checks the `all_feat_paths` list to find and load the features of the given protein. It then checks if the features were loaded properly, if not, it will exit with an error message. After that, it retrieves the sequence of amino acids for the protein and starts writing it to the RR file. 

Then it creates a copy of the prediction matrix and averages the values of each pair of symmetric elements. The function then writes the averaged distance values to the RR file, skipping pairs of residues that are less than 5 positions apart in the sequence.

14. `save_contacts_rr(pdb, all_feat_paths, pred_matrix, file_rr)`

This function saves contact prediction results into an RR format file. The parameters are the same as for `save_dist_rr`.

This function works similarly to `save_dist_rr`. It loads the features and checks if they were properly loaded. Then, it retrieves the sequence of amino acids and writes it to the RR file. 

It then creates a copy of the prediction matrix and averages the values of each pair of symmetric elements. For each pair of amino acids that are not less than 5 positions apart in the sequence, it writes a line to the RR file with the pair's indices, two zero values (which could represent unused features such as lower and upper bounds), and the averaged prediction value. 

In both cases, the functions print a confirmation message when the RR file is successfully written. Note that the use of `exit(1)` will cause the program to terminate if the protein features cannot be loaded, so these functions should be used with care.

## Generators

The `DistGenerator` and `CorelGenerator` classes are subclasses of the `Sequence` class provided by Keras, which is a useful tool to handle large amounts of data in chunks (batches) instead of loading everything into memory at once. They are essentially data loaders for the training data.

1. `DistGenerator` class:

The `DistGenerator` class is designed to generate data for the model training process specifically for distance map predictions. Here's the documentation for the class:

- `__init__(self, pdb_id_list, features_path, distmap_path, dim, pad_size, batch_size, expected_n_channels)`: 

    This is the constructor method for the `DistGenerator` class. It initializes the object with the following attributes:
    - `pdb_id_list`: List of Protein Data Bank IDs for the proteins whose data will be used for training.
    - `features_path`: Path to the features files.
    - `distmap_path`: Path to the distance map files.
    - `dim`: Dimension of the data.
    - `pad_size`: Padding size for the data.
    - `batch_size`: Number of samples per gradient update.
    - `expected_n_channels`: Expected number of channels in the data.

- `on_epoch_begin(self)`: 

    This method is called at the beginning of each epoch. It creates an array of indices for the PDB IDs and shuffles them.

- `__len__(self)`: 

    This method returns the number of batches per epoch based on the batch size.

- `__getitem__(self, index)`: 

    This method retrieves the batch of data at the specified index. It calls the `get_input_output_dist` function to fetch the batch of input and output data and scales the output data.

2. `CorelGenerator` class:

The `CorelGenerator` class is designed to generate data for the model training process specifically for correlation map predictions. Here's the documentation for the class:

- `__init__(self, pdb_id_list, features_path, corelmap_path, dim, pad_size, batch_size, expected_n_channels)`: 

    This is the constructor method for the `CorelGenerator` class. It initializes the object with the following attributes:
    - `pdb_id_list`: List of Protein Data Bank IDs for the proteins whose data will be used for training.
    - `features_path`: Path to the features files.
    - `corelmap_path`: Path to the correlation map files.
    - `dim`: Dimension of the data.
    - `pad_size`: Padding size for the data.
    - `batch_size`: Number of samples per gradient update.
    - `expected_n_channels`: Expected number of channels in the data.

- `on_epoch_begin(self)`: 

    This method is called at the beginning of each epoch. It creates an array of indices for the PDB IDs and shuffles them.

- `__len__(self)`: 

    This method returns the number of batches per epoch based on the batch size.

- `__getitem__(self, index)`: 

    This method retrieves the batch of data at the specified index. It calls the `get_input_output_corel` function to fetch the batch of input and output data.

## Plots
​​1. `plot_protein_io(X, Y)`:

This function visualizes the protein input and output data. 

- Inputs: `X` is a 3D numpy array that represents the input data for a protein. The shape of `X` is typically (L, L, C), where L is the protein sequence length and C is the number of channels. `Y` is a 2D numpy array representing the output data for a protein. The shape of `Y` is (L, L).
- Functionality: This function creates a figure with multiple subplots - one for each channel of `X` and one for `Y`. For each subplot, it creates a heatmap using seaborn to visualize the data.
- Outputs: This function does not return anything; its primary purpose is visualization.

2. `plot_learning_curves(history)`:

This function plots the learning curves of the model during training.

- Inputs: `history` is an object returned by the `fit` method of a keras Model. It contains the training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
- Functionality: This function plots either the mean absolute error (MAE) or accuracy curves for training and validation data across epochs. 
- Outputs: This function does not return anything; its primary purpose is visualization.

3. `plot_four_pair_maps(T, P, pdb_list, length_dict)`:

This function plots the true and predicted maps for four proteins.

- Inputs: `T` is a 4D numpy array representing the true maps for the proteins, `P` is a 4D numpy array representing the predicted maps, `pdb_list` is a list of protein ids, and `length_dict` is a dictionary mapping protein ids to their sequence lengths.
- Functionality: This function creates a figure with multiple subplots - two for each of the four proteins. The first subplot is a heatmap of the true map and the second is a heatmap of the predicted map.
- Outputs: This function does not return anything; its primary purpose is visualization.

4. `plot_channel_histograms(X)`:

This function plots histograms for each channel of the input data.

- Inputs: `X` is a 3D numpy array that represents the input data for a protein. The shape of `X` is typically (L, L, C), where L is the protein sequence length and C is the number of channels.
- Functionality: This function creates a histogram for each channel of `X` to visualize the distribution of the data in each channel.
- Outputs: This function does not return anything; its primary purpose is visualization. 

## Errors
1. `calculate_mae(PRED, YTRUE, pdb_list, length_dict)`:

This function calculates the mean absolute error (MAE) for the predicted and true distance maps of proteins at two different distance thresholds (8 and 12) and two different separation lengths (12 and 24).

- Inputs: `PRED` is a 4D numpy array representing the predicted distance maps for proteins. `YTRUE` is a 4D numpy array representing the true distance maps. `pdb_list` is a list of protein ids, and `length_dict` is a dictionary mapping protein ids to their sequence lengths.
- Functionality: The function calculates four types of MAEs for each protein. It considers only those elements of the distance maps that have a sequence separation above a certain threshold and a distance below a certain threshold. It averages the values in the upper and lower triangles of the predicted distance maps to create an averaged map. It then calculates the MAEs between the true and averaged maps.
- Outputs: The function prints the calculated MAEs for each protein and the average MAE across all proteins. The function does not return anything.

2. `calculate_mae_corel(PRED, YTRUE, pdb_list, length_dict)`:

This function calculates the mean absolute error (MAE) for the predicted and true correlation maps of proteins.

- Inputs: `PRED` is a 4D numpy array representing the predicted correlation maps for proteins. `YTRUE` is a 4D numpy array representing the true correlation maps. `pdb_list` is a list of protein ids, and `length_dict` is a dictionary mapping protein ids to their sequence lengths.
- Functionality: The function averages the values in the upper and lower triangles of the predicted correlation maps to create an averaged map. It then calculates the MAEs between the true and averaged maps.
- Outputs: The function prints the calculated MAEs for each protein and the average MAE across all proteins. The function does not return anything.

## Contact Processing

Here is a brief documentation for each of the functions defined in your code:

1. `distance_to_contacts(distance_matrix)`: This function converts a distance matrix to a contact probability matrix. The contact probability is defined as 4.0 divided by the distance. If the probability is greater than 1.0, it's capped at 1.0.

2. `distance_to_contacts_corel(corel_matrix)`: This function converts a correlation matrix to a contact probability matrix. If the correlation value is greater than 1.0, it's capped at 1.0.

3. `calculate_contact_precision_in_distances(PRED, YTRUE, pdb_list, length_dict)`: This function calculates the contact precision in the context of distances. Contact is considered if the distance is less than 8.0. The contact precision is then computed using the function `calculate_contact_precision`.

4. `calculate_contact_precision(PRED, YTRUE, pdb_list, length_dict)`: This is the main function to calculate the contact precision based on predicted and true values. This function uses several measures: "top_L5", "top_L", and "top_Nc" for both long-range and medium-long-range contacts. The function computes these measures for each structure in the dataset and finally reports average values.

    - Long-range (lr) contacts: Residues are considered to be in contact if they are more than 24 residues apart in the sequence.
    - Medium-long-range (mlr) contacts: Residues are considered to be in contact if they are 12 or more residues apart in the sequence.
    - Top L/5, Top L, Top Nc: The function considers the top L/5, top L, and top Nc scoring predicted contacts, where L is the length of the protein and Nc is the number of true contacts. The precision is computed as the proportion of these that are true contacts.

In each of the functions above, a major input to the function is a contact map or distance matrix. A contact map (or a distance matrix) is a square matrix where each element represents the contact (or distance) between a pair of residues in the protein structure. The input contact map or distance map could either be predicted (PRED) or the true contact map or distance map (YTRUE).

Another important input is a dictionary (length_dict) where the keys are Protein Data Bank (PDB) identifiers (pdb_list) and the values are the corresponding lengths of the proteins. The PDB is a database of experimentally determined protein structures.

Please note that while the code assumes contacts are defined as distances less than 8.0, in practice the definition of a contact may vary slightly depending on the specific study or method being used.



## `eval_distance_predictions` 

This function is used to evaluate a model's predictions for either distance or corel (correlation) matrices related to protein structures. It uses the model and information about protein sequences to generate predicted matrices, compare them with the true matrices, and output various evaluation metrics.

Function parameters:

1. `my_model`: This is the trained model that will be used for generating the predicted distance or corel matrices.

2. `corel_or_dist`: This is a string input that determines whether the function will evaluate for distance or corel matrices. Accepted values are 'distance' or 'corel'.

3. `my_list`: This is a list of protein identifiers (PDB IDs) for which the evaluation will be carried out.

4. `my_length_dict`: A dictionary where the keys are the PDB IDs and the values are the corresponding protein lengths.

5. `my_dir_features`: This is the directory where the feature files for each protein in `my_list` are stored.

6. `my_dir_distance`: This is the directory where the true distance matrices for each protein in `my_list` are stored.

7. `pad_size`: The size of padding to be added around the matrices.

8. `flag_plots`: A boolean flag. If True, the function will also generate plots for the true and predicted matrices.

9. `flag_save`: A boolean flag. If True, the function will save the predicted matrices and other outputs in a specified directory.

10. `LMAX`: The maximum protein length considered for the analysis.

11. `expected_n_channels`: The expected number of channels in the input data for the model.

The function starts by generating the predicted matrices using the model and the protein features. It then compares these predicted matrices with the true matrices to compute the Mean Absolute Error (MAE). If the function is operating in 'distance' mode, it will also calculate the precision of the predicted contact maps at various thresholds. If the function is in 'corel' mode, it will calculate the mean absolute error for the corel matrices. If the `flag_save` parameter is True, the function will also save the predicted matrices and the protein sequences in a specified format.

Deep Learning Structure

Sure, here is a brief documentation for each function in the given code block:

1. `basic_fcn`: This function creates a fully convolutional network (FCN) with a specified number of convolutional layers, each followed by batch normalization and a tanh activation function. The model accepts a 3D input and produces a single-channel output.

   Parameters:
   - `L`: Length of the input sequences.
   - `num_blocks`: Number of blocks (convolutional layers) in the FCN.
   - `width`: Number of filters in each convolutional layer.
   - `expected_n_channels`: Expected number of channels in the input data.

2. `deepcon_rdd`: This function creates the DEEPCON model with a specified number of residual blocks. Each block contains two convolutional layers with a tanh activation function and batch normalization. This architecture also includes a dropout layer.

   Parameters:
   - `L`: Length of the input sequences.
   - `num_blocks`: Number of blocks (convolutional layers) in the network.
   - `width`: Number of filters in each convolutional layer.
   - `expected_n_channels`: Expected number of channels in the input data.

3. `deepcon_rdd_distances`: This is a variant of the DEEPCON model, where the dilation rate changes cyclically within the residual blocks.

   Parameters:
   - `L`: Length of the input sequences.
   - `num_blocks`: Number of blocks (convolutional layers) in the network.
   - `width`: Number of filters in each convolutional layer.
   - `expected_n_channels`: Expected number of channels in the input data.

4. `val_train`: This function plots the mean absolute error (mae) of the model for both the training and validation data sets as a function of the epoch number.

   Parameter:
   - `history`: A `History` object. Its `History.history` attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values.

5. `loss_plot`: This function plots the loss of the model for both the training and validation data sets as a function of the epoch number.

   Parameter:
   - `history`: A `History` object. Its `History.history` attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values.


# Jupyter Notebook


## Visualization Functions

### corel_get_map

Sure, here's the documentation for the `corel_get_map` function.

`corel_get_map` is a function that converts protein distance maps into correlation maps.

Parameters:
- `pdb`: The Protein Data Bank identifier for the protein of interest. This identifier is used to locate the correct distance map file in the provided path.
- `path`: The file path to the directory where the protein distance map files are stored. The function will look for a file with a name that matches the PDB identifier and the extension '-cb.npy'.
- `expected_l`: (Optional) The expected length of the protein sequence. This parameter is not used in the current function but is available for potential future use.

Returns:
- `C`: A correlation matrix derived from the input distance map. The correlation matrix is a symmetric square matrix where the value at position (i, j) represents the correlation between positions i and j in the protein sequence.

Procedure:

1. The function loads the distance map from the specified file path. The distance map is stored in the variable `cb_map`.
2. Any distance values less than or equal to 8 are set to -1, and values greater than 8 are set to 0.
3. The `KK` matrix is computed, where each diagonal element is the negative sum of its row (excluding the diagonal element itself), and off-diagonal elements are copied from `Y`.
4. Then, the eigenvalues and eigenvectors of `KK` are computed.
5. Next, the `u` matrix is computed by summing up the outer products of the eigenvectors divided by their corresponding eigenvalues, excluding the first eigenvalue and eigenvector.
6. Finally, the correlation matrix `C` is computed, where each element at position (i, j) equals `CC[i][j]` divided by the square roots of `CC[i][i]` and `CC[j][j]`.

Note: This function comes with a warning to handle missing or NaN values in the input distance map. When the dataset is 'cameo', these values are ignored, but for other datasets, the function will stop if any NaN value is found.




### Function name: `predicted_map`

`predicted_map` is a function that loads a predicted protein contact map and plots it as an image.

#### Parameters:

- `directory`: The relative directory path where the predicted protein contact map files are stored. The function will look for a file named 'P-cb.npy'.
- `protein_number`: The index of the protein for which the contact map will be plotted. This refers to the index of the protein in the loaded numpy array.
- `precision_multiplier`: The precision multiplier is a scaling factor used to adjust the precision of the predicted contact maps. It is used to divide the values of the predicted map.

#### Returns:
This function doesn't return anything but saves a .png file of the plotted contact map image.

#### Procedure:
1. The function constructs the full file path by appending the directory to a base path, and loads the predicted contact map from this file.
2. It then removes the padding from the loaded map. The size of padding is determined by `pad_size`.
3. The function then selects the contact map of the specified protein and scales it by dividing by `precision_multiplier`.
4. It determines the size of the protein by finding the index of the first diagonal element that equals zero in the scaled map.
5. The contact map is then cropped to this determined size.
6. Finally, the cropped contact map is plotted using `matplotlib.pyplot.imshow`, with a red-blue colormap. The colorbar is also included in the plot. The figure is saved to a .png file with dpi 500.

#### Note:
In this function, the base path is hard-coded, so this function may need to be adjusted to fit the user's specific directory structure. Also, this function currently limits the length of the protein to 200 residues, so it might not work properly for proteins that are longer than this. This limitation could be removed or adjusted as needed.

### Function name: `predicted_diagonal`

`predicted_diagonal` is a function that loads a predicted protein contact map, extracts the diagonal, and plots it.

#### Parameters:

- `directory`: The relative directory path where the predicted protein contact map files are stored. The function will look for a file named 'P-cb.npy'.
- `protein_number`: The index of the protein for which the contact map diagonal will be plotted. This refers to the index of the protein in the loaded numpy array.
- `precision_multiplier`: The precision multiplier is a scaling factor used to adjust the precision of the predicted contact maps. It is used to divide the values of the predicted map.

#### Returns:
This function doesn't return anything but generates a plot of the diagonal of the contact map for the specified protein.

#### Procedure:
1. The function constructs the full file path by appending the directory to a base path, and loads the predicted contact map from this file.
2. It then removes the padding from the loaded map. The size of padding is determined by `pad_size`.
3. The function then selects the contact map of the specified protein and scales it by dividing by `precision_multiplier`.
4. It crops the contact map to a predetermined size (currently set to 200x200).
5. The diagonal of the cropped contact map is extracted.
6. Finally, the diagonal is plotted using `matplotlib.pyplot.plot`, with a dotted line style.

#### Note:
In this function, the base path is hard-coded, so this function may need to be adjusted to fit the user's specific directory structure. Also, this function currently limits the length of the protein to 200 residues, so it might not work properly for proteins that are longer than this. This limitation could be removed or adjusted as needed.


