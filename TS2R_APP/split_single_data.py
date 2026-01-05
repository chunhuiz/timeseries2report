"""
Before using our APP, you can prepare your own data in the same format as 'test.pickle' in this script.
"""
import pickle  # Import the pickle module for serializing and deserializing Python objects
import os  # Import os module to handle file path operations

if __name__ == '__main__':  # Check if the script is being run directly
    SAVE_INDEX = 1  # Specify the index of the data to be saved
    CURRENT_DIR = os.getcwd()  # Get the current working directory
    FILE_PATH = os.path.join(CURRENT_DIR, 'TJU_raw_data.pickle')  # Construct the path to the input pickle file
    SAVE_PATH = os.path.join(CURRENT_DIR, 'test.pickle')  # Construct the path to the output pickle file
    
    # Open the input pickle file in read-binary mode
    with open(FILE_PATH, 'rb') as f:
        data = pickle.load(f)  # Load the data from the pickle file
    
    # Prepare a dictionary to save selected subsets of data
    save_data = {
        'charge_pure': [data['charge_pure'][SAVE_INDEX]],  # Extract the charge_pure data at the specified index
        'charge_tran': [data['charge_tran'][SAVE_INDEX]],  # Extract the charge_tran data at the specified index
        'discharge_pure': [data['discharge_pure'][SAVE_INDEX]],  # Extract the discharge_pure data at the specified index
        'discharge_tran': [data['discharge_tran'][SAVE_INDEX]],  # Extract the discharge_tran data at the specified index
    }
    
    # Open the output pickle file in write-binary mode
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump(save_data, f)  # Serialize the save_data dictionary and write it to the output file

