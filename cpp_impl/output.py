import os
import pandas as pd



# Function to process the input file
def process_data(file_path):
    # Dictionary to store the aggregated results and times for each batch size
    batch_data = {}

    current_batch = None

    index = 0;

    with open(file_path, 'r') as file:
        for line in file:
            if line[0].isalpha(): # REPLACE THIS LETTER WITH THE FIRST LETTER OF YOUR GRAPHS or write this script better than I did :)
                index = 0;
                continue
            if index == 0:
                index += 1
                current_batch = int(line)
                if current_batch not in batch_data:
                    batch_data[current_batch] = {"result_sum": 0, "time_sum": 0.0, "count": 0}
                batch_data[current_batch]["count"] += 1
                continue
            if index == 1:
                index += 1
                result = int(line)
                if result == 0:
                    batch_data[current_batch]["count"] -= 1
                batch_data[current_batch]["result_sum"] += result
                continue
            if index == 2:
                index = 0
                time = float(line)
                batch_data[current_batch]["time_sum"] += time
                continue
    
    return batch_data


        
def main(directory_path):
    # Iterate through all files in the directory that start with "er_700"
    # Initialize a dictionary to store data for the DataFrame
    data = {}

    for file_name in os.listdir(directory_path):
        if file_name.startswith("er_700"):
            # Extract gamma, gamma_prime, and learning_rate from the file name
            parts = file_name.split('_')
            gamma = float(parts[2])
            gamma_prime = int(parts[3])
            learning_rate = float(parts[4][:-4])

            file_path = os.path.join(directory_path, file_name)
            print(f"Processing file: {file_name}")
            
            # Process the data
            batch_data = process_data(file_path)

            # Create a DataFrame for each batch size
            for batch_size in sorted(batch_data.keys()):
                result_sum = batch_data[batch_size]["result_sum"]
                count = batch_data[batch_size]["count"]
                average_result = result_sum / count

                # Store the result in the dictionary
                if (gamma, learning_rate, batch_size) not in data:
                    data[(gamma, learning_rate, batch_size)] = {}
                data[(gamma, learning_rate, batch_size)][gamma_prime] = average_result

    # Create and print a DataFrame for each batch size
    batch_sizes = set(key[2] for key in data.keys())
    for batch_size in sorted(batch_sizes):
        df_data = {}
        for (gamma, learning_rate, b_size), values in data.items():
            if b_size == batch_size:
                if gamma not in df_data:
                    df_data[gamma] = {}
                df_data[gamma][learning_rate] = f"{values.get(0, 'N/A'):.2f} - {values.get(7, 'N/A'):.2f}" if isinstance(values.get(0, 'N/A'), (int, float)) and isinstance(values.get(7, 'N/A'), (int, float)) else f"{values.get(0, 'N/A')} - {values.get(7, 'N/A'):.2f}"

        df = pd.DataFrame(df_data).sort_index(axis=0).sort_index(axis=1)
        print(f"Number of Batches Solved {batch_size}:\n{df.to_markdown()}\n")
    #     if file_name.startswith("er_700"):
    #         file_path = os.path.join(directory_path, file_name)
    #         print(f"Processing file: {file_name}")
            
    #         # Process the data
    #         batch_data = process_data(file_path)

    #         # Calculate the average results and total times for each batch size
    #         for batch_size in sorted(batch_data.keys()):
    #             result_sum = batch_data[batch_size]["result_sum"]
    #             total_time = batch_data[batch_size]["time_sum"]
    #             count = batch_data[batch_size]["count"]
    #             average_result = result_sum / count
    #             print(f"{batch_size} & {average_result:.3f} & {total_time:.3f}\\\\")

# Path to your directory containing the input files
directory_path = './'

# Process and print the data for all matching files
main(directory_path)