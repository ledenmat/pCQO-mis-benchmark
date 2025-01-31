

# Function to process the input file
def process_data(file_path):
    # Dictionary to store the aggregated results and times for each batch size
    batch_data = {}

    current_batch = None

    index = 0;

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("G"): # REPLACE THIS LETTER WITH THE FIRST LETTER OF YOUR GRAPHS or write this script better than I did :)
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
                batch_data[current_batch]["result_sum"] += result
                continue
            if index == 2:
                index = 0
                time = float(line)
                batch_data[current_batch]["time_sum"] += time
                continue
    
    return batch_data


        
def main(file_path):
    # Process the data
    batch_data = process_data(file_path)

    # Calculate the average results and total times for each batch size
    for batch_size in sorted(batch_data.keys()):
        result_sum = batch_data[batch_size]["result_sum"]
        total_time = batch_data[batch_size]["time_sum"]
        count = batch_data[batch_size]["count"]
        average_result = result_sum / count
        print(f"{batch_size} & {average_result:.3f} & {total_time:.3f}\\\\")

# Path to your input file
file_path = './result.txt'

# Process and print the data
main(file_path)