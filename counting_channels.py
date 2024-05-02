# load the output file with significant electrodes
import pandas as pd

# Load the output file with significant electrodes
significant_electrodes_file = 'C:/Users/jjbte/Documents/TM3/Afstuderen/Significant_Electrodes/sub-ccepAgeUMCU02/output.tsv'
significant_electrodes = pd.read_csv(significant_electrodes_file, sep='\t')

# Save number of unique channel_name and amount of repetitions in dictionary
channel_name_counts = significant_electrodes['channel_name'].value_counts().to_dict()
print(channel_name_counts)

# Save channel_name_counts to a tsv file
channel_name_counts_df = pd.DataFrame(list(channel_name_counts.items()), columns=['channel_name', 'count'])
channel_name_counts_file = 'C:/Users/jjbte/Documents/TM3/Afstuderen/Significant_Electrodes/sub-ccepAgeUMCU02/channel_name_counts.tsv'
channel_name_counts_df.to_csv(channel_name_counts_file, sep='\t', index=False)