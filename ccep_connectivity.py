# Gather all tsv files from a folder and join them into a single tsv file
import os
import pandas as pd
from pathlib import Path

SUBJECT = 'sub-ccepAgeUMCU09'

FOLDER = f'C:/Users/jjbte/Documents/TM3/Afstuderen/Significant_Electrodes/{SUBJECT}/'
OUTPUT = f'C:/Users/jjbte/Documents/TM3/Afstuderen/Significant_Electrodes/{SUBJECT}/'+ \
             f'output/output.tsv'
COUNT = f'C:/Users/jjbte/Documents/TM3/Afstuderen/Significant_Electrodes/'+ \
                              f'{SUBJECT}/output/channel_name_counts.tsv'
Path(f"C:/Users/jjbte/Documents/TM3/Afstuderen/Significant_Electrodes/{SUBJECT}/output").mkdir(parents=True, exist_ok=True)

def main():
    significant_electrodes = join_tsv(FOLDER, OUTPUT)
    sig_channel_count(significant_electrodes)    

def join_tsv(folder, output):
    # Get all tsv files in the folder
    tsv_files = [f for f in os.listdir(folder) if f.endswith('.tsv')]

    # Read all tsv files and concatenate them
    significant_electrodes_local = pd.concat([pd.read_csv(os.path.join(folder, f), sep='\t')
                                        for f in tsv_files])

    # Write the concatenated dataframe to a tsv file
    significant_electrodes_local.to_csv(output, sep='\t', index=False)

    print(f'Joined {len(tsv_files)} tsv files into {output}')

    return significant_electrodes_local

def sig_channel_count(sig_electrodes):
    # Save number of unique channel_name and amount of repetitions in dictionary
    channel_name_counts = sig_electrodes['channel_name'].value_counts().to_dict()

    # Save channel_name_counts to a tsv file
    channel_name_counts_df = pd.DataFrame(list(channel_name_counts.items()),
                                          columns=['channel_name', 'count'])
    channel_name_counts_df.to_csv(COUNT, sep='\t', index=False)

    print(f'Amount of unique channel names: {len(channel_name_counts)}')

if __name__ == '__main__':
    main()
    print('Done')