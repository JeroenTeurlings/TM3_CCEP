# Gather all tsv files from a folder and join them into a single tsv file
import os
from pathlib import Path
import pandas as pd

SUBJECT = None

def main():
    print('Joining tsv files and counting significant electrodes')
    global FOLDER, OUTPUT, COUNT
    FOLDER = f'C:/Users/jjbte/Documents/TM3/Afstuderen/Significant_Electrodes/{SUBJECT}/'
    OUTPUT = FOLDER + f'output/{SUBJECT}_output.tsv'
    COUNT = FOLDER + f'/output/{SUBJECT}_channel_name_counts.tsv'
    
    Path(FOLDER + "output").mkdir(parents=True, exist_ok=True)
    significant_electrodes = join_tsv(FOLDER, OUTPUT)
    sig_channel_count(significant_electrodes)

def join_tsv(folder, output):
    # Get all tsv files in the folder
    tsv_files = [f for f in os.listdir(folder) if f.endswith('.tsv')]
    significant_electrodes_local = pd.DataFrame()

    # Read all tsv files and concatenate them
    for f in tsv_files:
        try:
            df = pd.read_csv(os.path.join(folder, f), sep='\t')
            if not df.empty:
                significant_electrodes_local = pd.concat([significant_electrodes_local, df])
        except pd.errors.EmptyDataError:
            print(f'Empty file: {f}')
            continue

    # Write the concatenated dataframe to a tsv file
    significant_electrodes_local.to_csv(output, sep='\t', index=False)

    print(f'Joined {len(tsv_files)} tsv files into {output}')

    return significant_electrodes_local

def sig_channel_count(sig_electrodes):
    # Save number of unique channel_name and amount of repetitions in dictionary
    channel_name_counts = sig_electrodes.groupby('channel_name').agg(
        count=pd.NamedAgg(column='channel_name', aggfunc='size'),
        x=pd.NamedAgg(column='x_coordinate', aggfunc='first'),  # replace 'x' with your actual column name
        y=pd.NamedAgg(column='y_coordinate', aggfunc='first'),  # replace 'y' with your actual column name
        z=pd.NamedAgg(column='z_coordinate', aggfunc='first')   # replace 'z' with your actual column name
    ).reset_index()
    channel_name_counts.to_csv(COUNT, sep='\t', index=False)

    print(f'Amount of unique channel names: {len(channel_name_counts)}')

if __name__ == '__main__':
    main()
    print('Done')
