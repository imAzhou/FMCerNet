from tqdm import tqdm
import pandas as pd

dataroot = 'data_resource/HiCervix'
clsname_map = {
    'Normal': 'NILM',
    'ECC': 'NILM',
    'RPC': 'NILM',
    'TRI': 'NILM',
    'MPC': 'NILM',
    'PG': 'NILM',
    'EMC': 'NILM',
    'Atrophy': 'NILM',
    'HSV': 'NILM',
    'CC': 'NILM',
    'HCG': 'NILM',
    'FUNGI': 'NILM',
    'ACTINO': 'NILM',
    'AGC-NOS': 'AGC',
    'AGC': 'AGC',
    'ADC': 'AGC',
    'AGC-FN': 'AGC',
    'AGC-ECC-NOS': 'AGC',
    'AGC-EMC-NOS': 'AGC',
    'ADC-ECC': 'AGC',
    'ADC-EMC': 'AGC',
    'ASC-US': 'ASC-US',
    'ASC-H': 'ASC-H',
    'LSIL': 'LSIL',
    'HSIL': 'HSIL',
    'SCC': 'HSIL',
}
clsnames = ['NILM', 'AGC', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL']

def main():
    for mode in ['train','val']:
        totallines = []
        df_data = pd.read_csv(f'{dataroot}/{mode}.csv')
        for row in tqdm(df_data.itertuples(index=False), total=len(df_data), ncols=80):
            clsname = clsname_map[row.class_name]
            clsid = clsnames.index(clsname)
            totallines.append(f'{row.image_name} {clsid}\n')
        with open(f'{dataroot}/{mode}.txt', 'w') as f:
            f.writelines(totallines)
            

if __name__ == "__main__":
    main()