import pandas as pd

from glob import glob
from gpr.test.symbolic_utils import get_sym_model, read_file



if __name__=='__main__':
    datadir = 'pmlb/datasets/'
    frames = []
    for i, f in enumerate(glob(datadir+'/*/*.tsv.gz')):
        group = 'feynman' if 'feynman' in f else 'strogatz' if 'strogatz' in f else 'black-box'
        if group in ['black-box', 'strogatz']: # skip the black-box functions
            continue

        df = pd.read_csv(f, sep='\t')
        features, labels, feature_names = read_file(
            f, use_dataframe=True, sep='\t'
        )
        expression = get_sym_model(f, return_str=False) if group != 'black-box' else ''
        frames.append(dict(
            name=f.split('/')[-1][:-7],
            nsamples = df.shape[0],
            nfeatures = df.shape[1],
            npoints = df.shape[0]*df.shape[1],
            Group=group,
            Features = features,
            Labels = labels,
            Feature_names = feature_names,
            Equation = expression
        ))
        if i == 10:
            break # just so it does not iterate over all the 419 datasets

    df = pd.DataFrame.from_records(frames)

    print(df)
