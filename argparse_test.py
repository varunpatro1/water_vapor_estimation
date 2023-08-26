import numpy as np
import argparse


def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('train_fname', help = 'path to train data file')
    args = parser.parse_args()

    
    data = np.load(args.train_fname)
    refls = data['output_rfl']
    print(refls.shape)


if __name__ == "__main__":
    main()