import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import scrapbook as sb
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Automated execution and analysis of all examples')
parser.add_argument('--execute', action="store_true", help="execute notebook")
parser.add_argument('--save', action="store_true", help="save state of notebook")
args = parser.parse_args()

dirs = [
    'training/logistic_regression'
]

df = pd.DataFrame()

for d in dirs:
    print(">> Running notebooks in directory: %s" % (d))
    for filename in os.listdir(d):
        if filename.endswith(".ipynb"):

            filename_with_path = os.path.join(d, filename)
            print(">> Found notebook: %s" % (filename_with_path))

            with open(filename_with_path) as f:
                nb = nbformat.read(f, as_version=4)

            if args.execute:
                print(">> Executing notebook: %s" % (filename_with_path))
                ep = ExecutePreprocessor(store_widget_state=False)
                ep.preprocess(nb, {'metadata': {'path': d}})
            else:
                print(">> Skipping execution.")

            if args.save:
                print(">> Saving state of notebook: %s" % (filename_with_path))
                with open(filename_with_path, 'w', encoding='utf-8') as f:
                    nbformat.write(nb, f)
            else:
                print(">> Skipping save.")

            print(">> Reading scraps from notebook: %s" % (filename_with_path))
            scraps = sb.read_notebook(nb).scraps
            res = pd.Series(scraps['result'].data, name=filename_with_path)
            df = df.append(res)
            print(res)


pd.set_option('display.max_columns', None)
print(df)
df.to_csv("benchmark.csv")
