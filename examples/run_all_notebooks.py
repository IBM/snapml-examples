# Copyright 2021 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
    'training/logistic_regression',
    'training/decision_tree'
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
            #print(res)


print(df[['dataset','model','speed_up','score_diff']])
df.to_csv("benchmark.csv")
