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

import platform
import os
import psutil
import json
import snapml
import sklearn
import xgboost
import lightgbm

def get_environment():
    return {
        'platform': platform.platform(),
        'cpu_count': os.cpu_count(),
        'cpu_freq_min': psutil.cpu_freq().min,
        'cpu_freq_max': psutil.cpu_freq().max,
        'total_memory': psutil.virtual_memory().total/1024/1024/1024,
        'snapml_version': snapml.__version__,
        'sklearn_version': sklearn.__version__,
        'xgboost_version': xgboost.__version__,
        'lightgbm_version': lightgbm.__version__,
    }