'''
* Copyright (c) 2023 TECNALIA <esther.villar@tecnalia.com;eneko.osaba@tecnalia.com;sebastian.vidal@tecnalia.com>
*
* This file is free software: you may copy, redistribute and/or modify it
* under the terms of the GNU General Public License as published by the
* Free Software Foundation, either version 3.
*
* This file is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*
* This file incorporates work covered by the following copyright and
* permission notice:
*
*   MIT License
*
*   Copyright (c) 2021 Alessio Falai
*
*   Permission is hereby granted, free of charge, to any person obtaining a copy
*   of this software and associated documentation files (the "Software"), to deal
*   in the Software without restriction, including without limitation the rights
*   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
*   copies of the Software, and to permit persons to whom the Software is
*   furnished to do so, subject to the following conditions:
*
*   The above copyright notice and this permission notice shall be included in all
*   copies or substantial portions of the Software.
*
*   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
*   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
*   SOFTWARE.
'''

import warnings
warnings.filterwarnings("ignore")

import os
import itertools
import numpy as np
import pandas as pd
import pickle as pkl
from tabulate import tabulate

'''
IMPORTANT NOTE TO USERS: The creation of positive affinities and incompatibilities is made at random considering the 
numbers introduced in parameters 'num_positive_affinities' and 'num_incompatibilities'. Please, be aware that if the 
number of categories chosen for generating an instance is not large enough for building all the asked positive 
affinities and incompatibilities combinations, the code will arise an error.

For this reason, try to balance the number of categories ('num_categories'), the number of items ('num_items') and the 
set of positive affinities ('num_positive_affinities') and incompatibilities ('num_incompatibilities').
'''

class ProductDataset:
    """
    Construct a dataset of rectangular products,
    given minimum and maximum values for each dimension

    Samir Elhedhli, Fatma Gzara, Burak Yildiz,
    "Three-Dimensional Bin Packing and Mixed-Case Palletization",
    INFORMS Journal on Optimization, 2019.
    """

    def __init__(
        self,
        products_path,
        num_categories,
        min_width,
        max_width,
        min_depth,
        max_depth,
        min_height,
        max_height,
        min_weight,
        max_weight,
        force_overload=False,
    ):
        self.products_path = products_path
        self.num_categories = num_categories
        self.min_width, self.max_width = min_width, max_width
        self.min_depth, self.max_depth = min_depth, max_depth
        self.min_height, self.max_height = min_height, max_height
        self.min_weight, self.max_weight = min_weight, max_weight
        self.products = self._load_products(force_overload)

    def _load_products(self, force_overload=False):
        """
        Either load the products from the given path or
        generate the dataset (if forced or if the path does not exist)
        """
        if not os.path.exists(self.products_path) or force_overload:
            products = self._gen_products()
            products.to_pickle(self.products_path)
        else:
            products = pd.read_pickle(self.products_path)
        return products

    def _gen_products(self):
        """
        Generate a sample of products, by reproducing distributions
        reported on the cited paper
        """
        # Define ratios and volumes as specified in the paper
        dw_ratios = np.random.normal(loc=0.695, scale=0.118, size=(self.num_categories, 1))
        hw_ratios = np.random.lognormal(mean=-0.654, sigma=0.453, size=(self.num_categories, 1))
        volumes = np.random.lognormal(mean=2.568, sigma=0.705, size=(self.num_categories, 1)) * 1e6

        # Generate each dimension separately
        widths = np.clip(
            np.power(volumes / (dw_ratios * hw_ratios), 1 / 3),
            self.min_width,
            self.max_width,
        )
        depths = np.clip(widths * dw_ratios, self.min_depth, self.max_depth)
        heights = np.clip(widths * hw_ratios, self.min_height, self.max_height)
        weights = np.clip(
            np.random.lognormal(mean=2, sigma=2, size=(self.num_categories, 1)),
            self.min_weight,
            self.max_weight,
        )

        # Repeat products with the given frequency of occurrence
        dims = np.concatenate((widths, depths, heights, weights), axis=1).astype(int)
        frequencies = np.ceil(
            np.random.lognormal(mean=0.544, sigma=0.658, size=(1,)) # harcoded size=(1,) to keep product as categories (distinct templates of items)
        ).astype(int)
        dims = np.repeat(dims, frequencies, axis=0)
        indexes = np.arange(0, len(dims), 1)
        dims = dims[np.random.choice(indexes, size=self.num_categories)]

        # Create a DataFrame with the generated info
        df = pd.DataFrame(dims, columns=["width", "depth", "height", "weight"])
        df["volume"] = df.width * df.depth * df.height
        return df

    def get_mean_std_volumes(self):
        """
        Randomly sample (in a uniform way) from each volume category, with
        the specified sizes, in order to obtain mean and standard deviation
        statistics of the dataset used in the paper
        """
        category_one = np.random.uniform(low=2.72, high=12.04, size=72037)
        category_two = np.random.uniform(low=12.05, high=20.23, size=55436)
        category_three = np.random.uniform(low=20.28, high=32.42, size=26254)
        category_four = np.random.uniform(low=32.44, high=54.08, size=9304)
        category_five = np.random.uniform(low=54.31, high=100.21, size=3376)
        volumes = np.concatenate(
            (category_one, category_two, category_three, category_four, category_five)
        )
        log_volumes = np.log(volumes)
        return log_volumes.mean(), log_volumes.std()


using_dataset = False
if using_dataset:
    products_path = './input/Elhedhli_dataset/products.pkl'
    force_overload = False
else:
    products_path = './input/Elhedhli_dataset/newproducts.pkl'
    force_overload = True

########################### Instance definition
using_dataset = False
num_bins = 2                              # Number of bins n
bins_dims = (1500,1500,1500)              # Bin dimensions: (L, W, H)
CoM = (bins_dims[0]//2, bins_dims[1]//2)  # Center of mass: tuple of values or None
max_bin_capacity = None                   # Maximum capacity of each bin M
mass_ratio = None                         # Mass ratio: float greater than 1 or None
num_incompatibilities = 3                 # Number of incompatibilities: int >= 0
num_positive_affinities = 2               # Number of compatibilities: int >= 0

num_categories = 15               # Number of categories to play around
min_width, max_width = 0, 1000    # Min and max value of w_i for each case
min_depth, max_depth = 0, 1000    # Min and max value of l_i for each case
min_height, max_height = 0, 1000  # Min and max value of h_i for each case
min_weight, max_weight = 20, 50   # Min and max value of mu_i for each case
num_items = 50                    # Number of case ids
#################################################

force_overload = force_overload
random_state = 1                  # Seed for reproducibility
instance = ProductDataset(
    products_path=products_path,
    num_categories=num_categories,
    min_width=min_width,
    max_width=max_width,
    min_depth=min_depth,
    max_depth=max_depth,
    min_height=min_height,
    max_height=max_height,
    min_weight=min_weight,
    max_weight=max_weight,
    force_overload=force_overload,
    )

products_df = instance.products
products_df = products_df.sample(frac=1).reset_index(drop=True)
items_df = products_df.sample(n=num_items, replace=True, random_state=1) # if num_product is too large, it wil be unlikely to get replicates of items
items_df = items_df.loc[:,['width', 'depth', 'height', 'weight']]
items_df.rename({'depth': 'length'}, axis=1, inplace=True)
items_df.index = items_df.index.rename('id')
items_df = items_df.groupby(['width', 'length', 'height', 'weight']).size().reset_index(name='quantity')

if using_dataset:
    ## resize and remove decimals item
    items_df[['width', 'length', 'height']] = items_df[['width', 'length', 'height']].div(10)
    items_df = items_df.astype(int)

# Print input data file
fname = './input/3dBPP_test_2.txt'
with open(fname, 'w') as f:
    f.write('# Max num of bins : ' + str(num_bins) + '\n')
    f.write('# Bin dimensions (L * W * H): ' + str(bins_dims).replace(', ',',') + '\n')

    if max_bin_capacity is None:
        f.write('# Max weight: \n')
    else:
        f.write('# Max weight: ' + str(max_bin_capacity) + '\n')

    if mass_ratio is None:
        mass_ratio = ''
        f.write('# Relative Pos: ' + str(mass_ratio) + '\n')
    else:
        relations=[]
        for i in range(len(items_df.weight)):
            for j in range(len(items_df.weight)):
                if i!=j:
                    if items_df.weight[j]/items_df.weight[i]>mass_ratio:
                        relations.append((i,j))
        dictionary = {
            6:relations
        }
        f.write('# Relative Pos: ' + str(dictionary) + '\n')

    if num_incompatibilities + num_positive_affinities == 0:
        incomp, comp = '', ''
    else:
        idx = items_df.index.values
        combs = list(itertools.combinations(idx, 2))
        sel_tuples = np.random.choice(range(len(combs)), num_incompatibilities+num_positive_affinities, replace=False)
        if num_incompatibilities != 0:
            incomp = str(sorted(str(combs[t]).replace(', ',',') for t in sel_tuples[:num_incompatibilities])).replace("'",'')
        else:
            incomp = ''
        if num_positive_affinities != 0:
            comp = str(sorted(str(combs[t]).replace(', ',',') for t in sel_tuples[num_incompatibilities:num_incompatibilities+num_positive_affinities])).replace("'",'')
        else:
            comp = ''
    f.write('# Incompatibilities: ' + incomp.replace('[','').replace(']','').replace(',',', ') + '\n')
    f.write('# Positive Affinities: ' + comp.replace('[','').replace(']','').replace(',',', ') + '\n')
    if CoM is None:
        CoM = ''
    f.write('# Center of mass: ' + str(CoM) + '\n')

    f.write('\n')

    # Print table with input data
    header = ['id', 'quantity', 'length', 'width', 'height', 'weight']
    case_info = [[i,
                  items_df['quantity'].values[i],  # Plug in a random quantity for each id from [1,10)
                  items_df['length'].values[i],
                  items_df['width'].values[i],
                  items_df['height'].values[i],
                  items_df['weight'].values[i]] for i in items_df.index.values]
    f.write(tabulate([header, *list(case_info)], headers='firstrow'))

    # f.write('  id    quantity    length    width    height    weight' + '\n')
    # f.write('----  ----------  --------  -------  --------  --------' + '\n')

    # for index, item in items_df.iterrows():
    #     f.write(' '*3 + str(index) + ' '*10 + str(item['quantity']) + ' '*9 + str(item['length']) +
    #              ' '*8 + str(item['width']) + ' '*9 + str(item['height']) + ' '*9 + str(item['weight']) + '\n')