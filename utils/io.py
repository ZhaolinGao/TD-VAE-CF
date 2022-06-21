import datetime
import json
import pickle
import pandas as pd
import time
import scipy.sparse as sparse
from scipy.sparse import csr_matrix, save_npz, load_npz
import numpy as np
import os
from os import listdir
from os.path import isfile, join

def save_dataframe_csv(df, path, name):
    if not os.path.exists(path):
        os.makedirs(path)

    csv_filename = "{:s}.csv".format(name)
    df.to_csv(path + csv_filename, index=False)
    print('Dataframe Saved Successfully: ', path + csv_filename)


def load_dataframe_csv(path, name):
    return pd.read_csv(path + name)


def save_numpy_csr(matrix, path, model):
    save_npz('{}{}'.format(path, model), matrix)


def load_numpy_csr(path, name):
    return load_npz(path + name).tocsr()


def save_numpy(matrix, path, model):
    np.save('{}{}'.format(path, model), matrix)


# def load_numpy(path, name):
#     return np.load(path + name)

def load_numpy(path, name):
    return load_npz(path + name).tocsr()


def saveDictToJson(dictionary, path, fileName):  # , trainOrTest='train'
    json_fileName = "{:s}.json".format(fileName)
    # if (trainOrTest == 'train'):
    #     json.dump(dictionary, open(path + json_fileName, 'w'))
    # else:
    json.dump(dictionary, open(path + json_fileName, 'w'))

def loadTextJson(path, fileName):
    file = path + fileName + '.txt'
    with open(file, "r") as fp:
        b = json.load(fp)
    return b

def loadDict(file_dir):  # trainOrTest='train'
    # json_fileName = "{:s}.json".format(fileName)
    # Read data from file:
    # if (trainOrTest == 'train'):
    #     dataDict = json.load(open(path + fileName))
    # else:
    dataDict = json.load(open(file_dir))
    return dataDict


def get_yelp_df(filename='Export_CleanedReview.json', sampling=True, top_user_num=7000,
                top_item_num=5000):
    """
    Get the pandas dataframe
    Sampling only the top users/items by density
    Implicit representation applies
    """
    with open(filename, 'r') as f:
        data = f.readlines()
        data = list(map(json.loads, data))

    data = data[0]
    # Get all the data from the dggeata file
    df = pd.DataFrame(data)

    df.rename(columns={'stars': 'review_stars', 'text': 'review_text', 'cool': 'review_cool',
                       'funny': 'review_funny', 'useful': 'review_useful'},
              inplace=True)

    df['business_num_id'] = df.business_id.astype('category'). \
        cat.rename_categories(range(0, df.business_id.nunique()))
    df['business_num_id'] = df['business_num_id'].astype('int')

    df['user_num_id'] = df.user_id.astype('category'). \
        cat.rename_categories(range(0, df.user_id.nunique()))
    df['user_num_id'] = df['user_num_id'].astype('int')

    df['timestamp'] = df['date'].apply(date_to_timestamp)

    if sampling:
        df = filter_yelp_df(df, top_user_num=top_user_num, top_item_num=top_item_num)
        # Refresh num id
        df['business_num_id'] = df.business_id.astype('category'). \
            cat.rename_categories(range(0, df.business_id.nunique()))
        df['business_num_id'] = df['business_num_id'].astype('int')

        df['user_num_id'] = df.user_id.astype('category'). \
            cat.rename_categories(range(0, df.user_id.nunique()))
        df['user_num_id'] = df['user_num_id'].astype('int')

    df = df.reset_index(drop=True)

    return df

# implemented code to depopularize items
def filter_yelp_df(df, top_user_num=7000, top_item_num=5000):
    # total_items = len(df.business_num_id.unique())
    # Getting the reviews where starts are above 3
    df_implicit = df[(df['review_stars'] > 3) & (df['ghost'] == False) & (df['user_id'] != 'CxDOIDnH8gp9KXzpBHJYXw')]
    frequent_user_id = df_implicit['user_num_id'].value_counts().head(top_user_num).index.values
    frequent_item_id = df_implicit['business_num_id'].value_counts().head(top_item_num).index.values
    # frequent_item_id = np.random.choice(total_items, top_item_num, replace=False)
    return df.loc[(df['user_num_id'].isin(frequent_user_id)) & (df['business_num_id'].isin(frequent_item_id))]


def date_to_timestamp(date):
    dt = datetime.datetime.strptime(date, '%Y-%m-%d')
    return time.mktime(dt.timetuple())


def df_to_sparse(df, row_name='userId', col_name='movieId', value_name='rating',
                 shape=None):
    rows = df[row_name]
    cols = df[col_name]
    if value_name is not None:
        values = df[value_name]
    else:
        values = [1] * len(rows)

    return csr_matrix((values, (rows, cols)), shape=shape)

def get_file_names(folder_path, extension='.yml'):
    return [f for f in listdir(folder_path) if isfile(join(folder_path, f)) and f.endswith(extension)]

def write_file(folder_path, file_name, content, exe=False):
    full_path = folder_path+'/'+file_name
    with open(full_path, 'w') as the_file:
        the_file.write(content)

    if exe:
        st = os.stat(full_path)
        os.chmod(full_path, st.st_mode | stat.S_IEXEC)

"""

Matrix Generation

"""
def get_rating_timestamp_matrix(df):
    rating_matrix = df_to_sparse(df, row_name='user_num_id',
                                 col_name='business_num_id',
                                 value_name='review_stars',
                                 shape=None)

    timestamp_matrix = df_to_sparse(df, row_name='user_num_id',
                                    col_name='business_num_id',
                                    value_name='timestamp',
                                    shape=None)

    return rating_matrix, timestamp_matrix


def get_IC_matrix(df):
    lst = df.categories.values.tolist()
    cat = []
    for i in range(len(lst)):
        if lst[i] is None:
            print(i)
        cat.extend(lst[i].split(', '))

    unique_cat = set(cat)
    #     set categories id
    df_cat = pd.DataFrame(list(unique_cat), columns=["Categories"])
    df_cat['cat_id'] = df_cat.Categories.astype('category').cat.rename_categories(range(0, df_cat.Categories.nunique()))
    dict_cat = df_cat.set_index('Categories')['cat_id'].to_dict()

    df_I_C = pd.DataFrame(columns=['business_num_id', 'cat_id'])

    for i in range((df['business_num_id'].unique().shape)[0]):
        df_temp = df[df['business_num_id'] == i].iloc[:1]
        temp_lst = df_temp['categories'].to_list()[0].split(",")
        for j in range(len(temp_lst)):
            df_I_C = df_I_C.append({'business_num_id': i, 'cat_id': dict_cat[temp_lst[j].strip()]}, ignore_index=True)

    IC_Matrix = df_to_sparse(df_I_C, row_name='business_num_id',
                             col_name='cat_id',
                             value_name=None,
                             shape=None)
    return IC_Matrix, dict_cat


def getImplicitMatrix(sparseMatrix, threashold=0):
    temp_matrix = sparse.csr_matrix(sparseMatrix.shape)
    temp_matrix[(sparseMatrix > threashold).nonzero()] = 1
    return temp_matrix


def get_UC_Matrix(IC_Matrix, rtrain_implicit):
    U_C_matrix_explicit = rtrain_implicit * IC_Matrix
    U_C_matrix_implicit = getImplicitMatrix(U_C_matrix_explicit, 3)
    return U_C_matrix_explicit, U_C_matrix_implicit


def get_csr_matrix(df, rowname, colname, value=None, shape=None):
    row = df[rowname]
    col = df[colname]
    if value == None:
        value = [1] * len(row)
    return csr_matrix((value, (row, col)), shape=shape)


# get original dataframe, returns idx2item, keyphrase2idx, idx2keyphrase
def get_idx_mapping(df):
    idx_2_itemName = dict(zip(df.business_num_id, df.name))
    idx_2_itemId = dict(zip(df.business_num_id, df.business_id))
    itemidx_2_category = dict(zip(df.business_num_id, df.categories))
    return idx_2_itemName, idx_2_itemId, itemidx_2_category


def pickle_dump(file, path, file_name):
    file_name = path + file_name + '.pickle'
    with open(file_name, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(path, file_name):
    file_name = path + file_name + '.pickle'
    with open(file_name, 'rb') as handle:
        return pickle.load(handle)


# Create folders under data dir
# For the usage of storing use case data
def get_data_dir(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data_dirs = os.listdir(data_dir)
    if len(data_dirs) == 0:
        idx = 0
    else:
        idx_lis = []
        for d in data_dirs:
            try:
                current_idx = int(d.split('_')[0])
                idx_lis.append(current_idx)
            except:
                continue

        idx_list = sorted(idx_lis)
        idx = idx_list[-1] + 1

    cur_data_dir = '%d_%s' % (idx, time.strftime('%Y%m%d-%H%M'))
    full_data_dir = os.path.join(data_dir, cur_data_dir)
    if not os.path.exists(full_data_dir):
        os.mkdir(full_data_dir)

    return full_data_dir

"""
NOT USED
"""
def get_test_df(ratings_tr, tags_tr, tags_val):
    '''
    Remove user/item/tag which only exist in validation set(remove cold-start case)
    '''

    valid_user = ratings_tr['userId'].unique()
    valid_item = ratings_tr['itemId'].unique()
    valid_tag = tags_tr['tagId'].unique()

    tags_val = tags_val.loc[tags_val['userId'].isin(valid_user) &
                            tags_val['itemId'].isin(valid_item) &
                            tags_val['tagId'].isin(valid_tag)]

    return tags_val.groupby(['userId', 'itemId'])['tagId'].apply(list).reset_index(name='tagIds')
