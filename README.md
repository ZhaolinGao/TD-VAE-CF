# TD-VAE-CF

This repo covers the implementation for our paper:

Zhaolin Gao, Tianshu Shen, Zheda Mai, Mohamed Reda Bouadjenek, Isaac Waller, Ashton Anderson, Ron Bodkin, and Scott Sanner. "Mitigating the Filter Bubble while Maintaining Relevance: Targeted Diversification with VAE-based Recommender Systems" In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 22).

## Instructions

1. Download dataset from `https://drive.google.com/drive/folders/1S5QMSkweJmAbqZLom3V0gI_OvvsgS0hB?usp=sharing`.

2. Modify the `api_key` in line 96 of `main.py` to your api key on comet_ml.

3. Train and evaluate:
```
python main.py --data_name yelp_SIGIR --target veg_bbq --lamb LAMB_VALUE --std STD_VALUE
python main.py --data_name yelp_SIGIR --target fried_salad --lamb LAMB_VALUE --std STD_VALUE
python main.py --data_name reddit --target men_women --lamb LAMB_VALUE --std STD_VALUE
python main.py --data_name reddit --target rep_dem --lamb LAMB_VALUE --std STD_VALUE
```

## Citation

If you find this code useful in your research, please cite the following paper:

    @inproceedings{gao2022sigir,
      title={Mitigating the Filter Bubble while Maintaining Relevance: Targeted Diversification with VAE-based Recommender Systems},
      author={Zhaolin Gao, Tianshu Shen, Zheda Mai, Mohamed Reda Bouadjenek, Isaac Waller, Ashton Anderson, Ron Bodkin, Scott Sanner},
      booktitle={Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
      year={2022}
    }

## Credit

Reddit dataset is obtained using [[PushShift](https://github.com/pushshift/api)]
