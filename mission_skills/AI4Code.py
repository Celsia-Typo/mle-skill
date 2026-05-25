import json
import logging
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
from xgboost import XGBRanker
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from bisect import bisect

# 配置日志和显示
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AI4Code")
pd.options.display.width = 180
pd.options.display.max_colwidth = 120

# --- 配置参数 ---
DATA_DIR = Path('../input/AI4Code')
NUM_TRAIN = 10000
NVALID = 0.1  # 验证集比例

# --- 数据读取工具 ---

def read_notebook(path):
    """读取单个 JSON 笔记本文件"""
    return (
        pd.read_json(path, dtype={'cell_type': 'category', 'source': 'str'})
        .assign(id=path.stem)
        .rename_axis('cell_id')
    )

def load_all_notebooks(paths, desc="Loading NBs"):
    """批量加载笔记本并合并为 DataFrame"""
    notebooks = [read_notebook(path) for path in tqdm(paths, desc=desc)]
    df = (
        pd.concat(notebooks)
        .set_index('id', append=True)
        .swaplevel()
        .sort_index(level='id', sort_remaining=False)
    )
    return df

# --- 指标评估工具 ---

def count_inversions(a):
    inversions = 0
    sorted_so_far = []
    for i, u in enumerate(a):
        j = bisect(sorted_so_far, u)
        inversions += i - j
        sorted_so_far.insert(j, u)
    return inversions

def kendall_tau(ground_truth, predictions):
    """计算 Kendall tau 相关系数"""
    total_inversions = 0
    total_2max = 0 
    for gt, pred in zip(ground_truth, predictions):
        ranks = [gt.index(x) for x in pred]
        total_inversions += count_inversions(ranks)
        n = len(gt)
        total_2max += n * (n - 1)
    return 1 - 4 * total_inversions / total_2max

# --- 主流程 ---

def main():
    # 1. 加载训练数据
    paths_train = list((DATA_DIR / 'train').glob('*.json'))[:NUM_TRAIN]
    df = load_all_notebooks(paths_train, desc="Train NBs")
    
    # 读取正确排序标签
    df_orders = pd.read_csv(DATA_DIR / 'train_orders.csv', index_col='id', squeeze=True).str.split()
    
    # 2. 计算训练集的 Rank（目标变量）
    df_orders_ = df_orders.to_frame().join(
        df.reset_index('cell_id').groupby('id')['cell_id'].apply(list),
        how='right',
    )
    
    def get_ranks(base, derived):
        return [base.index(d) for d in derived]

    ranks = {}
    for id_, cell_order, cell_id in df_orders_.itertuples():
        ranks[id_] = {'cell_id': cell_id, 'rank': get_ranks(cell_order, cell_id)}

    df_ranks = (
        pd.DataFrame.from_dict(ranks, orient='index')
        .rename_axis('id')
        .apply(pd.Series.explode)
        .set_index('cell_id', append=True)
    )

    # 3. 划分验证集 (基于 Ancestor 避免泄露)
    df_ancestors = pd.read_csv(DATA_DIR / 'train_ancestors.csv', index_col='id')
    splitter = GroupShuffleSplit(n_splits=1, test_size=NVALID, random_state=0)
    
    ids = df.index.unique('id')
    ancestors = df_ancestors.loc[ids, 'ancestor_id']
    ids_train_idx, ids_valid_idx = next(splitter.split(ids, groups=ancestors))
    ids_train, ids_valid = ids[ids_train_idx], ids[ids_valid_idx]

    df_train = df.loc[ids_train, :]
    df_valid = df.loc[ids_valid, :]

    # 4. 特征工程 (TF-IDF + Code Cell Enumeration)
    tfidf = TfidfVectorizer(min_df=0.01)
    X_train_text = tfidf.fit_transform(df_train['source'].astype(str))
    
    # 辅助特征：代码单元格的原始相对顺序
    def add_code_order_feature(df_part, text_feat):
        return sparse.hstack((
            text_feat,
            np.where(
                df_part['cell_type'] == 'code',
                df_part.groupby(['id', 'cell_type']).cumcount().to_numpy() + 1,
                0,
            ).reshape(-1, 1)
        ))

    X_train = add_code_order_feature(df_train, X_train_text)
    y_train = df_ranks.loc[ids_train].to_numpy()
    groups = df_ranks.loc[ids_train].groupby('id').size().to_numpy()

    # 5. 模型训练
    logger.info("Training XGBRanker model...")
    model = XGBRanker(
        min_child_weight=10,
        subsample=0.5,
        tree_method='hist', # 2026年建议根据显卡情况使用 'gpu_hist'
    )
    model.fit(X_train, y_train, group=groups)

    # 6. 验证集评估
    X_valid_text = tfidf.transform(df_valid['source'].astype(str))
    X_valid = add_code_order_feature(df_valid, X_valid_text)
    y_true_valid = df_orders.loc[ids_valid]

    # 预测并转化为排序列表
    y_pred_df = pd.DataFrame({'rank': model.predict(X_valid)}, index=df_valid.index)
    y_pred_valid = (
        y_pred_df.sort_values(['id', 'rank'])
        .reset_index('cell_id')
        .groupby('id')['cell_id'].apply(list)
    )

    score = kendall_tau(y_true_valid, y_pred_valid)
    logger.info(f"Validation Kendall Tau Score: {score:.4f}")

    # 7. 测试集预测与提交
    paths_test = list((DATA_DIR / 'test').glob('*.json'))
    if paths_test:
        df_test = load_all_notebooks(paths_test, desc="Test NBs")
        X_test_text = tfidf.transform(df_test['source'].astype(str))
        X_test = add_code_order_feature(df_test, X_test_text)
        
        y_infer = pd.DataFrame({'rank': model.predict(X_test)}, index=df_test.index)
        y_infer = (
            y_infer.sort_values(['id', 'rank'])
            .reset_index('cell_id')
            .groupby('id')['cell_id'].apply(list)
            .apply(' '.join)
            .rename_axis('id')
            .rename('cell_order')
        )
        
        y_infer.to_csv('submission.csv')
        logger.info("Submission file saved as submission.csv")

if __name__ == "__main__":
    main()