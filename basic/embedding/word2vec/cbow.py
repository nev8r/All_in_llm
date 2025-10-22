import os
import re
import time
import jieba
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from gensim.models import Word2Vec
import swanlab # type: ignore


# ===============================
# 数据预处理
# ===============================
def clean_text(text):
    """去除非中文字符"""
    return re.sub(r"[^\u4e00-\u9fa5]", "", str(text))


def tokenize_text(text):
    """单句分词"""
    text = clean_text(text)
    if not text:
        return []
    return list(jieba.cut(text))


def load_douban_data(path, n_process=None):
    """多进程 jieba 分词"""
    df = pd.read_csv(path)
    texts = df["Comment"].dropna().tolist()

    print(f"开始多进程分词，共 {len(texts)} 条评论 ...")
    n_process = n_process or max(cpu_count() - 1, 1)

    with Pool(n_process) as pool:
        sentences = list(
            tqdm(pool.imap(tokenize_text, texts), total=len(texts), desc="分词中")
        )

    sentences = [s for s in sentences if s]  # 去掉空句
    print(f"✅ 分词完成，有效样本数: {len(sentences)} 条")
    return sentences


# ===============================
# 训练函数（记录 loss）
# ===============================
def train_word2vec(sentences, hs=1, negative=0, save_path="model.model", project="douban-word2vec"):
    """训练 Word2Vec 并记录每轮 loss"""
    # 确保保存目录存在
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    swanlab.init(project=project, run_name=f"{'HS' if hs else 'NS'}_training")

    model = Word2Vec(
        vector_size=100,
        window=5,
        min_count=5,
        sg=0,  # CBOW
        hs=hs,
        negative=negative,
        workers=8,
        compute_loss=True
    )

    model.build_vocab(sentences)
    print(f"词汇表大小: {len(model.wv)}")

    start = time.time()
    # 初始损失值
    model.train(sentences, total_examples=len(sentences), epochs=1)
    
    for epoch in range(5):  # 已经训练了1轮，再训练4轮总共5轮
        model.train(sentences, total_examples=len(sentences), epochs=1, compute_loss=True)
        loss = model.get_latest_training_loss()
        epoch_loss = loss / len(sentences)

        print(f"Epoch {epoch + 1}: loss = {epoch_loss:.4f}")  # 从2开始计数，因为已经有1轮
        swanlab.log({"epoch": epoch + 1, "loss": epoch_loss})

    elapsed = time.time() - start
    model.save(save_path)
    print(f"✅ 模型保存至 {save_path}，训练耗时: {elapsed:.2f} 秒")

    swanlab.finish()
    return model, elapsed


# ===============================
# 主函数
# ===============================
if __name__ == "__main__":
    data_path = "/root/All_in_llm/basic/embedding/word2vec/data/DMSC.csv"

    # ====== 1. 加载 & 分词 ======
    sentences = load_douban_data(data_path)

    # ====== 2. 模型 A: Hierarchical Softmax ======
    print("\n=== 模型 A: Hierarchical Softmax ===")
    model_hs, hs_time = train_word2vec(
        sentences, 
        hs=1, 
        negative=0, 
        save_path="/root/All_in_llm/basic/embedding/cbow_models/word2vec_hs_jieba.model"
    )

    # ====== 3. 模型 B: Negative Sampling ======
    print("\n=== 模型 B: Negative Sampling ===")
    model_ns, ns_time = train_word2vec(
        sentences, 
        hs=0, 
        negative=10, 
        save_path="/root/All_in_llm/basic/embedding/cbow_models/word2vec_ns_jieba.model"
    )

    # ====== 4. 性能对比 ======
    print("\n=== 性能对比 ===")
    print(f"Hierarchical Softmax 训练时间: {hs_time:.2f} s")
    print(f"Negative Sampling 训练时间: {ns_time:.2f} s")

    # ====== 5. 相似词展示 ======
    for word in ["电影", "演技", "剧情"]:
        if word in model_hs.wv.key_to_index and word in model_ns.wv.key_to_index:
            print(f"\n【{word}】相似词（HS）:")
            print(model_hs.wv.most_similar(word, topn=5))
            print(f"【{word}】相似词（NS）:")
            print(model_ns.wv.most_similar(word, topn=5))
        else:
            print(f"\n⚠️ 词汇【{word}】未出现在词表中（min_count=5 可能过滤掉了）")
