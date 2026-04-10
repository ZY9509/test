import torch
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

MODEL_NAME = "hfl/chinese-bert-wwm-ext"
MAX_LENGTH = 128
BATCH_SIZE = 16


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME)
    device = get_device()
    model.to(device)
    model.eval()
    return tokenizer, model, device


def encode_batch(texts, tokenizer, model, device, pooling="mean"):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc)
        last_hidden_state = outputs.last_hidden_state  # [B, L, H]

    if pooling == "cls":
        sent_embed = last_hidden_state[:, 0, :]        # [B, H]
    else:
        attention_mask = enc["attention_mask"]
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1)
        sent_embed = summed / counts                   # [B, H]

    return sent_embed.cpu()


def main():
    tokenizer, model, device = load_model_and_tokenizer()
    print("Using device:", device)

    # 假设中文评论保存在 data_zh.csv，且有一列叫 text
    df = pd.read_csv("data_zh.csv", encoding="gbk")
    texts = df["text"].astype(str).tolist()

    all_embeddings = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i + BATCH_SIZE]
        emb = encode_batch(batch_texts, tokenizer, model, device, pooling="mean")
        all_embeddings.append(emb)

    all_embeddings = torch.cat(all_embeddings, dim=0)  # [N, 768]
    print("All embeddings shape:", all_embeddings.shape)

    torch.save(all_embeddings, "text_embeddings_zh.pt")

    import numpy as np
    np.save("text_embeddings_zh.npy", all_embeddings.numpy())


if __name__ == "__main__":
    main()
