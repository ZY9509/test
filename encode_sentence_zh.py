import torch
from transformers import BertTokenizer, BertModel

MODEL_NAME = "hfl/chinese-bert-wwm-ext"  # 中文BERT


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME)
    device = get_device()
    model.to(device)
    model.eval()
    return tokenizer, model, device


def encode_sentence(text, tokenizer, model, device, pooling="cls"):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=False,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        last_hidden_state = outputs.last_hidden_state  # [1, L, H]

    if pooling == "cls":
        sent_emb = last_hidden_state[:, 0, :]          # [1, H]
    else:  # mean pooling
        attention_mask = encoded["attention_mask"]
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1)
        sent_emb = summed / counts

    return sent_emb.squeeze(0).cpu()


if __name__ == "__main__":
    tokenizer, model, device = load_model_and_tokenizer()
    print("Device:", device)

    text = "这辆车的外观很好看，但是油耗有点高。"
    emb_cls = encode_sentence(text, tokenizer, model, device, pooling="cls")
    emb_mean = encode_sentence(text, tokenizer, model, device, pooling="mean")

    print("CLS embedding shape:", emb_cls.shape)
    print("Mean embedding shape:", emb_mean.shape)
    print("CLS first 5 dims:", emb_cls[:5])
    print("Mean first 5 dims:", emb_mean[:5])
