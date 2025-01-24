import torch
from tqdm import tqdm as tqdm
import copy
import numpy as np

def encoder_inference_to_feature(model, tokenizer, texts):
    with torch.no_grad():
        representations = []
        for query in texts:
            tknzd_data = tokenizer(query, return_tensors="pt").input_ids.to(model.device)
            result = model(tknzd_data, output_hidden_states = True)
            representations.append(result.pooler_output[-1].detach().to(torch.float))
        return representations

def feature_merge(text_feature, label_feature, overlap_ratio = 1): # 1D torch.tensor
    ret = copy.deepcopy(label_feature)
    overlap_dims = int(label_feature.shape[0] * overlap_ratio)
    added_features = torch.concat([text_feature[:overlap_dims], torch.zeros_like(text_feature[overlap_dims:])])
    concating_features = text_feature[overlap_dims:]
    ret += added_features
    return torch.concat([ret, concating_features])

def attention(queries, keys, values):
    attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
    # attention_probs = torch.nn.functional.softmax(attention_scores, dim = -1)
    max_loca = torch.argmax(attention_scores, dim = -1)
    # return torch.matmul(attention_probs, values)
    return values[max_loca]

def decode(feature_after_attention, label_features):
    cos = torch.nn.CosineSimilarity(dim = 0)
    cos_res = []
    for vector in label_features:
        cos_res.append(cos(feature_after_attention, vector).cpu())
    return np.argmax(cos_res)

def renormlized_ICL(encoder_model, tokenizer, demonstration_text, demonstration_labels, queries, label_spaces, overlap_ratio = 1, remain_magnitude = 0.15):
    ret = []
    label_features = encoder_inference_to_feature(encoder_model, tokenizer, label_spaces)
    for i in range(len(label_features)):
        label_features[i] = feature_merge(torch.zeros_like(label_features[i]), label_features[i], overlap_ratio)
    for i in tqdm(range(len(queries))):
        query = queries[i]
        query_features = encoder_inference_to_feature(encoder_model, tokenizer, query)[0]
        demonstration_features = encoder_inference_to_feature(encoder_model, tokenizer, demonstration_text[i])
        demonstration_label_features = encoder_inference_to_feature(encoder_model, tokenizer, demonstration_labels[i])
        merged_features = []
        for j in range(len(demonstration_features)):
            merged_features.append(feature_merge(demonstration_features[j], remain_magnitude * demonstration_label_features[j], overlap_ratio))
        query_features = feature_merge(query_features, torch.zeros_like(query_features), overlap_ratio)
        merged_features = torch.stack(merged_features)

        feature_after_attention = attention(query_features, merged_features, merged_features)
        label_index = decode(feature_after_attention, label_features)
        ret.append(label_index)
    return ret
