import torch
import torch.nn.functional as F
import nltk


def replace_strategy(self, sentences, token_ids):
    batch_size, seq_len = token_ids.shape
    device = token_ids.device

    # 获取原始嵌入 (B, L, D)
    with torch.no_grad():
        token_embeddings = self.text_encoder.token_embedding(token_ids)

        # 处理每个句子
    new_embeddings = token_embeddings.clone()
    for b in range(batch_size):
        tokens = nltk.word_tokenize(sentences[b])
        pos_tags = nltk.pos_tag(tokens)  # [(word, tag), ...]

        for l in range(min(len(tokens), seq_len)):  # 防越界
            word, tag = pos_tags[l]
            pos_type = self.pos_dict.get(tag[:2], None)

            # 仅处理动词/形容词
            if pos_type in ['verb', 'adjective']:
                # 获取候选词索引
                candidate_indices = [
                    i for i, w in enumerate(self.vocab)
                    if w in self.pos_tags[pos_type]
                ]
                if not candidate_indices:
                    continue

                # 计算相似度
                word_emb = token_embeddings[b, l]  # (D,)
                candidate_emb = self.text_encoder.token_embedding(
                    torch.tensor(candidate_indices, device=device)
                )  # (C, D)

                sim_scores = F.cosine_similarity(
                    word_emb.unsqueeze(0), candidate_emb, dim=-1
                )  # (C,)

                # 选择满足阈值的最相似词
                mask = sim_scores < self.gamma
                if mask.any():
                    selected = candidate_indices[sim_scores[mask].argmax()]
                    new_embeddings[b, l] = self.text_encoder.token_embedding(
                        torch.tensor(selected, device=device)
                    )

    return new_embeddings


def mask_strategy(self, token_ids):
    # 获取原始嵌入 (B, L, D)
    token_embeddings = self.text_encoder.token_embedding(token_ids)

    # 动态获取MASK ID
    mask_id = self.vocab.index('[MASK]') if '[MASK]' in self.vocab else -1
    if mask_id == -1:
        return token_embeddings  # 无MASK则跳过

    # 计算掩码概率 (B, L)
    norms = torch.norm(token_embeddings, dim=-1)  # (B, L)
    probs = 1 - torch.sigmoid(norms)  # 更合理的概率分布

    # 生成掩码
    mask = torch.bernoulli(probs).bool()  # (B, L)
    mask_emb = self.text_encoder.token_embedding(
        torch.tensor(mask_id, device=token_ids.device)
    )  # (D,)

    # 应用掩码
    masked_embeddings = torch.where(
        mask.unsqueeze(-1),
        mask_emb.expand_as(token_embeddings),
        token_embeddings
    )
    return masked_embeddings


def modify_strategy(self, sentences, token_ids):
    batch_size, seq_len = token_ids.shape
    device = token_ids.device

    # 计算TF-IDF权重 (B, V)
    with torch.no_grad():
        # 生成词频矩阵 (B, V)
        count_matrix = torch.zeros(batch_size, len(self.vocab), device=device)
        for b, sent in enumerate(sentences):
            tokens = nltk.word_tokenize(sent)
            for word in tokens:
                if word in self.vocab:
                    count_matrix[b, self.vocab.index(word)] += 1
        # 应用预计算IDF
        tfidf_weights = count_matrix * self.idf_matrix.to(device)  # (B, V)

    # 获取词嵌入矩阵 (V, D)
    vocab_emb = self.text_encoder.token_embedding(
        torch.arange(len(self.vocab), device=device)
    )  # (V, D)

    # 计算相似度矩阵 (V, V)
    sim_matrix = 1 - F.cosine_similarity(
        vocab_emb.unsqueeze(1), vocab_emb.unsqueeze(0), dim=-1
    )  # 余弦距离

    # 修改嵌入
    modified_embeddings = self.text_encoder.token_embedding(token_ids).clone()
    for b, sent in enumerate(sentences):
        tokens = nltk.word_tokenize(sent)
        for l in range(min(len(tokens), seq_len)):
            word = tokens[l]
            if word not in self.vocab:
                continue

            word_idx = self.vocab.index(word)
            # 获取top-k最近邻 (排除自身)
            distances = sim_matrix[word_idx]  # (V,)
            _, indices = torch.topk(distances, self.k + 1, largest=False)
            neighbor_ids = indices[indices != word_idx][:self.k]  # (k,)

            # 计算综合得分
            tfidf_scores = tfidf_weights[b, neighbor_ids]  # (k,)
            sim_scores = 1 - distances[neighbor_ids]  # (k,)
            combined = sim_scores - self.lambda_ * tfidf_scores

            # 选择最佳替换
            best_idx = neighbor_ids[combined.argmax()]
            modified_embeddings[b, l] = vocab_emb[best_idx]

    return modified_embeddings





