import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
from model import build_model
from transformers import CLIPTokenizer
import nltk
from utils.checkpoint import Checkpointer

nltk.download('punkt')

hook_outputs = {}

def get_text_attn_weights_hook(name):

    def hook(module, input, output):
        # 获取 last_attn_weights 属性，需根据自定义模型调整
        attn_output_weights = getattr(module, 'last_attn_weights', None)
        if attn_output_weights is not None:
            hook_outputs[name] = attn_output_weights.detach()
            print(f"Hook captured attention weights for {name} with shape {attn_output_weights.shape}")
        else:
            print(f"Warning: Attention output weights for {name} are None.")
    return hook

def register_text_attn_weights_hooks(model, layers_to_hook):

    for i in layers_to_hook:
        try:
            # 假设文本Transformer路径为 model.base_model.transformer.resblocks[i]
            resblock = model.base_model.transformer.resblocks[i]
            resblock.register_forward_hook(get_text_attn_weights_hook(f'text_block_{i + 1}'))
            print(f"Registered hook for text_block_{i + 1}")
        except AttributeError:
            raise AttributeError(f"Text Layer index {i} does not exist in the model.")

def plot_text_attention_map(attn_weights, tokens, model_name, output_dir, layer_num, head_num, num_text_tokens):

    # 提取文本相关的attention部分
    attn_text = attn_weights[:num_text_tokens, :num_text_tokens]

    plt.figure(figsize=(12, 10))
    sns.heatmap(attn_text, xticklabels=tokens[:num_text_tokens], yticklabels=tokens[:num_text_tokens], cmap='viridis')
    plt.title(f'Text Attention Map - Layer {layer_num + 1} Head {head_num + 1}')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # 构建保存文件名
    output_file = os.path.join(output_dir, f"{model_name}_layer{layer_num +1}_head{head_num +1}_text_attention.png")
    plt.savefig(output_file)
    plt.close()

    del attn_weights

def plot_fusion_token_attention(attn_weights, tokens, model_name, output_dir, layer_num, head_num, fusion_idx, num_text_tokens):
    """
    绘制EOT token与其他token的注意力分布
    """
    # 提取融合token对其他token的注意力
    attn_fusion = attn_weights[fusion_idx, :num_text_tokens]

    plt.figure(figsize=(12, 2))
    sns.heatmap([attn_fusion], xticklabels=tokens[:num_text_tokens], yticklabels=[f'Fusion Token (Layer {layer_num +1} Head {head_num +1})'], cmap='viridis')
    plt.title(f'Fusion Token Attention - Layer {layer_num + 1} Head {head_num + 1}')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # 构建保存文件名
    output_file = os.path.join(output_dir, f"{model_name}_layer{layer_num +1}_head{head_num +1}_fusion_token_attention.png")
    plt.savefig(output_file)
    plt.close()

    del attn_weights

if __name__ == "__main__":
    # 参数设置
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir = "vis_Maba_EOT_text_attention"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 定义模型参数类
    class Args:
        pretrain_choice = 'ViT-B/16'
        img_size = (224, 224)
        stride_size = 16
        temperature = 0.07
        loss_names = 'itc+id+mlm'
        vocab_size = 49408
        cmt_depth = 2
        id_loss_weight = 1.0
        mlm_loss_weight = 1.0
        context_length = 77  # 添加 context_length
        output_dir = 'logs/CUHK-PEDES/20250108_105420_iira'  # 确保与图像可视化代码一致

    args = Args()

    # 构建并加载自定义模型
    model = build_model(args).to(device).eval()
    checkpointer = Checkpointer(model)
    checkpoint_path = os.path.join(args.output_dir, 'best.pth')  # 替换为您的检查点路径
    checkpointer.load(f=checkpoint_path)

    # 打印模型结构以确认Transformer的位置
    print("模型结构：")
    for name, module in model.named_modules():
        print(name)

    # 注册文本注意力钩子
    hook_outputs = {}
    # 根据您的模型调整layers_to_hook
    layers_to_hook = list(range(len(model.base_model.transformer.resblocks)))  # 例如 [0, 1, ..., 11]
    print(f"Registering hooks for layers: {layers_to_hook}")
    register_text_attn_weights_hooks(model, layers_to_hook)

    # 初始化模型的分词器
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')

    # 文本预处理
    text = "a man in blue and orange sneakers dark knee length shorts and white tshirt is holding a plastic bag with something red inside and strolling past an escalator"
    encoded = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=args.context_length)
    text_tensor = encoded['input_ids'].to(device)  # [1, seq_len]
    tokens = tokenizer.convert_ids_to_tokens(text_tensor[0])
    num_text_tokens = (text_tensor != tokenizer.pad_token_id).sum(dim=1).item()

    print(f"Number of text tokens (non-padded): {num_text_tokens}")

    # 准备文本批次数据
    batch = {
        'texts': text_tensor  # 添加文本数据
    }

    # 前向传播以捕获注意力权重
    with torch.no_grad():
        try:
            # 根据您的模型调整前向传播方法
            outputs = model.encode_text(batch['texts'])
            print("encode_text completed successfully.")
        except Exception as e:
            print(f"Error during encode_text: {e}")
            # 可选：打印形状以进行调试
            print(f"text_tensor shape: {text_tensor.shape}")
            if hasattr(model.base_model, 'positional_embedding'):
                print(f"positional_embedding shape: {model.base_model.positional_embedding.shape}")
            else:
                print("model.base_model.positional_embedding not found.")
            raise e

    # 确定EOT token的索引
    fusion_idx = text_tensor.argmax(dim=-1).item()
    print(f"Fusion (EOT) token index: {fusion_idx}")
    fusion_token = tokens[fusion_idx]
    print(f"Fusion (EOT) token: {fusion_token}")

    # 提取并可视化文本注意力权重
    for name, tensor in hook_outputs.items():
        if name.startswith('text_block'):
            # 检查注意力权重的维度
            if tensor.ndim == 4:
                # [batch_size, num_heads, seq_len, seq_len]
                num_heads = tensor.shape[1]
                seq_len = tensor.shape[2]
                print(f"Visualizing attention weights for {name}: num_heads={num_heads}, seq_len={seq_len}")
                for head in range(num_heads):
                    attn = tensor[0, head].cpu().numpy()  # [seq_len, seq_len]
                    if attn.ndim == 2:
                        # 绘制全局注意力图
                        plot_text_attention_map(
                            attn,
                            tokens,
                            name,
                            output_dir,
                            layer_num=int(name.split('_')[-1]) - 1,
                            head_num=head,
                            num_text_tokens=num_text_tokens
                        )
                        # 绘制EOT token的注意力分布
                        plot_fusion_token_attention(
                            attn,
                            tokens,
                            name,
                            output_dir,
                            layer_num=int(name.split('_')[-1]) - 1,
                            head_num=head,
                            fusion_idx=fusion_idx,
                            num_text_tokens=num_text_tokens
                        )
                    else:
                        print(f"Unexpected attn shape for {name}, head {head}: {attn.shape}")
            elif tensor.ndim == 3:
                # [batch_size, seq_len, seq_len]，假设是单头
                seq_len = tensor.shape[1]
                attn = tensor[0].cpu().numpy()  # [seq_len, seq_len]
                if attn.ndim == 2:
                    # 绘制全局注意力图
                    plot_text_attention_map(
                        attn,
                        tokens,
                        name,
                        output_dir,
                        layer_num=int(name.split('_')[-1]) - 1,
                        head_num=0,
                        num_text_tokens=num_text_tokens
                    )
                    # 绘制EOT token的注意力分布
                    plot_fusion_token_attention(
                        attn,
                        tokens,
                        name,
                        output_dir,
                        layer_num=int(name.split('_')[-1]) - 1,
                        head_num=0,
                        fusion_idx=fusion_idx,
                        num_text_tokens=num_text_tokens
                    )
                else:
                    print(f"Unexpected attn shape for {name}: {attn.shape}")
            else:
                print(f"Unexpected tensor shape for {name}: {tensor.shape}")

    print("文本注意力可视化完成。")