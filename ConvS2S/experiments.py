import spacy as spacy
import torch
from matplotlib import pyplot as plt, ticker

from utils import *
from model import *
from dataloader import *
from transform import TextTransform
from train import train



# load configurations
CONF = load_conf()
# add special characters to config
CONF["special_tokens"] = {
    CONF["unk_token"]: 0,
    CONF["pad_token"]: 1,
    CONF["sos_token"]: 2,
    CONF["eos_token"]: 3,
}

# set seed
set_seed(CONF['seed'])

# get device
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load tokenizers
src_tokenizer = load_tokenizer(CONF['tokenizer'], CONF['src'])
tgt_tokenizer = load_tokenizer(CONF['tokenizer'], CONF['tgt'])

# load datset
train_data, valid_data, test_data = load_multi30k(CONF['src'], CONF['tgt'])

# load vocab
src_vocab = get_vocab( get_src_iter(train_data), src_tokenizer, CONF)
tgt_vocab = get_vocab( get_tgt_iter(train_data), tgt_tokenizer, CONF)

# data transforms
src_transform = TextTransform(src_tokenizer, src_vocab, CONF)
tgt_transform = TextTransform(tgt_tokenizer, tgt_vocab, CONF)

# get data loaders
train_dataloader = get_dataloader(train_data, src_transform, tgt_transform, CONF)
valid_dataloader = get_dataloader(valid_data, src_transform, tgt_transform, CONF)
test_dataloader = get_dataloader(test_data, src_transform, tgt_transform, CONF)


# create model
encoder = Encoder(input_dim = len(src_vocab),
                  emb_dim = CONF['embed_size'],
                  hid_dim = CONF['hidden_size'],
                  n_layers = CONF['encoder_layers'],
                  kernel_size = CONF['kernel_size'],
                  dropout = CONF['dropout'],
                  device = DEVICE,
                  max_length=CONF['max_length'],
)
decoder = Decoder(output_dim = len(tgt_vocab),
                  emb_dim = CONF['embed_size'],
                  hid_dim = CONF['hidden_size'],
                  n_layers = CONF['decoder_layers'],
                  kernel_size = CONF['kernel_size'],
                  dropout = CONF['dropout'],
                  tgt_pad_idx = CONF["special_tokens"][CONF['pad_token']],
                  device = DEVICE,
                  max_length=CONF['max_length'],
)
convs2s = Seq2Seq(encoder, decoder)

# create optimizer and criterion
optimizer = torch.optim.Adam(convs2s.parameters())
criterion = torch.nn.CrossEntropyLoss(ignore_index = CONF["special_tokens"][CONF['pad_token']])

# train
train(  convs2s,
        criterion,
        train_dataloader,
        valid_dataloader,
        optimizer,
        CONF["epochs"],
        CONF["clip"],
)

# with torch.no_grad():
#
#     for idx, data in enumerate(test_dataloader):
#         if idx > 0:
#             break
#         src, dst = data
#         y_pred = convs2s(src, dst)
#         print(idx)
#         print(y_pred)
#         print("**************")
#         print(dst)

def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):
    model.eval()

    if isinstance(sentence, str):
        nlp = spacy.load('en')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_conved, encoder_combined = model.encoder(src_tensor)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, encoder_conved, encoder_combined)

        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attention


def display_attention(sentence, translation, attention):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    attention = attention.squeeze(0).cpu().detach().numpy()

    cax = ax.matshow(attention, cmap='bone')

    ax.tick_params(labelsize=15)
    ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>'],
                       rotation=45)
    ax.set_yticklabels([''] + translation)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()


def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences.

    Defined in :numref:`sec_machine_translation`"""
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad

#@save
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """序列到序列模型的预测"""
    # 在预测时将net设置为评估模式
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['</s>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<s>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['</s>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


# src = ['a', 'little', 'girl', 'climbing', 'into', 'a', 'wooden', 'playhouse', '.']
# tgt = ['ein', 'kleines', 'mädchen', 'klettert', 'in', 'ein', 'spielhaus', 'aus', 'holz', '.']
# translation, attention = translate_sentence(src, src_vocab, tgt_vocab, convs2s, torch.device("cpu"))

engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = predict_seq2seq(
        convs2s, eng, src_vocab, tgt_vocab, 10, DEVICE)
    print(f'{eng} => {translation}')

