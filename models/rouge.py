import numpy as np
from utils.dict import index2sentence
from rouge import FilesRouge
from rouge import Rouge


def rouge_score(filename_gold, filename_result):
    files_rouge = FilesRouge(filename_result, filename_gold)
    scores = files_rouge.get_scores(avg=True)
    return scores


def write_rouge(filename, score, epoch):
    rouge_1 = 'ROUGE-1 f ' + str(score['rouge-1']['f']) + ' p ' \
              + str(score['rouge-1']['p']) + ' r ' \
              + str(score['rouge-1']['r'])
    rouge_2 = 'ROUGE-2 f ' + str(score['rouge-2']['f']) + ' p ' \
              + str(score['rouge-2']['p']) + ' r ' \
              + str(score['rouge-2']['r'])
    rouge_l = 'ROUGE-l f ' + str(score['rouge-l']['f']) + ' p ' \
              + str(score['rouge-l']['p']) + ' r ' \
              + str(score['rouge-l']['r'])
    rouge = [rouge_1, rouge_2, rouge_l]
    with open(filename, 'a') as f:
        a = 'EPOCH ' + str(epoch) + '\n'
        f.write(a)
        f.write('\n'.join(rouge))
        f.write('\n\n')


def rouge_l(result, gold, idx2word):
    """
    :param result: (batch, len)
    :param gold: (batch, len)
    :return: Rouge-L
    """
    scores = 0
    rouge = Rouge()
    result = result.cpu()
    gold = gold.cpu()
    result = np.array(result)
    gold = np.array(gold)
    for i in range(result.shape[0]):
        hyp = index2sentence(list(result[i]), idx2word)
        hyp = ' '.join(hyp)
        ref = index2sentence(list(gold[i]), idx2word)
        ref = ' '.join(ref)

        scores += rouge.get_scores(hyp, ref)[0]['rouge-l']['f']

    r_l = scores/result.shape[0]
    return r_l