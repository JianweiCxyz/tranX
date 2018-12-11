from __future__ import print_function
import pickle
import sys
import json
import argparse
import conala_eval.bleu_score as bleu_score
from conala_eval.conala_eval import tokenize_for_bleu_eval


def correct_hyps(hyps):
    """
    Get the 1-indexed correct hypothesis if any else return 0
    """
    for i, hyp in enumerate(hyps[1:]):
        if hyp.correct:
            return i
    return 0


def eval_bleu(decodes):
    total = len(decodes)
    len_candidates = len(min(decodes, key=lambda x: len(x))) - 1
    decodes = list(filter(lambda x: len(x) > 2, decodes))
    size = len(decodes)
    print("Got {} test examples with at least {} hyps each, {} examples has no predictions.".format(
        size, len_candidates, total - size))
    ref_list = [[tokenize_for_bleu_eval(hyps[0].tgt_code)] for hyps in decodes]
    top_hyps= [tokenize_for_bleu_eval(hyps[1].code) for hyps in decodes]
    bleu_exact= bleu_score.compute_bleu(ref_list, top_hyps, smooth=False)
    print("BLEU for top prediction {:.5f}".format(
        bleu_exact[0]), file=sys.stderr)
    exact_matches= sum([hyps[1].correct for hyps in decodes])
    print("Exact matches: {}/{} = {:.2%}".
          format(exact_matches, size, exact_matches / float(size)), file=sys.stderr)
    oracle_idx = [correct_hyps(hyps) for hyps in decodes]
    oracle_matches = sum([idx > 0 for idx in oracle_idx])
    print("Oracle matches: {}/{} = {:.2%}".
          format(oracle_matches, size, oracle_matches / float(size)), file=sys.stderr)
    oracle_hyps = [tokenize_for_bleu_eval(
        hyps[max(1, i)].code) for hyps, i in zip(decodes, oracle_idx)]
    bleu_oracle = bleu_score.compute_bleu(ref_list, oracle_hyps, smooth=False)
    print("BLEU for oracle prediction {:.5f}".format(
        bleu_oracle[0]), file=sys.stderr)
    return dict(exact_matches=exact_matches, oracle_matches=oracle_matches,
                bleu_exact=bleu_exact, bleu_oracle=bleu_oracle)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("decode_file", type=str)
    args = parser.parse_args()
    decodes = pickle.load(open(args.decode_file))
    eval_bleu(decodes)
    oracle_idx = [correct_hyps(hyps) for hyps in decodes]
    with open(args.decode_file + ".json", 'w') as f:
        data = []
        for hyps, correct_idx in zip(decodes, oracle_idx):
            obj = {}
            obj["intent"] = ' '.join(hyps[0].src_sent)
            obj["gold"] = hyps[0].tgt_code
            obj["candidates"] = [h.code for h in hyps[1:]]
            obj["correct_idx"] = correct_idx
            data.append(obj)
        json.dump(data, f, indent=4)
