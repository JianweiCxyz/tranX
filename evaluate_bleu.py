from __future__ import print_function
import pickle
import sys
import re
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


def eval_bleu(decodes, missing_exs):
    total = len(decodes) + len(missing_exs)
    len_candidates = len(min(decodes, key=lambda x: len(x))) - 1
    decodes = list(filter(lambda x: len(x) > 2, decodes))
    size = len(decodes) + len(missing_exs)
    print("Got {} test examples with at least {} hyps each, {} examples has no predictions.".format(
        size, len_candidates, total - size))
    ref_list = [[tokenize_for_bleu_eval(hyps[0].tgt_code)] for hyps in decodes]
    ref_list += [[k] for k in missing_exs]
    top_hyps = [tokenize_for_bleu_eval(hyps[1].code) for hyps in decodes]
    top_hyps += [[] for _ in range(len(missing_exs))]
    print("len(ref_list) = ", len(ref_list))
    print("len(top_hyps) = ", len(top_hyps))
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
    oracle_hyps += [[] for _ in range(len(missing_exs))]
    bleu_oracle = bleu_score.compute_bleu(ref_list, oracle_hyps, smooth=False)
    print("BLEU for oracle prediction {:.5f}".format(
        bleu_oracle[0]), file=sys.stderr)
    return dict(exact_matches=exact_matches, oracle_matches=oracle_matches,
                bleu_exact=bleu_exact, bleu_oracle=bleu_oracle)

def remap_code(str_map, code):
    for k, v in str_map.items():
        code = code.replace(k, v)
    return code

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("decode_file", type=str)
    parser.add_argument("--test_file", type=str, default="data/conala/conala-corpus/conala-test.json")
    args = parser.parse_args()
    test_data = json.load(open(args.test_file))
    decodes = pickle.load(open(args.decode_file))
    all_code = set([hyps[0].meta['raw_code'] for hyps in decodes])
    missing_exs = {}
    for entry in test_data:
        if entry["snippet"] not in all_code:
            missing_exs[entry["rewritten_intent"]] = entry["snippet"]
    eval_bleu(decodes, missing_exs)
    oracle_idx = [correct_hyps(hyps) for hyps in decodes]
    with open(args.decode_file + ".json", 'w') as f:
        data = []
        for hyps, correct_idx in zip(decodes, oracle_idx):
            obj = {}
            str_map = {v:k for k, v in hyps[0].meta['str_map'].items()}
            obj["intent"] = ' '.join(hyps[0].src_sent)
            obj["gold"] = hyps[0].meta['raw_code']
            obj["candidates"] = [remap_code(str_map, h.code) for h in hyps[1:]]
            obj["correct_idx"] = correct_idx
            obj["str_map"] = str_map
            data.append(obj)
        json.dump(data, f, indent=4)
