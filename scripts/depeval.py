# -encoding:utf8-
import sys
import depio
import re
from operator import itemgetter
from pprint import pprint

g_reP = re.compile(r"^[,?!:;]$|^-LRB-$|^-RRB-$|^[.]+$|^[`]+$|^[']+$|^（$|^）$|^、$|^。$|^！$|^？$|^…$|^，$|^；$|^／$|^：$|^“$|^”$|^「$|^」$|^『$|^』$|^《$|^》$|^一一$")  # ^\($|^\)$")

unlabeled_func = itemgetter(1, 6, 8)
labeled_func = itemgetter(1, 6, 7, 8)

def replace_head_idx_with_form(rows):
    retval = []
    for r in rows:
        if r[1] == '0':
            if len(r) == 4:
                retval.append(tuple([r[0], 'ROOT', 'ROOT'] + list(r[3:])))
            else:
                retval.append(tuple([r[0], 'ROOT'] + list(r[2:])))
        else:
            retval.append(tuple([r[0], rows[int(r[1])-1][0]] + list(r[2:])))
    return retval

filt = lambda v: not g_reP.match(v[0])
filt_unmapped = lambda v: not g_reP.match(v[1])


def eval(output, reference):
    unlabeled_out = set(filter(filt, replace_head_idx_with_form(list(map(unlabeled_func, output)))))
    labeled_out = set(filter(filt, replace_head_idx_with_form(list(map(labeled_func, output)))))
    unlabeled_ref = set(filter(filt, replace_head_idx_with_form(list(map(unlabeled_func, reference)))))
    labeled_ref = set(filter(filt, replace_head_idx_with_form(list(map(labeled_func, reference)))))
    correct_head = len(unlabeled_out.intersection(unlabeled_ref))
    correct_label = len(labeled_out.intersection(labeled_ref))
    incorrect_head = len(unlabeled_out.difference(unlabeled_ref))
    incorrect_label = len(labeled_out.difference(labeled_ref))
    missing_head = len(unlabeled_ref.difference(unlabeled_out))
    missing_label = len(labeled_ref.difference(labeled_out))
    total_uem = 1 if unlabeled_out == unlabeled_ref else 0
    total = len(list(filter(filt_unmapped, output)))
    # for index, word in enumerate(output):
    #ref_word ref_word = reference[index]
    #assert assert word[1] == ref_word[1]
    #if if g_reP.match( word[1] ) :
    #continue continue
    #if if word[6] == ref_word[6]:
    #correct_head correct_head += 1
    #if if word[7] == ref_word[7]:
    #correct_label correct_label += 1
    #else else:
    #total_uem total_uem = 0
    #total total += 1
    return correct_head, correct_label, total, total_uem, incorrect_head, incorrect_label, missing_head, missing_label, len(list(filter(filt_unmapped, reference)))

if __name__ == '__main__':
    file_output = list(depio.depread(sys.argv[1]))
    file_ref = list(depio.depread(sys.argv[2]))
    total_sent = 0
    total_uem = 0
    total = 0
    total_gold = 0
    correct_head = 0
    incorrect_head = 0
    correct_label = 0
    incorrect_label = 0
    missing_head = 0
    missing_label = 0
    for ref, output in zip(file_ref, file_output):
        # ref = file_ref.next()
        ret = eval(output, ref)
        correct_head += ret[0]
        correct_label += ret[1]
        total += ret[2]
        total_uem += ret[3]
        incorrect_head += ret[4]
        incorrect_label += ret[5]
        missing_head += ret[6]
        missing_label += ret[7]
        total_sent += 1
        total_gold += ret[8]
    unlabeled_precision = float(correct_head)/total
    unlabeled_recall = float(correct_head)/total_gold
    labeled_precision = float(correct_label)/total
    labeled_recall = float(correct_label)/total_gold
    f1_unlabeled = 2*(unlabeled_precision * unlabeled_recall)/(unlabeled_precision + unlabeled_recall)
    f1_labeled = 2*(labeled_precision * labeled_recall)/(labeled_precision + labeled_recall)
    print('UAS', '#correctlabel', 'LAS', '#UEM', 'UEM', 'F1unlabeled', 'F1labeled', 'total')
    # print float(correct_head)/total, correct_label, float(correct_label)/total, total_uem, float(total_uem)/total_sent, f1_unlabeled, f1_labeled, total
    print(unlabeled_recall, correct_label, float(correct_label)/total, total_uem, labeled_recall, f1_unlabeled, f1_labeled, total)