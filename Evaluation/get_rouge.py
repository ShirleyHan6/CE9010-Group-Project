from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from pyrouge import Rouge155

gold_dir = './gold'
system_dir = "./system"

r = Rouge155()

r.system_dir = system_dir
r.model_dir = gold_dir

r.system_filename_pattern = '([a-zA-Z0-9]*).model'
r.model_filename_pattern = '#ID#.story'

output = r.convert_and_evaluate(rouge_args='-e /Users/shirley/Desktop/evaluation-master/ROUGE-RELEASE-1.5.5/data -a -c 95 -n 4 -w 1.2')
# print output
output_dict = r.output_to_dict(output)

print(output_dict["rouge_1_recall"])
print(output_dict["rouge_2_recall"])
print(output_dict["rouge_l_recall"])

print(output_dict["rouge_1_precision"])
print(output_dict["rouge_2_precision"])
print(output_dict["rouge_l_precision"])

print(output_dict["rouge_1_f_score"])
print(output_dict["rouge_2_f_score"])
print(output_dict["rouge_l_f_score"])


