
# coding: utf-8

# In[ ]:


from rouge import Rouge 

rouge = Rouge()

test_title_text = testdf.issue_title.tolist()
test_body_text = testdf.body.tolist()
predict_title_text = [None]*len(test_body_text)

rouge_1_p, rouge_1_r, rouge_1_f, rouge_2_p, rouge_2_r, rouge_2_f, rouge_l_p, rouge_l_f, rouge_l_r = 0, 0, 0, 0, 0, 0, 0, 0, 0
for i in range(len(test_body_text)):
    exm, predict_title_text[i] = seq2seq_inf.generate_issue_title(raw_input_text = test_body_text[i])
    scores = rouge.get_scores(predict_title_text[i], test_title_text[i])
    rouge_1_p = rouge_1_p + scores[0]['rouge-1']['p']
    rouge_1_r = rouge_1_r + scores[0]['rouge-1']['r']
    rouge_1_f = rouge_1_f + scores[0]['rouge-1']['f']

    rouge_2_p = rouge_2_p + scores[0]['rouge-2']['p']
    rouge_2_r = rouge_2_r + scores[0]['rouge-2']['r']
    rouge_2_f = rouge_2_f + scores[0]['rouge-2']['f']

    rouge_l_p = rouge_l_p + scores[0]['rouge-l']['p']
    rouge_l_r = rouge_l_r + scores[0]['rouge-l']['r']
    rouge_l_f = rouge_l_f + scores[0]['rouge-l']['f']

print("ROUGE-1:", rouge_1_f)
print("ROUGE-2:",rouge_2_f)
print("ROUGE-l:",rouge_l_f)
print("Average of ROUGE-1, ROUGE-2 and ROUGE-l: ", (rouge_1_f + rouge_2_f + rouge_l_f)/3)

